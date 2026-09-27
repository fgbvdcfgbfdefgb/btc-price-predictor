"""
Publish the working tree to GitHub as a normal incremental commit.

This is the everyday publish path. It never rewrites history, so anyone with
the repo cloned gets a clean `git pull` fast-forward:

    export GITHUB_TOKEN=github_pat_xxxxxxxx
    python3 scripts/publish.py -m "what changed"

How it stays correct even though this sandbox's .git is not durable: rather
than trusting the local history, it fetches the remote, points HEAD at
origin/<branch> with `git reset --mixed` (which leaves the working tree
untouched), and commits the working tree on top. The result is always a
fast-forward for everyone else.

Use scripts/wipe_and_republish.py only for a deliberate nuke-and-replace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"

# Large artifacts that live as release assets rather than in the git tree.
RELEASE_ASSETS = [
    "data/processed/btc_1m.parquet",
    "data/processed/btc_1m.csv.gz",
    "checkpoints/best_model.pt",
]


def sh(cmd, check=True, env=None, cwd=ROOT):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, env=e, text=True,
                          capture_output=True, check=check)


def api(method, path, token, body=None, raw=None, ctype=None, url=None):
    req = urllib.request.Request(url or f"{API}{path}", method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    data = None
    if raw is not None:
        data = raw
        req.add_header("Content-Type", ctype or "application/octet-stream")
    elif body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data) as r:
            txt = r.read().decode() or "{}"
            return r.status, json.loads(txt)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--message", default="Update")
    p.add_argument("--repo", default="btc-price-predictor")
    p.add_argument("--owner", default=None)
    p.add_argument("--branch", default="main")
    p.add_argument("--tag", default="dataset-v1")
    p.add_argument("--assets", action="store_true",
                   help="also refresh release assets whose size changed")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("set GITHUB_TOKEN first (never hardcode it)", file=sys.stderr)
        return 2

    st, me = api("GET", "/user", token)
    if st != 200:
        print(f"token rejected ({st})", file=sys.stderr)
        return 2
    owner = a.owner or me["login"]
    repo = f"{owner}/{a.repo}"
    print(f"repo   : {repo}")

    # ---- make sure the remote exists ------------------------------------- #
    st, _ = api("GET", f"/repos/{repo}", token)
    if st == 404:
        print("creating repository")
        api("POST", "/user/repos", token, {"name": a.repo, "private": False})

    askpass = ROOT / ".git" / "_askpass.sh"
    askpass.parent.mkdir(parents=True, exist_ok=True)
    askpass.write_text('#!/bin/sh\ncase "$1" in\n'
                       '*Username*) echo "$GIT_USER";;\n'
                       '*) echo "$GITHUB_TOKEN";;\nesac\n')
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = {"GIT_ASKPASS": str(askpass), "GIT_USER": owner,
           "GITHUB_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"}

    try:
        if not (ROOT / ".git").exists():
            sh(["git", "init", "-q"])
        sh(["git", "remote", "remove", "origin"], check=False)
        sh(["git", "remote", "add", "origin",
            f"https://github.com/{repo}.git"])
        sh(["git", "config", "user.email", "bot@local"], check=False)
        sh(["git", "config", "user.name", "publisher"], check=False)

        # Rebase our view onto whatever the remote currently is, WITHOUT
        # touching the working tree. --mixed moves HEAD + index only.
        f = sh(["git", "fetch", "-q", "origin", a.branch], check=False, env=env)
        remote_exists = f.returncode == 0
        sh(["git", "checkout", "-q", "-B", a.branch], check=False)
        if remote_exists:
            sh(["git", "reset", "--mixed", "-q", f"origin/{a.branch}"])
            base = sh(["git", "rev-parse", "--short",
                       f"origin/{a.branch}"]).stdout.strip()
            print(f"base   : origin/{a.branch} @ {base}")
        else:
            print(f"base   : none (new branch {a.branch})")

        sh(["git", "add", "-A"])
        staged = sh(["git", "diff", "--cached", "--name-status"]).stdout.strip()
        if not staged:
            print("\nnothing changed - remote already matches the working tree")
            return 0

        n = len(staged.splitlines())
        print(f"\n{n} change(s):")
        for line in staged.splitlines()[:40]:
            print(f"    {line}")
        if n > 40:
            print(f"    ... and {n - 40} more")

        if a.dry_run:
            sh(["git", "reset", "-q"], check=False)
            print("\n(dry run - nothing pushed)")
            return 0

        sh(["git", "commit", "-q", "-m", a.message])
        new = sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        print(f"\ncommitted {new}, pushing (fast-forward, no force)...")
        push = sh(["git", "push", "-u", "origin", a.branch],
                  check=False, env=env)
        out = (push.stdout or "") + (push.stderr or "")
        print("    " + out.strip().replace("\n", "\n    "))
        if push.returncode != 0:
            print("push failed - the remote moved; re-run to rebase onto it",
                  file=sys.stderr)
            return 5
    finally:
        askpass.unlink(missing_ok=True)

    # ---- optional release-asset refresh ----------------------------------- #
    if a.assets:
        st, rel = api("GET", f"/repos/{repo}/releases/tags/{a.tag}", token)
        if st != 200:
            st, rel = api("POST", f"/repos/{repo}/releases", token,
                          {"tag_name": a.tag, "name": f"BTC dataset {a.tag}",
                           "body": "Dataset and champion checkpoint."})
        if st in (200, 201):
            have = {x["name"]: x for x in rel.get("assets", [])}
            up = rel["upload_url"].split("{")[0]
            for rp in RELEASE_ASSETS:
                fp = ROOT / rp
                if not fp.exists():
                    continue
                cur = have.get(fp.name)
                if cur and cur["size"] == fp.stat().st_size:
                    print(f"  asset unchanged, skipping {fp.name}")
                    continue
                if cur:
                    api("DELETE", f"/repos/{repo}/releases/assets/{cur['id']}",
                        token)
                ctype = (mimetypes.guess_type(fp.name)[0]
                         or "application/octet-stream")
                print(f"  uploading {fp.name} "
                      f"({fp.stat().st_size / 1e6:.1f} MB)...")
                s2, _ = api("POST", "", token, raw=fp.read_bytes(),
                            ctype=ctype, url=f"{up}?name={fp.name}")
                print(f"    {'ok' if s2 in (200, 201) else 'FAILED ' + str(s2)}")

    # ---- verify remote == working tree ------------------------------------ #
    st, c = api("GET", f"/repos/{repo}/commits/{a.branch}", token)
    st, tree = api("GET",
                   f"/repos/{repo}/git/trees/{c['commit']['tree']['sha']}"
                   "?recursive=1", token)
    remote = {x["path"] for x in tree["tree"] if x["type"] == "blob"}
    local = set(sh(["git", "ls-files"]).stdout.split())
    print(f"\nverification: {len(remote)} remote blobs vs {len(local)} local")
    if remote == local:
        print("  exact match - repo mirrors the working tree")
    else:
        for x in sorted(local - remote):
            print(f"  missing on remote: {x}")
        for x in sorted(remote - local):
            print(f"  extra on remote  : {x}")
    print(f"\nhttps://github.com/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
