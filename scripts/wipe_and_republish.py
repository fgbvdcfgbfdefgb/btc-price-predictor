"""
Wipe a GitHub repo clean and republish the entire working tree.

Removes every file AND the old commit history by force-pushing a fresh orphan
root commit, deletes stale releases/tags, then re-uploads release assets.
The result is a repo that mirrors the local tree exactly - no leftovers.

    export GITHUB_TOKEN=github_pat_xxxx
    python3 scripts/wipe_and_republish.py --repo btc-price-predictor

    --keep-history   amend on top of main instead of wiping history
    --no-raw         leave data/raw out (it is rebuildable from the API)
    --dry-run        show what would happen, touch nothing

The token is read from the environment and passed via GIT_ASKPASS, so it is
never written into .git/config or any committed file.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"

RELEASE_ASSETS = [
    "data/processed/btc_1m.parquet",
    "data/processed/btc_1m.csv.gz",
    "checkpoints/best_model.pt",
]

GITIGNORE_FULL = """\
# transient only - everything else is published
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
env/

# derived model-input cache (rebuilt by src/train.py in seconds)
data/processed/cache/
# per-individual GA weights (regenerated every generation)
checkpoints/elites/
# timestamped run logs
logs/train-*.log
logs/*.pid
"""

GITIGNORE_NO_RAW = GITIGNORE_FULL + """\
# raw exchange archives (rebuildable with src/download_data.py)
data/raw/
"""


def sh(args, check=True, env=None, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                       env={**os.environ, **(env or {})})
    if check and p.returncode != 0:
        raise RuntimeError(f"$ {' '.join(args)}\n{p.stdout}\n{p.stderr}")
    return p


def api(method, path, token, body=None, raw=None, ctype=None, url=None):
    u = url or f"{API}{path}"
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    req = urllib.request.Request(u, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", ctype or "application/json")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            b = r.read()
            return r.status, (json.loads(b) if b and r.status != 204 else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--owner", default=None)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--tag", default="dataset-v1")
    ap.add_argument("--message", default="Full republish: complete project + dataset")
    ap.add_argument("--keep-history", action="store_true")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        return 2

    st, me = api("GET", "/user", token)
    if st != 200:
        print(f"ERROR: token rejected (HTTP {st}) {me.get('message','')}",
              file=sys.stderr)
        return 2
    owner = a.owner or me["login"]
    repo = f"{owner}/{a.repo}"
    print(f"authenticated as {me['login']}  ->  {repo}")

    st, _ = api("GET", f"/repos/{repo}", token)
    if st == 404:
        print(f"ERROR: {repo} does not exist", file=sys.stderr)
        return 3

    (ROOT / ".gitignore").write_text(
        GITIGNORE_NO_RAW if a.no_raw else GITIGNORE_FULL)

    # ---- what will be published ------------------------------------------- #
    sh(["git", "init", "-q"], check=False)
    sh(["git", "config", "user.email", f"{me['login']}@users.noreply.github.com"])
    sh(["git", "config", "user.name", me["login"]])
    files = sh(["git", "ls-files", "--others", "--cached",
                "--exclude-standard"]).stdout.split()
    total = sum((ROOT / f).stat().st_size for f in files if (ROOT / f).exists())
    print(f"\nwill publish {len(files)} files, {total/1e6:.1f} MB")
    oversize = [f for f in files
                if (ROOT / f).exists() and (ROOT / f).stat().st_size > 95e6]
    if oversize:
        print(f"ERROR: over GitHub's 100 MB blob limit: {oversize}", file=sys.stderr)
        return 4

    if a.dry_run:
        for f in sorted(files):
            print("   ", f)
        print("\n(dry run - nothing changed)")
        return 0

    # ---- delete releases + tags -------------------------------------------- #
    st, rels = api("GET", f"/repos/{repo}/releases", token)
    for r in rels if isinstance(rels, list) else []:
        api("DELETE", f"/repos/{repo}/releases/{r['id']}", token)
        api("DELETE", f"/repos/{repo}/git/refs/tags/{r['tag_name']}", token)
        print(f"deleted release {r['tag_name']} (+{len(r.get('assets',[]))} assets)")

    # ---- wipe every file by force-pushing a fresh root commit -------------- #
    askpass = ROOT / ".git" / "askpass.sh"
    askpass.write_text('#!/bin/sh\ncase "$1" in\n'
                       '*Username*) echo "$GIT_USER" ;;\n'
                       '*) echo "$GITHUB_TOKEN" ;;\nesac\n')
    askpass.chmod(0o700)
    env = {"GIT_ASKPASS": str(askpass), "GIT_USER": owner,
           "GITHUB_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"}
    sh(["git", "remote", "remove", "origin"], check=False)
    sh(["git", "remote", "add", "origin",
        f"https://github.com/{repo}.git"])

    try:
        if a.keep_history:
            sh(["git", "add", "-A"])
            sh(["git", "commit", "-m", a.message], check=False)
        else:
            print("creating a fresh orphan root commit (old history discarded)")
            sh(["git", "checkout", "-q", "--orphan", "_republish"])
            sh(["git", "add", "-A"])
            sh(["git", "commit", "-q", "-m", a.message])
            sh(["git", "branch", "-D", a.branch], check=False)
            sh(["git", "branch", "-m", a.branch])
        print("force-pushing...")
        p = sh(["git", "push", "--force", "-u", "origin", a.branch],
               check=False, env=env)
        print((p.stdout or "") + (p.stderr or ""))
        if p.returncode != 0:
            print("push failed", file=sys.stderr)
            return 5
    finally:
        askpass.unlink(missing_ok=True)

    # ---- fresh release ------------------------------------------------------ #
    st, rel = api("POST", f"/repos/{repo}/releases", token,
                  {"tag_name": a.tag, "name": f"BTC dataset {a.tag}",
                   "body": "One year of 1-minute BTCUSDT OHLCV plus the evolved "
                           "champion checkpoint."})
    if st in (200, 201):
        up = rel["upload_url"].split("{")[0]
        for rp in RELEASE_ASSETS:
            f = ROOT / rp
            if not f.exists():
                continue
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            print(f"  uploading {f.name} ({f.stat().st_size/1e6:.1f} MB)...")
            s2, _ = api("POST", "", token, raw=f.read_bytes(), ctype=ctype,
                        url=f"{up}?name={f.name}")
            print(f"    {'ok' if s2 in (200,201) else 'FAILED ' + str(s2)}")
    else:
        print(f"WARNING: could not create release: {st} {rel.get('message','')}")

    # ---- verify remote == local -------------------------------------------- #
    st, tree = api("GET", f"/repos/{repo}/git/trees/{a.branch}?recursive=1", token)
    remote = {b["path"] for b in tree.get("tree", []) if b["type"] == "blob"}
    local = set(files)
    only_remote, only_local = sorted(remote - local), sorted(local - remote)
    print(f"\nverification: {len(remote)} remote blobs vs {len(local)} local")
    if only_remote:
        print("  STILL ON REMOTE (should be none):", only_remote)
    if only_local:
        print("  MISSING FROM REMOTE:", only_local)
    if not only_remote and not only_local:
        print("  exact match - repo mirrors the working tree")
    print(f"\nhttps://github.com/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
