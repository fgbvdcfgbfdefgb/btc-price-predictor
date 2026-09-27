"""
Publish the dataset + code to GitHub.

The token is read from the GITHUB_TOKEN environment variable and is never
written to disk, never logged, and never embedded in the git remote URL that
gets stored in .git/config.

    export GITHUB_TOKEN=github_pat_xxxxxxxx      # fine-grained PAT
    python src/upload_to_github.py --repo btc-price-predictor

Large files
-----------
  --mode lfs      track *.parquet / *.csv.gz / *.pt with Git LFS  (default)
  --mode release  keep the repo light and attach the dataset to a GitHub
                  Release instead - no LFS bandwidth quota, better for
                  anything over ~50 MB
  --mode both
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"

LFS_PATTERNS = ["*.parquet", "*.csv.gz", "*.pt", "*.npy", "*.zip"]

GITIGNORE = """\
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
env/

# raw exchange archives - rebuildable with src/download_data.py
data/raw/
# derived model-input cache - rebuildable with src/train.py
data/processed/cache/
# per-individual weights from the GA
checkpoints/elites/
logs/*.log
"""


def sh(args: list[str], cwd: Path = ROOT, check: bool = True,
       env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                       env={**os.environ, **(env or {})})
    if check and p.returncode != 0:
        raise RuntimeError(f"$ {' '.join(args)}\n{p.stdout}\n{p.stderr}")
    return p


def api(method: str, path: str, token: str, body: dict | None = None,
        raw: bytes | None = None, ctype: str | None = None, url: str | None = None):
    u = url or f"{API}{path}"
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    req = urllib.request.Request(u, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", ctype or "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            b = r.read()
            return r.status, (json.loads(b) if b and r.status != 204 else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repository name")
    ap.add_argument("--owner", default=None, help="defaults to the token's user")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--mode", choices=["lfs", "plain", "release", "both"],
                    default="both")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="BTC 1-minute dataset + evolutionary forecaster")
    ap.add_argument("--tag", default="dataset-v1")
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("ERROR: GITHUB_TOKEN is not set.\n"
              "  export GITHUB_TOKEN=github_pat_xxxx\n"
              "Never paste a token into a file, a notebook cell, or a chat.",
              file=sys.stderr)
        return 2

    # ---- who am I ---------------------------------------------------------- #
    st, me = api("GET", "/user", token)
    if st != 200:
        print(f"ERROR: token rejected by GitHub (HTTP {st}). "
              f"{me.get('message','')}", file=sys.stderr)
        return 2
    owner = a.owner or me["login"]
    print(f"authenticated as {me['login']}  ->  target {owner}/{a.repo}")

    # ---- ensure the repo exists -------------------------------------------- #
    st, _ = api("GET", f"/repos/{owner}/{a.repo}", token)
    if st == 404:
        if a.owner and a.owner != me["login"]:
            st, r = api("POST", f"/orgs/{owner}/repos", token,
                        {"name": a.repo, "private": a.private})
        else:
            st, r = api("POST", "/user/repos", token,
                        {"name": a.repo, "private": a.private,
                         "description": "BTC 1-minute OHLCV dataset and an "
                                        "evolutionary multi-horizon forecaster"})
        if st not in (200, 201):
            print(f"ERROR creating repo: HTTP {st} {r.get('message','')}",
                  file=sys.stderr)
            return 3
        print(f"created {'private' if a.private else 'public'} repo {owner}/{a.repo}")
    else:
        print("repo already exists - pushing into it")

    # ---- local git --------------------------------------------------------- #
    (ROOT / ".gitignore").write_text(GITIGNORE)
    if not (ROOT / ".git").exists():
        sh(["git", "init", "-q"])
    sh(["git", "symbolic-ref", "HEAD", f"refs/heads/{a.branch}"], check=False)
    sh(["git", "config", "user.email", f"{me['login']}@users.noreply.github.com"])
    sh(["git", "config", "user.name", me["login"]])

    DATA_FILES = [ROOT / "data" / "processed" / "btc_1m.parquet",
                  ROOT / "data" / "processed" / "btc_1m.csv.gz"]
    GH_FILE_LIMIT = 95 * 1024 * 1024          # GitHub rejects blobs over 100 MB

    commit_data = a.mode in ("lfs", "plain", "both")
    make_release = a.mode in ("release", "both")
    if a.mode == "release":
        commit_data = False

    if commit_data and a.mode in ("lfs", "both"):
        if shutil.which("git-lfs"):
            sh(["git", "lfs", "install", "--local"])
            for pat in LFS_PATTERNS:
                sh(["git", "lfs", "track", pat])
            print(f"git-lfs tracking: {', '.join(LFS_PATTERNS)}")
        else:
            oversize = [f for f in DATA_FILES
                        if f.exists() and f.stat().st_size > GH_FILE_LIMIT]
            if oversize:
                print("WARNING: git-lfs missing and these exceed 100 MB -> "
                      "release assets only: "
                      + ", ".join(f.name for f in oversize))
                commit_data, make_release = False, True
            else:
                print("note: git-lfs not installed, but every data file is under "
                      "100 MB -> committing them directly to git")

    if not commit_data:
        # keep bulk data out of the git history entirely
        extra = "\ndata/processed/*.parquet\ndata/processed/*.csv.gz\n"
        (ROOT / ".gitignore").write_text(GITIGNORE + extra)

    sh(["git", "add", "-A"])
    p = sh(["git", "commit", "-m", a.message], check=False)
    if p.returncode != 0 and "nothing to commit" not in (p.stdout + p.stderr):
        print(p.stdout, p.stderr, file=sys.stderr)
        return 4
    print("committed")

    # Token goes in the credential helper's stdin, NOT into .git/config.
    remote = f"https://github.com/{owner}/{a.repo}.git"
    sh(["git", "remote", "remove", "origin"], check=False)
    sh(["git", "remote", "add", "origin", remote])
    askpass = ROOT / ".git" / "askpass.sh"
    askpass.write_text('#!/bin/sh\ncase "$1" in\n'
                       '*Username*) echo "$GIT_USER" ;;\n'
                       '*) echo "$GITHUB_TOKEN" ;;\nesac\n')
    askpass.chmod(0o700)
    env = {"GIT_ASKPASS": str(askpass), "GIT_USER": owner,
           "GITHUB_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"}
    try:
        print("pushing (this uploads LFS objects too, may take a minute)...")
        p = sh(["git", "push", "-u", "origin", a.branch], check=False, env=env)
        print(p.stdout or p.stderr)
        if p.returncode != 0:
            print("push failed - check the token has 'Contents: read and write' "
                  "on this repo", file=sys.stderr)
            return 5
    finally:
        askpass.unlink(missing_ok=True)
    print(f"pushed -> https://github.com/{owner}/{a.repo}")

    # ---- release assets ----------------------------------------------------- #
    if make_release:
        assets = [p for p in [
            ROOT / "data" / "processed" / "btc_1m.parquet",
            ROOT / "data" / "processed" / "btc_1m.csv.gz",
            ROOT / "checkpoints" / "best_model.pt",
        ] if p.exists()]
        st, rel = api("GET", f"/repos/{owner}/{a.repo}/releases/tags/{a.tag}", token)
        if st == 404:
            st, rel = api("POST", f"/repos/{owner}/{a.repo}/releases", token,
                          {"tag_name": a.tag, "name": f"BTC dataset {a.tag}",
                           "body": "1-minute BTCUSDT OHLCV, 1 year, plus the "
                                   "evolved champion checkpoint."})
            if st not in (200, 201):
                print(f"ERROR creating release: {st} {rel.get('message','')}",
                      file=sys.stderr)
                return 6
        up = rel["upload_url"].split("{")[0]
        existing = {x["name"] for x in rel.get("assets", [])}
        for f in assets:
            if f.name in existing:
                print(f"  asset {f.name} already present - skipping")
                continue
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            print(f"  uploading {f.name} ({f.stat().st_size/1e6:.1f} MB)...")
            st, r = api("POST", "", token, raw=f.read_bytes(), ctype=ctype,
                        url=f"{up}?name={f.name}")
            print(f"    {'ok' if st in (200,201) else 'FAILED ' + str(st)}")
        print(f"release -> https://github.com/{owner}/{a.repo}/releases/tag/{a.tag}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
