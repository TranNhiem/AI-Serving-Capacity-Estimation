#!/usr/bin/env python3
"""Pre-publication safety gate for ASCEP.

ASCEP reports are generated inside private infrastructure and published outside it. That is
exactly the path along which internal hostnames, cluster paths, job identifiers, commercial
pricing and credentials escape. This script is the gate on that path.

Two layers:

1. **Generic patterns** (below) — credential *shapes* and private address ranges that are
   unsafe for anyone. These live in the public repo because they are not site-specific.

2. **A local deny-list** at ``.ascep-denylist`` (gitignored) — your own hostnames, domains,
   project codenames and customer names, one regex per line, ``#`` for comments. This file is
   deliberately NOT committed: a scanner that ships your internal hostnames in a public repo
   has leaked the very thing it was written to protect.

The deny-list is found relative to this checkout, not to whatever tree is being scanned, so
pointing ``--path`` at a subdirectory or a staged export does not quietly drop half the rules.

Usage::

    python tools/check_no_secrets.py            # scan the repo, exit 1 on any finding
    python tools/check_no_secrets.py --path X   # scan a specific tree, or a single file
    python tools/check_no_secrets.py --cache    # skip bytes already cleared under these rules
    python tools/check_no_secrets.py --jobs 8   # one process per file

Exit code 0 means clean. It does not mean safe — no scanner is a substitute for reading the
diff before you push.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import sys

# (name, pattern, why it matters)
GENERIC_PATTERNS: list[tuple[str, str, str]] = [
    ("github-pat-classic", r"\bghp_[A-Za-z0-9]{30,}\b", "GitHub personal access token"),
    ("github-pat-fine", r"\bgithub_pat_[A-Za-z0-9_]{50,}\b", "GitHub fine-grained token"),
    ("github-oauth", r"\bgh[ousr]_[A-Za-z0-9]{30,}\b", "GitHub OAuth/refresh token"),
    ("openai-key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b", "OpenAI-style API key"),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{20,}\b", "Anthropic API key"),
    ("hf-token", r"\bhf_[A-Za-z0-9]{30,}\b", "Hugging Face token"),
    ("aws-akid", r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    ("slack-token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", "Slack token"),
    ("private-key-block", r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", "private key"),
    (
        "bearer-literal",
        r"Authorization:\s*Bearer\s+[A-Za-z0-9._~+/=-]{16,}",
        "hard-coded bearer token",
    ),
    (
        "password-assignment",
        r"(?i)\b(?:password|passwd|pwd|secret|api_key|apikey|token)\s*[=:]\s*[\"'][^\"'\s]{6,}[\"']",
        "credential assigned inline",
    ),
    (
        "rfc1918-ip",
        r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
        "private-range IP — an internal endpoint",
    ),
    (
        "home-cluster-path",
        r"/home/[a-z][a-z0-9_-]{2,}/",
        "absolute home path from a shared cluster",
    ),
    ("slurm-jobid", r"(?i)\bSLURM_JOB_ID\s*[=:]\s*\d+", "concrete Slurm job id"),
    ("currency-vnd", r"\b\d[\d,._]{4,}\s*(?:VND|₫)\b", "commercial pricing"),
    (
        "currency-generic",
        r"(?i)\b(?:unit price|quotation|contract value|list price)\b",
        "commercial terms",
    ),
]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".eggs",
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".so",
    ".dylib",
    ".bin",
    ".safetensors",
    ".parquet",
    ".ico",
    ".woff2",
}

# Files that necessarily contain the shapes this scanner looks for, and would otherwise flag
# themselves. The first two hold the patterns. The third is the test suite for
# tools/redact_bundle.py, whose fixtures must carry a cluster-shaped path and a private-range
# address or there is nothing for the redactor to redact and the tests prove nothing.
#
# This is a hole and it is worth naming: a real credential pasted into any of these three is
# not caught. The mitigation is a rule, not a mechanism -- every leak-shaped string in the
# redaction fixtures is fabricated, and one that was ever real belongs in a revocation, not
# in a test.
SELF = {pathlib.Path(__file__).name, ".ascep-denylist", "test_redact_bundle.py"}

#: Where the operator's deny-list lives. It belongs to the checkout, not to whatever tree is
#: being scanned: the terms in it are the operator's, and they are just as forbidden in a
#: subdirectory, a staged export or an unpacked bundle as they are at the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def denylist_paths(root: pathlib.Path) -> list[pathlib.Path]:
    """Every ``.ascep-denylist`` that applies to a scan of ``root``.

    Both the checkout's and the scanned tree's, because resolving it from the scan target
    alone is how the site-specific half of this gate goes missing. Scan a subdirectory, a
    ``git archive`` export or a bundle staged under /tmp and there is no deny-list beside it,
    so every operator hostname, codename and customer name stops being checked -- and the run
    still prints OK and exits 0. A gate that silently loses half its rules and reports success
    is worse than no gate, because it is believed.
    """
    here = root if root.is_dir() else root.parent
    return [
        p
        for p in dict.fromkeys([REPO_ROOT / ".ascep-denylist", here / ".ascep-denylist"])
        if p.exists()
    ]


def load_denylist(root: pathlib.Path) -> list[tuple[str, str, str]]:
    out = []
    for f in denylist_paths(root):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                re.compile(line)
            except re.error as exc:
                print(f"  {f}:{i}: invalid regex ({exc}) — skipped", file=sys.stderr)
                continue
            out.append((f"denylist:{i}", line, "site-specific term from .ascep-denylist"))
    return out


def iter_files(root: pathlib.Path):
    # A file root yields itself. rglob("*") on a file yields nothing, so without this branch
    # `--path some/report.md` scans zero files and prints "OK: no findings" -- a clean bill of
    # health for a document nobody looked at.
    candidates = [root] if root.is_file() else root.rglob("*")
    for p in candidates:
        if not p.is_file() or p.name in SELF:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield p


#: Content digests of files that scanned clean, filed under the digest of the rules that
#: cleared them. This gate is CPU-bound at roughly 5 MB/s per pattern and it re-reads every
#: published bundle on every run, so its cost grows with the archive the project exists to
#: accumulate -- a full scan is already tens of minutes and only gets worse. A published
#: bundle is immutable, so re-deriving its verdict is pure waste. Filing under the rule
#: digest is what keeps this honest: add one deny-list term and every entry is void, because
#: "clean" is only ever a claim about a specific set of rules.
CACHE_NAME = ".ascep-secretscan-cache.json"


def ruleset_digest(patterns: list[tuple[str, str, str]]) -> str:
    joined = "\n".join(f"{n}\t{p}" for n, p, _ in patterns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _scan_text(text: str, patterns) -> list[tuple[int, str, str, str]]:
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx, why in patterns:
            m = rx.search(line)
            if not m:
                continue
            hit = m.group(0)
            shown = hit[:8] + "…" if len(hit) > 12 else hit  # never echo a full secret
            out.append((lineno, name, why, shown))
    return out


_WORKER: dict = {}


def _worker_init(raw: list[tuple[str, str, str]]) -> None:
    _WORKER["patterns"] = [(n, re.compile(p), why) for n, p, why in raw]


def _worker_scan(path_s: str):
    """Returns (path, content digest, findings). Digest is None when the file was unreadable.

    Hashing costs a read the scan already pays for, and runs orders of magnitude faster than
    the patterns do, so it is free relative to what it saves on the next run.
    """
    path = pathlib.Path(path_s)
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return path_s, None, []
    return path_s, hashlib.sha256(data).hexdigest(), _scan_text(text, _WORKER["patterns"])


def scan(root: pathlib.Path, jobs: int = 1, use_cache: bool = False) -> int:
    """Scan ``root``; returns the number of findings and prints one line per finding.

    ``jobs`` defaults to 1, not to the CPU count, because a pool has to re-import this module
    in every child. That works for the CLI and fails for a caller that loaded the file by path
    -- tools/redact_bundle.py and the tests both do. The command line opts in; a library call
    stays serial and stays working.
    """
    raw = GENERIC_PATTERNS + load_denylist(root)
    digest = ruleset_digest(raw)
    cache_file = REPO_ROOT / CACHE_NAME
    known: set[str] = set()
    if use_cache and cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if blob.get("ruleset") == digest:
                known = set(blob.get("clean", []))
        except (json.JSONDecodeError, OSError):
            known = set()  # an unreadable cache means scan everything, never means clean

    paths = [str(p) for p in iter_files(root)]
    base = root if root.is_dir() else root.parent

    # Hashing has to happen in the worker or the parent re-reads every file to check the
    # cache, which on this tree is most of the wall clock the cache was meant to remove.
    # So the cache is applied inside the worker and skipped files return no findings.
    if known:
        _worker_init(raw)

        def prescreen(path_s: str) -> bool:
            try:
                return hashlib.sha256(pathlib.Path(path_s).read_bytes()).hexdigest() in known
            except OSError:
                return False

        paths = [p for p in paths if not prescreen(p)]

    results = []
    if jobs > 1 and len(paths) > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, initializer=_worker_init, initargs=(raw,)
        ) as pool:
            results = list(pool.map(_worker_scan, paths, chunksize=1))
    else:
        _worker_init(raw)
        results = [_worker_scan(p) for p in paths]

    findings = 0
    clean_now = set(known)
    # Sorted so two runs over the same tree print the same report; a pool hands results back
    # in whatever order the workers finish.
    for path_s, sha, hits in sorted(results):
        rel = pathlib.Path(path_s).relative_to(base)
        for lineno, name, why, shown in hits:
            findings += 1
            print(f"{rel}:{lineno}: [{name}] {why} — matched {shown!r}")
        if sha is not None and not hits:
            clean_now.add(sha)

    if use_cache:
        try:
            cache_file.write_text(
                json.dumps({"ruleset": digest, "clean": sorted(clean_now)}), encoding="utf-8"
            )
        except OSError:
            pass  # a cache that cannot be written costs time, never correctness
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", default=str(pathlib.Path(__file__).resolve().parent.parent))
    ap.add_argument(
        "--jobs", type=int, default=None, help="parallel workers (default: one per CPU)"
    )
    ap.add_argument(
        "--cache",
        action="store_true",
        help="skip files whose bytes already scanned clean under these exact rules. Local "
        "convenience for repeated runs over a tree of immutable published bundles; the "
        "gate in CI runs without it so every publication is checked from scratch.",
    )
    args = ap.parse_args()
    root = pathlib.Path(args.path).resolve()

    n = scan(root, jobs=args.jobs or (os.cpu_count() or 1), use_cache=args.cache)
    if n:
        print(f"\nFAIL: {n} finding(s). Nothing is published until this is clean.", file=sys.stderr)
        return 1
    used = denylist_paths(root)
    # Say which rules actually ran. "OK" from generic patterns alone and "OK" from generic
    # patterns plus the operator's 40 site terms are very different claims, and the operator
    # is the only one who can tell whether the weaker one was the intended check.
    print(
        f"OK: no findings in {root}"
        + (
            "\nChecked against: generic patterns + " + ", ".join(str(p) for p in used)
            if used
            else "\nNote: no .ascep-denylist present — generic patterns only. Add one with your "
            "own hostnames and project names before publishing."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
