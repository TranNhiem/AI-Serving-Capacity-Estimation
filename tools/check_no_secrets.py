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

Usage::

    python tools/check_no_secrets.py           # scan the repo, exit 1 on any finding
    python tools/check_no_secrets.py --path X  # scan a specific tree

Exit code 0 means clean. It does not mean safe — no scanner is a substitute for reading the
diff before you push.
"""

from __future__ import annotations

import argparse
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


def load_denylist(root: pathlib.Path) -> list[tuple[str, str, str]]:
    f = root / ".ascep-denylist"
    if not f.exists():
        return []
    out = []
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            re.compile(line)
        except re.error as exc:
            print(f"  .ascep-denylist:{i}: invalid regex ({exc}) — skipped", file=sys.stderr)
            continue
        out.append((f"denylist:{i}", line, "site-specific term from .ascep-denylist"))
    return out


def iter_files(root: pathlib.Path):
    for p in root.rglob("*"):
        if not p.is_file() or p.name in SELF:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield p


def scan(root: pathlib.Path) -> int:
    patterns = [(n, re.compile(p), why) for n, p, why in GENERIC_PATTERNS + load_denylist(root)]
    findings = 0
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, rx, why in patterns:
                m = rx.search(line)
                if not m:
                    continue
                findings += 1
                hit = m.group(0)
                shown = hit[:8] + "…" if len(hit) > 12 else hit  # never echo a full secret
                rel = path.relative_to(root)
                print(f"{rel}:{lineno}: [{name}] {why} — matched {shown!r}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", default=str(pathlib.Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = pathlib.Path(args.path).resolve()

    n = scan(root)
    if n:
        print(f"\nFAIL: {n} finding(s). Nothing is published until this is clean.", file=sys.stderr)
        return 1
    has_deny = (root / ".ascep-denylist").exists()
    print(
        f"OK: no findings in {root}"
        + (
            ""
            if has_deny
            else "\nNote: no .ascep-denylist present — generic patterns only. Add one with your "
            "own hostnames and project names before publishing."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
