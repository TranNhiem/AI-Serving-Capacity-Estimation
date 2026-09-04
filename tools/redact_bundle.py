#!/usr/bin/env python3
"""Honest redaction for ASCEP reproduction bundles.

A bundle pins every artifact by sha256 in manifest.json, which is what makes a published
report reproducible: a bundle that verifies is the bundle that ran. The engine log is the
artifact that leaks. A server started from a host checkpoint path writes that path into
its startup banner, and check_no_secrets.py then refuses the push -- correctly. The
operator is left choosing between publishing the internal paths and deleting the log, and
both are wrong. Editing the log in place is worse than either, because the manifest then
disagrees with the bytes and the whole reproduction claim collapses.

This tool is the third move: replace named literal strings in named bundle artifacts,
record every substitution in the manifest, and re-seal the digests over the published
bytes. The manifest's promise narrows from "these are the bytes the run wrote" to "these
are the bytes published, and here is exactly how they differ from what the run wrote" --
and the second promise is the one a reader outside the operator's network can check.

Usage::

    python tools/redact_bundle.py BUNDLE_DIR --replace 'OLD=>NEW' [--replace ...] [--dry-run]

OLD is a literal string, not a regex: an operator redacting a path under time pressure
must not have to think about what "." means, and a regex that matches more than intended
silently corrupts evidence. Exit 0 on success, 1 on a refusal or a remaining finding, 2
on a usage error. A leaked credential is never redacted by this tool: it is revoked and
the artifact is regenerated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# `python tools/redact_bundle.py` puts tools/ on sys.path, not the repo root, so the
# ascep import below fails on a checkout that is not installed unless the root is added.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import check_no_secrets  # noqa: E402

from ascep.bench import persist  # noqa: E402

#: The rules in check_no_secrets.GENERIC_PATTERNS that name something an operator may
#: legitimately redact: internal topology, job identifiers and commercial terms, the
#: leaks this tool exists for. A match on any other generic rule is treated as
#: credential-shaped and refused, so a rule added to the scanner later fails closed
#: instead of quietly renaming a new kind of secret.
_REDACTABLE_RULES = {
    "rfc1918-ip",
    "home-cluster-path",
    "slurm-jobid",
    "currency-vnd",
    "currency-generic",
}


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[tuple[str, str]]]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle_dir", type=pathlib.Path, help="directory holding manifest.json")
    ap.add_argument(
        "--replace",
        action="append",
        required=True,
        metavar="OLD=>NEW",
        help="replace the literal string OLD with NEW; repeatable",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would change, per file, and write nothing",
    )
    args = ap.parse_args(argv)
    replacements = []
    for spec in args.replace:
        # Split on the first "=>" so a replacement may itself contain one, and refuse an
        # empty OLD: the empty string matches everywhere, which is never the intent.
        old, sep, new = spec.partition("=>")
        if not sep or not old:
            ap.error(f"--replace {spec!r} must be OLD=>NEW with a non-empty OLD")
        replacements.append((old, new))
    return args, replacements


def _credential_match(old: str) -> tuple[str, str] | None:
    """The scanner rule ``old`` trips, if it is credential-shaped rather than a site term."""
    for name, pattern, why in check_no_secrets.GENERIC_PATTERNS:
        if name in _REDACTABLE_RULES:
            continue
        if re.search(pattern, old):
            return name, why
    return None


def main(argv: list[str] | None = None) -> int:
    args, replacements = _parse_args(argv)
    bundle_dir = args.bundle_dir
    manifest_file = bundle_dir / persist._MANIFEST_NAME

    # Redacting a bundle that does not verify would record a transformation from bytes
    # nobody can identify: the substitution record would describe a change to evidence
    # that was already broken, and look like provenance while being none.
    problems = persist.verify_bundle(bundle_dir)
    if problems:
        for problem in problems:
            print(f"refusing to redact an unverifiable bundle: {problem}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = manifest["sha256"]

    for old, _new in replacements:
        hit = _credential_match(old)
        if hit is None:
            continue
        name, why = hit
        # Truncate before echoing, as the scanner does: the string is credential-shaped,
        # and the stderr of a command run on a shared host is not a safe place for it.
        shown = old[:8] + "..." if len(old) > 12 else old
        print(
            f"refusing to redact {shown!r}: it matches {name} ({why}). A leaked "
            "credential is revoked and the artifact regenerated, never renamed -- "
            "renaming leaves the live secret in the operator's history and produces a "
            "bundle that only looks clean.",
            file=sys.stderr,
        )
        return 1

    texts: dict[str, str] = {}
    for name in entries:
        try:
            texts[name] = (bundle_dir / name).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            # A binary artifact cannot be edited by string substitution; pretending
            # otherwise would truncate it.
            print(
                f"skipping {name}: not valid UTF-8, cannot redact by substitution",
                file=sys.stderr,
            )

    counts = {name: [text.count(old) for old, _ in replacements] for name, text in texts.items()}
    for i, (old, _new) in enumerate(replacements):
        if sum(c[i] for c in counts.values()) == 0:
            # A replacement that matches nothing is almost always a typo in the very
            # string the operator believed they were removing; succeeding silently would
            # tell them the leak was handled.
            print(
                f"refusing: --replace {old!r} matches nothing anywhere in the bundle",
                file=sys.stderr,
            )
            return 1

    changed = {name: c for name, c in counts.items() if any(c)}

    if args.dry_run:
        for name in sorted(changed):
            for i, (old, new) in enumerate(replacements):
                if not changed[name][i]:
                    continue
                n = changed[name][i]
                print(f"{name}: would apply {old!r} => {new!r} ({n} occurrence(s))")
        return 0

    originals = {name: (bundle_dir / name).read_bytes() for name in changed}
    for name in changed:
        text = texts[name]
        for old, new in replacements:
            text = text.replace(old, new)
        (bundle_dir / name).write_bytes(text.encode("utf-8"))

    # Re-scan through the scanner's own entry point. scan() walks a tree, so the roots
    # are the bundle plus the parent of any redacted artifact that lives beside it -- a
    # pass-through engine log is part of the tree being published even though the
    # manifest names it through "..".
    roots = {bundle_dir.resolve()}
    for name in changed:
        target = (bundle_dir / name).resolve()
        if not target.is_relative_to(bundle_dir.resolve()):
            roots.add(target.parent)
    findings = sum(check_no_secrets.scan(root) for root in sorted(roots))
    if findings:
        # A partial redaction must never survive a failure: restore first, then refuse.
        for name, data in originals.items():
            (bundle_dir / name).write_bytes(data)
        print(
            f"refusing to finish: {findings} finding(s) remain after redaction; originals restored",
            file=sys.stderr,
        )
        return 1

    redactions = manifest.setdefault("redactions", {})
    prefixes = manifest.get("hashed_prefix_bytes")
    summary = []
    for name in sorted(changed):
        digest = persist._sha256(bundle_dir / name)
        # A file redacted before keeps the sha256_original from its first redaction: the
        # record must point back to what the run wrote, not to the previous pass.
        entry = redactions.setdefault(name, {"sha256_original": entries[name], "substitutions": []})
        # The record holds the replacement and the count, never the redacted string
        # itself: writing OLD into the manifest would publish the internal hostname the
        # operator just removed, in the one file every reader opens.
        entry["substitutions"].extend(
            {"replacement": new, "occurrences": changed[name][i]}
            for i, (_old, new) in enumerate(replacements)
            if changed[name][i]
        )
        entries[name] = digest
        if prefixes and name in prefixes:
            # A prefix hash means "the server kept writing after the run, and the first
            # N bytes are the evidence"; a substitution that changes the length of those
            # first N bytes makes the recorded length meaningless.
            del prefixes[name]
            print(
                f"dropping hashed_prefix_bytes for {name}: the redaction changed the "
                "bytes the recorded prefix length describes",
                file=sys.stderr,
            )
        summary.append((name, sum(changed[name]), digest))
    if prefixes is not None and not prefixes:
        del manifest["hashed_prefix_bytes"]
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, count, digest in summary:
        print(f"{name}: {count} substitution(s), sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
