"""Redacting a bundle without breaking the one promise it makes.

check_no_secrets.py refuses to publish a bundle whose engine log carries internal paths,
and it is right to. The operator's remaining moves are publishing the leak, deleting the
log, or editing it in place and leaving the manifest disagreeing with the bytes -- each
one trades the reproduction claim away. tools/redact_bundle.py is the honest move:
substitute named literals, record the substitutions in the manifest, and re-seal the
digests over the published bytes.

The failure these tests exist to prevent is the redaction that rewrites history. A bundle
that no longer verifies, or a manifest that names what was removed, is worse than the
leak the redaction was meant to fix.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

from ascep.bench import persist

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "redact_bundle.py"

# Fabricated, and they must stay fabricated: check_no_secrets.py skips this file by name,
# because a fixture for the redactor has to carry the shapes the scanner refuses or there is
# nothing to redact. Nothing real goes in here -- a string that was ever live belongs in a
# revocation, not in a test the scanner has been told not to read.
INTERNAL_PATH = "/home/clustergroup/jdoe/pretraining_weights/checkpoint-42/gemma-4-31b-it"
INTERNAL_IP = "10.10.10.10"
NODE_NAME = "node-17.internal"

REDACTED_PATH = "/models/gemma-4-31b-it"
REDACTED_IP = "REDACTED-NODE-IP"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine_log(extra: str = "") -> str:
    return (
        f"model   {INTERNAL_PATH}\n"
        f"distributed_init_method=tcp://{INTERNAL_IP}:40009\n"
        f"node {NODE_NAME} ready\n"
    ) + extra


def _bundle(
    tmp_path: pathlib.Path, *, outside_log: bool = False, extra_log: str = ""
) -> pathlib.Path:
    """A small real bundle: files on disk and a manifest over their digests.

    Written directly rather than through persist.write_bundle, because what is under
    test is the redactor's handling of the manifest shape, not the writer's.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    records = bundle / "records.jsonl"
    records.write_text('{"request_id": "r0", "outcome": "ok"}\n', encoding="utf-8")
    entries = {"records.jsonl": _sha256(records)}
    if outside_log:
        log, key = tmp_path / "engine.log", "../engine.log"
    else:
        log, key = bundle / "engine.log", "engine.log"
    log.write_text(_engine_log(extra_log), encoding="utf-8")
    entries[key] = _sha256(log)
    manifest = {"sha256": entries}
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (bundle / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return bundle


def _run(bundle: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), str(bundle), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _redactions(bundle: pathlib.Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return manifest["redactions"]


# --- the bundle must still verify afterwards ------------------------------------------


def test_a_redaction_rewrites_the_artifact_and_leaves_a_bundle_that_still_verifies(tmp_path):
    """The whole contract: if the tool leaves a bundle that fails verify_bundle, it has
    traded one broken promise for another and the operator is worse off than before."""
    bundle = _bundle(tmp_path)
    result = _run(
        bundle,
        "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}",
        "--replace", f"{INTERNAL_IP}=>{REDACTED_IP}",
    )
    assert result.returncode == 0, result.stderr
    log = (bundle / "engine.log").read_text(encoding="utf-8")
    assert INTERNAL_PATH not in log
    assert REDACTED_PATH in log
    assert "engine.log" in result.stdout
    assert persist.verify_bundle(bundle) == []


def test_the_record_keeps_the_original_digest_and_count_but_never_the_redacted_string(tmp_path):
    """Writing OLD into the manifest would publish the internal path the operator just
    removed, in the one file every reader opens -- the mistake the record's shape exists
    to prevent."""
    bundle = _bundle(tmp_path)
    run_digest = _sha256(bundle / "engine.log")
    result = _run(
        bundle,
        "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}",
        "--replace", f"{INTERNAL_IP}=>{REDACTED_IP}",
    )
    assert result.returncode == 0, result.stderr
    raw = (bundle / "manifest.json").read_text(encoding="utf-8")
    assert INTERNAL_PATH not in raw
    assert INTERNAL_IP not in raw
    record = json.loads(raw)["redactions"]["engine.log"]
    assert record["sha256_original"] == run_digest
    assert record["substitutions"] == [
        {"replacement": REDACTED_PATH, "occurrences": 1},
        {"replacement": REDACTED_IP, "occurrences": 1},
    ]


# --- refusals -------------------------------------------------------------------------


def test_a_bundle_that_does_not_verify_going_in_is_refused_and_left_untouched(tmp_path):
    """Redacting evidence that is already broken records a transformation from bytes
    nobody can identify; the substitution record would look like provenance and be none."""
    bundle = _bundle(tmp_path)
    with (bundle / "records.jsonl").open("a", encoding="utf-8") as fp:
        fp.write('{"request_id": "forged"}\n')
    log_before = (bundle / "engine.log").read_bytes()
    manifest_before = (bundle / "manifest.json").read_bytes()
    result = _run(bundle, "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}")
    assert result.returncode == 1
    assert (bundle / "engine.log").read_bytes() == log_before
    assert (bundle / "manifest.json").read_bytes() == manifest_before


def test_a_replacement_whose_old_matches_a_credential_pattern_is_refused(tmp_path):
    """A leaked credential is revoked and the artifact regenerated, never renamed:
    renaming leaves the live secret in the operator's history and publishes a bundle
    that only looks clean."""
    token = "ghp_" + "a" * 30
    bundle = _bundle(tmp_path, extra_log=f"uploaded with {token}\n")
    digest_before = _sha256(bundle / "engine.log")
    result = _run(bundle, "--replace", f"{token}=>REDACTED-TOKEN")
    assert result.returncode == 1
    assert "revoked" in result.stderr
    assert _sha256(bundle / "engine.log") == digest_before


def test_a_replacement_that_matches_nothing_is_refused(tmp_path):
    """A --replace that matches nothing is almost always a typo in the very string the
    operator believed they were removing; silently succeeding would tell them the leak
    was handled."""
    bundle = _bundle(tmp_path)
    result = _run(bundle, "--replace", "/home/clustergroup/jdoe/weights-v2=>/models/x")
    assert result.returncode == 1
    assert persist.verify_bundle(bundle) == []


def test_dry_run_reports_what_would_change_and_writes_nothing(tmp_path):
    """A rehearsal that edits evidence is not a rehearsal: the operator reads the plan
    and then runs for real, and any byte moved by the first run changes what the second
    one reports."""
    bundle = _bundle(tmp_path)
    log_digest = _sha256(bundle / "engine.log")
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    result = _run(
        bundle,
        "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}",
        "--replace", f"{INTERNAL_IP}=>{REDACTED_IP}",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "engine.log" in result.stdout
    assert _sha256(bundle / "engine.log") == log_digest
    assert (bundle / "manifest.json").read_bytes() == manifest_bytes


def test_a_redaction_that_leaves_a_finding_restores_the_original_and_refuses(tmp_path):
    """The dangerous outcome is the half-done redaction that is kept. If the tool wrote the
    edited log, re-sealed the manifest and only then refused, the operator would hold a
    bundle whose evidence had been altered and whose leak was still in it -- and the digest
    would say the altered bytes were the ones to publish. Restore first, then refuse."""
    bundle = _bundle(tmp_path)
    log_before = (bundle / "engine.log").read_bytes()
    manifest_before = (bundle / "manifest.json").read_bytes()
    # Redact the path and leave the address: the scanner still has an rfc1918 hit to find.
    result = _run(bundle, "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}")
    assert result.returncode == 1
    assert "originals restored" in result.stderr
    assert (bundle / "engine.log").read_bytes() == log_before
    assert (bundle / "manifest.json").read_bytes() == manifest_before
    assert persist.verify_bundle(bundle) == []


# --- the record points back to what the run wrote --------------------------------------


def test_a_second_pass_keeps_the_first_original_digest_and_appends(tmp_path):
    """sha256_original must name what the run wrote. Overwriting it on each pass would
    point the record at the previous redaction instead, and the chain back to the run
    would be gone after the second pass."""
    bundle = _bundle(tmp_path)
    run_digest = _sha256(bundle / "engine.log")
    first = _run(
        bundle,
        "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}",
        "--replace", f"{INTERNAL_IP}=>{REDACTED_IP}",
    )
    assert first.returncode == 0, first.stderr
    second = _run(bundle, "--replace", f"{NODE_NAME}=>node")
    assert second.returncode == 0, second.stderr
    record = _redactions(bundle)["engine.log"]
    assert record["sha256_original"] == run_digest
    assert record["substitutions"] == [
        {"replacement": REDACTED_PATH, "occurrences": 1},
        {"replacement": REDACTED_IP, "occurrences": 1},
        {"replacement": "node", "occurrences": 1},
    ]
    assert persist.verify_bundle(bundle) == []


def test_an_artifact_named_through_dot_dot_in_the_manifest_is_reached_and_redacted(tmp_path):
    """A real engine log lives beside the bundle, not inside it. A tool that only reached
    inward would re-seal a bundle whose leakiest artifact still carried the leak."""
    bundle = _bundle(tmp_path, outside_log=True)
    result = _run(
        bundle,
        "--replace", f"{INTERNAL_PATH}=>{REDACTED_PATH}",
        "--replace", f"{INTERNAL_IP}=>{REDACTED_IP}",
    )
    assert result.returncode == 0, result.stderr
    log = (tmp_path / "engine.log").read_text(encoding="utf-8")
    assert INTERNAL_PATH not in log
    assert persist.verify_bundle(bundle) == []
    assert "../engine.log" in _redactions(bundle)
