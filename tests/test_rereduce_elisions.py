"""Tests for declared evidence elisions during bundle re-reduction.

A manifest can disclose bytes it no longer pins, but disclosure must never let a
rebuild publish figures from less evidence while presenting them as measured. These
tests pin the boundary at which elisions are loaded, acknowledged, and refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_bench_rereduce as fixture_ref

from ascep.bench import persist
from ascep.bench.rereduce import ReduceError, check_report, rebuild_report

TOKEN_TS_ELISION = {
    "records.jsonl:token_ts": (
        "emptied on 23,284 records (23,284 of which carried stamps, 26,783,390 "
        "timestamps in total, 444.1 MB reduced to the size on disk here). The "
        "per-token arrival stamps back the inter-token-latency percentiles only; "
        "every other field is intact, so output throughput, requests per second, "
        "TTFT, end-to-end latency, the in-window boundary decision and the error "
        "rate all re-derive exactly from this file. The ITL percentiles in "
        "report.json were computed from the full stream before it was elided and "
        "cannot be recomputed from this bundle. What this prevents: a reader "
        "inferring that ITL was never measured because the field that backs it is "
        "empty here."
    )
}


def _bundle_copy(tmp_path) -> Path:
    """A fresh, valid, verifying bundle directory built by the fixture_ref helpers."""
    bundle_dir = Path(tmp_path) / "bundle"
    specs = fixture_ref._window_specs()
    fixture_ref._write_bundle(bundle_dir, specs)
    assert persist.verify_bundle(bundle_dir) == []
    return bundle_dir


def _set_elisions(bundle_dir, elisions) -> None:
    """Write `elisions` into the bundle's manifest.json, or remove the key when None."""
    path = Path(bundle_dir) / "manifest.json"
    manifest = fixture_ref._read_json(path)
    if elisions is None:
        manifest.pop("elisions", None)
    else:
        manifest["elisions"] = elisions
    fixture_ref._write_json(path, manifest)


def _repin_manifest(bundle_dir: Path) -> None:
    """Recompute pinned digests without discarding manifest metadata beside them."""
    manifest = fixture_ref._read_json(bundle_dir / "manifest.json")
    manifest["sha256"] = {
        path.relative_to(bundle_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    fixture_ref._write_json(bundle_dir / "manifest.json", manifest)


def _previous_report(bundle_dir) -> dict:
    """The published report that bundle backs, as `check_report` expects to receive it."""
    specs = fixture_ref._window_specs()
    return fixture_ref._original_report(Path(bundle_dir), fixture_ref._window_runs(specs))


def _empty_token_ts(bundle_dir) -> None:
    """Empty every record's token_ts and re-pin the manifest digests."""

    def empty_stamps(records):
        for record in records:
            record["token_ts"] = []

    fixture_ref._edit_records(Path(bundle_dir), empty_stamps)
    _repin_manifest(Path(bundle_dir))
    assert persist.verify_bundle(bundle_dir) == []


def test_load_elisions_returns_the_declared_mapping_when_the_manifest_declares_one(tmp_path):
    """The reduction must see the declaration exactly as published, because a rewritten
    key or reason could move the elision outside the refusal that protects the report."""
    bundle_dir = _bundle_copy(tmp_path)
    _set_elisions(bundle_dir, TOKEN_TS_ELISION)

    assert persist.load_elisions(bundle_dir) == TOKEN_TS_ELISION
    assert persist.verify_bundle(bundle_dir) == []


def test_load_elisions_keeps_valid_entries_beside_bad_ones_and_ignores_bad_metadata(
    tmp_path,
):
    """Every invalid whole-document case yields no declaration, while one bad entry
    must not discard the valid declaration beside it. Discarding that entry would
    turn a declared loss of token stamps into an undeclared one, allowing rebuild to
    publish the substituted ITL figures as though they had been measured."""
    bundle_dir = _bundle_copy(tmp_path)
    assert persist.load_elisions(bundle_dir) == {}

    (bundle_dir / "manifest.json").unlink()
    assert persist.load_elisions(bundle_dir) == {}

    bundle_dir = _bundle_copy(tmp_path / "other-shape")
    _set_elisions(bundle_dir, ["records.jsonl:token_ts"])
    assert persist.load_elisions(bundle_dir) == {}

    bundle_dir = _bundle_copy(tmp_path / "mixed-entries")
    declared = copy.deepcopy(TOKEN_TS_ELISION)
    declared["records.jsonl:output_tokens"] = 17
    _set_elisions(bundle_dir, declared)
    assert persist.load_elisions(bundle_dir) == TOKEN_TS_ELISION


def test_a_rebuild_refuses_elided_token_stamps_and_quotes_the_declared_reason(tmp_path):
    """With token_ts emptied, the reduction can substitute per-request means for the
    pooled token gaps and pass rungs the measured stream failed. Rebuild must refuse
    the declared elision rather than publish that larger capacity from less evidence."""
    bundle_dir = _bundle_copy(tmp_path)
    _empty_token_ts(bundle_dir)
    _set_elisions(bundle_dir, TOKEN_TS_ELISION)

    with pytest.raises(ReduceError) as exc_info:
        rebuild_report(bundle_dir, previous_report=_previous_report(bundle_dir))

    message = str(exc_info.value)
    assert "records.jsonl:token_ts" in message
    assert TOKEN_TS_ELISION["records.jsonl:token_ts"] in message


def test_a_rebuild_uses_empty_token_stamps_once_the_elision_is_undeclared(tmp_path):
    """The records are exactly as empty on both sides of this boundary; only the
    declaration differs. Success after removal proves the refusal follows the manifest's
    disclosure, rather than guessing at missing evidence from the empty field itself."""
    bundle_dir = _bundle_copy(tmp_path)
    _empty_token_ts(bundle_dir)
    _set_elisions(bundle_dir, None)

    rebuilt = rebuild_report(bundle_dir, previous_report=_previous_report(bundle_dir))

    assert rebuilt["run"]["results"]
    assert persist.verify_bundle(bundle_dir) == []


def test_check_on_a_bundle_with_no_elisions_is_the_full_comparison(tmp_path):
    """A bundle that declares nothing missing must get the check it always got: if an
    absent elisions key somehow flipped the partial path on, an empty `skipped` beside
    `elided is True` would tell the operator everything was checked while the
    comparison had been quietly narrowed."""
    bundle_dir = _bundle_copy(tmp_path)
    result = check_report(bundle_dir, previous_report=_previous_report(bundle_dir))
    assert result.elided is False
    assert result.skipped == {}
    assert result.differing == ()


def test_check_on_an_elided_bundle_skips_the_itl_figures_without_diffing_them(tmp_path):
    """The partial check's promise is that the elision's blast radius is named, not
    hidden: a skipped figure that also appeared in `differing` would tell the operator
    the report is both uncheckable and wrong on the same figure, while a skipped list
    missing an ITL figure would let a substituted per-request-mean value pass as
    checked -- the exact silent substitution this module exists to refuse."""
    bundle_dir = _bundle_copy(tmp_path)
    _empty_token_ts(bundle_dir)
    _set_elisions(bundle_dir, TOKEN_TS_ELISION)
    declared_reason = TOKEN_TS_ELISION["records.jsonl:token_ts"]
    result = check_report(bundle_dir, previous_report=_previous_report(bundle_dir))
    assert result.elided is True
    itl_backed = {
        "itl_p50_s",
        "itl_p95_s",
        "itl_p99_s",
        "itl_population",
        "stream_chunk_gap_p50_s",
        "stream_chunk_gap_p95_s",
        "tokens_per_stream_chunk",
        "slo_pass",
        "outcome",
        "reasons",
    }
    for figure in itl_backed:
        # The operator must read the manifest's own words, not a paraphrase -- a
        # generic "field elided" would hide that 26.8M stamps were deliberately dropped.
        assert result.skipped[figure] == declared_reason
    # The report-level sections the verdicts feed are skipped with them; publishing a
    # capacity tier over unverifiable rungs would restate the elided evidence as measured.
    assert result.skipped["capacity_tiers"] == declared_reason
    assert result.skipped["unmeasured_assumptions"] == declared_reason
    assert itl_backed.isdisjoint(result.differing)


def test_check_on_an_elided_bundle_still_catches_an_untainted_disagreement(tmp_path):
    """A partial check that skips too much is a false pass in a reproducibility gate:
    if stripping the ITL figures also swallowed a throughput figure the records still
    pin, --check would report success on a report claiming output the evidence never
    produced."""
    bundle_dir = _bundle_copy(tmp_path)
    _empty_token_ts(bundle_dir)
    _set_elisions(bundle_dir, TOKEN_TS_ELISION)
    previous = json.loads(json.dumps(_previous_report(bundle_dir)))
    # results is a list of rung rows, so the tamper lands on rows[0], not on a
    # concurrency key that does not exist -- and it must be an untainted figure, or the
    # test would only re-prove that skips work.
    previous["run"]["results"][0]["output_tok_s"] += 1.0
    result = check_report(bundle_dir, previous_report=previous)
    assert result.elided is True
    assert "run" in result.differing
    # Scope, not mere non-emptiness: the same check that catches the tamper must still
    # skip what the elision backs, or it has drifted into refusing the whole bundle.
    assert result.skipped["itl_p95_s"] == TOKEN_TS_ELISION["records.jsonl:token_ts"]
    assert "slo_pass" in result.skipped


def test_check_on_an_elided_bundle_never_absorbs_a_rung_set_disagreement(tmp_path):
    """An elision can blank values inside a row; it cannot mint or erase a concurrency
    the harness measured at. If a missing rung were absorbed into the skip, a published
    report could drop the rung that failed the gate and still check clean against its
    own bundle."""
    bundle_dir = _bundle_copy(tmp_path)
    _empty_token_ts(bundle_dir)
    _set_elisions(bundle_dir, TOKEN_TS_ELISION)
    previous = json.loads(json.dumps(_previous_report(bundle_dir)))
    previous["run"]["results"].pop()
    result = check_report(bundle_dir, previous_report=previous)
    assert result.elided is True
    assert "run.results" in result.differing
    # "run" would differ only because the results inside it did; naming both would say
    # one thing twice and bury the rung-set claim under a second path.
    assert "run" not in result.differing
    assert "run.results" not in result.skipped
    assert result.skipped["itl_p95_s"] == TOKEN_TS_ELISION["records.jsonl:token_ts"]


def test_an_unscopable_elision_refuses_both_entry_points(tmp_path):
    """Membership in the impact table is knowledge of the blast radius, not a
    condition for having one: a field the table does not describe makes the partial
    check certify figures the elision may have moved, and the rebuild publish a ladder
    graded against substituted evidence -- so both paths must refuse rather than guess.
    An elision naming a file the reduction never reads, by contrast, cannot move a
    number the report rests on, and refusing it would punish an honest declaration."""
    unscopable = _bundle_copy(tmp_path / "unscopable")
    _set_elisions(
        unscopable,
        {"records.jsonl:first_token_ts": "emptied to shrink the bundle"},
    )
    with pytest.raises(ReduceError, match="cannot scope"):
        check_report(unscopable, previous_report=_previous_report(unscopable))
    with pytest.raises(ReduceError, match="declares elisions"):
        rebuild_report(unscopable, previous_report=_previous_report(unscopable))

    other_file = _bundle_copy(tmp_path / "other_file")
    _set_elisions(other_file, {"engine.log:hostname": "stripped for privacy"})
    result = check_report(other_file, previous_report=_previous_report(other_file))
    assert result.elided is False
    assert result.skipped == {}
    assert result.differing == ()
