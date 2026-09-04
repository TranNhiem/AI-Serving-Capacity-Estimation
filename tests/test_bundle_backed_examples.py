"""A bundle-backed example proves its numbers twice over, and this file runs both proofs.

``test_report_conformance.py`` admits two ways for a report to prove provenance: a
``build_report.py`` that must reproduce it byte for byte, or a reproduction bundle whose
manifest pins every byte the run left behind. This file exercises the second. The digest
check proves the files are intact; it says nothing about whether they describe the run the
report claims they do.

The failure these tests exist to prevent is the bundle that verifies yet does not
corroborate: a records file padded with another campaign's requests, a published row that
does not recompute from the records it cites, a path table that resolves nowhere. Each of
those reads as provenance on its own; here they must all agree at once.

Nothing is asserted to exist. While no example ships a bundle the parametrization is empty
and every test here skips -- the first bundle-backed example lands in a later commit, and
it must arrive to tests that are already watching.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import NamedTuple

import pytest

from ascep.bench.metrics import SloGates, reduce_window
from ascep.bench.persist import verify_bundle
from ascep.bench.records import RequestRecord, read_records

ROOT = pathlib.Path(__file__).parent.parent

#: Reduction outputs the report publishes per rung in ``run.results``, by the key it
#: publishes them under, in the row's own order. The row also carries per-request token
#: means, which are means over the served cohort rather than a window reduction, and are
#: not compared here. The three transport fields arrived in 0.3.0 and are optional in the
#: schema: a bundle recorded before they existed publishes rows without them, so the
#: comparison skips a key the row does not carry and reads a published null as "the
#: reduction produced None".
_RUNG_FIGURES = (
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "itl_p50_s",
    "itl_p95_s",
    "itl_population",
    "tokens_per_stream_chunk",
    "stream_chunk_gap_p50_s",
    "stream_chunk_gap_p95_s",
    "e2e_p95_s",
    "e2e_p99_s",
    "output_tok_s",
    "requests_per_s",
    "error_rate_pct",
    "slo_pass",
)

#: The non-null entries the chapter 8 path table must be able to resolve. A null is a
#: C1-declared gap and is test_report_conformance.py's business, not this file's.
_REPRODUCTION_PATHS = (
    "run_configs_path",
    "raw_records_path",
    "engine_logs_path",
    "environment_capture_path",
)

#: The rung figures each manifest-declared elision makes irreproducible, by the key the
#: manifest declares it under. With the per-token arrival stamps gone the inter-token
#: family cannot be recomputed at all, and comparing it anyway would report a wrong
#: selection rule where there is only an honestly declared gap. ``slo_pass`` is
#: deliberately NOT in this set: exempting it alongside the rest would let an elision
#: launder a rung the records grade as failing into a published pass, so it gets the
#: direction-checked comparison in ``_rung_mismatches`` instead.
_ELISION_DEPENDENTS = {
    "records.jsonl:token_ts": frozenset(
        {
            "itl_p50_s",
            "itl_p95_s",
            "itl_population",
            "tokens_per_stream_chunk",
            "stream_chunk_gap_p50_s",
            "stream_chunk_gap_p95_s",
        }
    ),
}


class _Bundle(NamedTuple):
    example_dir: pathlib.Path
    report: dict
    bundle_dir: pathlib.Path
    records: list[RequestRecord]
    run_configs: dict
    elisions: dict[str, str]


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _elisions(bundle_dir: pathlib.Path) -> dict[str, str]:
    """The manifest's declared elisions, or none when the manifest declares none.

    An absent manifest yields an empty mapping rather than an error: a bundle with no
    manifest fails verify_bundle in the test that owns that check, and treating "no
    manifest" as "no elisions" keeps the recomputation from either crashing here or
    excusing a gap nobody declared.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return _load(manifest_path).get("elisions", {})


def _find_bundle_backed_reports() -> list[pathlib.Path]:
    """Every report that ships a bundle: a records path whose directory holds a manifest.

    A builderless example with no bundle at all is NOT silently skipped here -- that hole
    is test_report_conformance.py's business, and it fails there. This list is only the
    examples the tests below can sensibly exercise.

    Having a `build_report.py` does not exempt an example, because the two checks answer
    different questions: a builder proves the published numbers come out of the script, and
    a bundle proves they come out of the records. An example that ships both and is checked
    only against its builder can publish a figure its own raw records contradict, and the
    contradiction is invisible precisely because a test appeared to cover it.
    """
    found = []
    for report_file in sorted((ROOT / "examples").glob("*/report.json")):
        example_dir = report_file.parent
        rel = _load(report_file).get("reproduction", {}).get("raw_records_path")
        if rel is None:
            continue
        if (example_dir / pathlib.Path(rel).parent / "manifest.json").is_file():
            found.append(report_file)
    return found


# Deliberately no module-level assert, unlike test_report_conformance.py: that file fails
# when no report exists because example reports are its whole subject, while this file's
# subject first appears in a later commit and the suite must collect cleanly until then.
BUNDLE_BACKED_REPORTS = _find_bundle_backed_reports()


def _same_figure(published, actual) -> bool:
    """Whether a published figure and the recomputed one are the same measurement.

    A published null is the C1 spelling of "the reduction produced None", so None on
    either side matches only None on the other. Numbers compare with ``math.isclose`` at
    rel_tol=1e-12, abs_tol=1e-15: the report stores the reduction's own unrounded doubles,
    so a wrong selection rule -- pooling the repetitions, counting the confirmation
    window, taking the upper median -- moves a figure by whole quantile steps, and the
    tolerance only has to absorb last-ulp differences between two independent summation
    orders of the same records. Strings (``itl_population``) and booleans (``slo_pass``)
    compare exactly, since a near-miss is not a concept either type has.
    """
    if published is None or actual is None:
        return published is None and actual is None
    if isinstance(published, float) or isinstance(actual, float):
        return math.isclose(published, actual, rel_tol=1e-12, abs_tol=1e-15)
    return published == actual


@pytest.fixture(params=BUNDLE_BACKED_REPORTS, ids=lambda p: p.parent.name)
def report_path(request):
    return request.param


@pytest.fixture
def bundle(report_path) -> _Bundle:
    example_dir = report_path.parent
    report = _load(report_path)
    reproduction = report["reproduction"]
    records_path = example_dir / reproduction["raw_records_path"]
    with records_path.open(encoding="utf-8") as fp:
        records = read_records(fp)
    return _Bundle(
        example_dir=example_dir,
        report=report,
        bundle_dir=records_path.parent,
        records=records,
        run_configs=_load(example_dir / reproduction["run_configs_path"]),
        elisions=_elisions(records_path.parent),
    )


def _window_records(records: list[RequestRecord], window: dict) -> list[RequestRecord]:
    """The records one declared window owns: exactly its (concurrency, repetition) pair."""
    policy = window["policy"]
    return [
        record
        for record in records
        if record.concurrency == policy["concurrency"] and record.repetition == policy["repetition"]
    ]


def _median_counted_repetition(summaries):
    """The one repetition a rung publishes, re-derived rather than imported.

    Ranks by ``output_tok_s`` and then by ``ttft_p95_s``, with a stable sort, and takes the
    LOWER median, so an even count picks the slower of the two middle windows. A repetition
    whose reduction produced no throughput figure is ranked alongside the others only when
    every repetition is in that state: ``None`` means the tokens were never counted, which
    is neither a fast window nor a zero one, and sorting it to either end is a claim -- at
    the top it becomes the median of a half-collapsed rung and publishes the best window as
    typical. A missing ``ttft_p95_s`` sorts last inside its throughput group, for the same
    reason: an unmeasured tail is not a short one.

    The second key is not decoration. Under ``ignore_eos`` with a declared output length,
    repetitions that complete the same number of requests tie exactly on ``output_tok_s``,
    the stable sort becomes a no-op, and the choice collapses to submission order -- so the
    rule would be "the second window submitted" while claiming to be a median.

    A free function so the rule can be exercised on synthetic input below, rather than only
    on a bundle that may not exist yet. It reads those two attributes and nothing else.
    """
    ranked = [s for s in summaries if s.output_tok_s is not None] or list(summaries)
    ordered = sorted(
        ranked,
        key=lambda s: (s.output_tok_s or 0.0, s.ttft_p95_s is None, s.ttft_p95_s or 0.0),
    )
    return ordered[(len(ordered) - 1) // 2]


class _Ranked(NamedTuple):
    """A stand-in for the two WindowSummary attributes the picker reads."""

    label: str
    output_tok_s: float | None
    ttft_p95_s: float | None = None


def test_the_rung_picker_takes_the_lower_median_and_never_ranks_a_dead_window():
    """The picker is the test's own copy of the harness's rule, so it is checked directly.

    Every recomputation below rests on this function choosing the same repetition the
    harness chose. If it silently took the upper median, or sorted a throughput-less window
    to the bottom as if it were the slowest, the comparison would keep reporting a clean
    match against the wrong window -- a green test that has stopped watching. The bundle
    tests skip until an example ships one; this one runs today.
    """
    three = [_Ranked("a", 300.0), _Ranked("b", 100.0), _Ranked("c", 200.0)]
    assert _median_counted_repetition(three).label == "c"

    # Even counts take the LOWER middle: with four windows the choice is index 1 of the
    # sorted four, not index 2. Taking the upper one publishes the faster window as typical.
    four = [_Ranked("a", 400.0), _Ranked("b", 100.0), _Ranked("c", 300.0), _Ranked("d", 200.0)]
    assert _median_counted_repetition(four).label == "d"

    # A window that counted no tokens is dropped from the ranking, not sorted to the floor:
    # ranked as zero it would become the median of the two survivors and publish the faster
    # of them, which is the half-collapsed rung reading as its best window.
    with_dead = [_Ranked("a", 300.0), _Ranked("dead", None), _Ranked("c", 100.0)]
    assert _median_counted_repetition(with_dead).label == "c"

    # Unless every window is in that state, in which case there is nothing else to pick and
    # the reduction's own nulls carry the (U) forward.
    all_dead = [_Ranked("a", None), _Ranked("b", None)]
    assert _median_counted_repetition(all_dead).output_tok_s is None


def test_windows_that_tie_on_throughput_are_ranked_by_latency_not_by_arrival_order():
    """A fixed output length makes throughput ties the normal case, not the edge one.

    Every request under ``ignore_eos`` emits exactly the declared number of tokens, so two
    windows that complete the same number of requests report the same ``output_tok_s`` to
    every digit. Ranking on throughput alone then leaves a stable sort with nothing to do
    and submission order picks the published window -- observed on a GB200 media ladder,
    where the report published the second repetition at all six rungs, once quoting 7.972 s
    from a rung whose three windows measured 8.769, 7.972 and 9.412 s. The fastest window,
    published as typical, is the exact failure the lower-median rule was written to prevent,
    reintroduced through the back door of a tie.
    """
    tied = [
        _Ranked("first", 833.3, 8.769),
        _Ranked("second", 833.3, 7.972),
        _Ranked("third", 833.3, 9.412),
    ]
    assert _median_counted_repetition(tied).label == "first"

    # Reordering the same three windows must not change the answer. Order-dependence is the
    # defect; a picker that still reads arrival order would return "second" here.
    reordered = [tied[1], tied[2], tied[0]]
    assert _median_counted_repetition(reordered).label == "first"

    # An unmeasured tail sorts last within its throughput group rather than to the front,
    # where it would displace a measured window and publish the faster survivor as typical.
    with_null_tail = [_Ranked("a", 500.0, 2.0), _Ranked("b", 500.0, None), _Ranked("c", 500.0, 3.0)]
    assert _median_counted_repetition(with_null_tail).label == "c"


def test_a_bundle_backed_example_verifies_its_own_manifest(bundle):
    """A bundle whose files changed after the manifest was written is not evidence.

    The manifest is what separates "here are the records" from "here are records": the
    digests pin the bytes, and a digest that no longer recomputes means the records, the
    configs, the environment capture or the engine log moved after publication. Every
    figure in the report that cites the bundle is then quoting a file nobody can identify.
    """
    problems = verify_bundle(bundle.bundle_dir)
    assert not problems, "\n".join(problems)


def _rung_mismatches(bundle: _Bundle) -> list[str]:
    """Every published rung figure that the bundled records do not reduce back to.

    The harness's rule, re-implemented here independently: group each rung's windows by
    concurrency; drop repetitions whose index is not below the declared repetition count
    (the confirmation repetition is deliberately not counted, chapter 7 section 254);
    reduce each surviving repetition SEPARATELY through ``reduce_window`` with that window's
    own ``t0`` and ``window_s``; take the median by :func:`_median_counted_repetition`; and
    compare that one reduction's fields against the published row. The harness's private
    picker is deliberately not imported: a test that calls the same helper the harness calls
    cannot catch the harness getting the rule wrong.

    The SLO gates are reconstructed from ``run.slo_gates`` rather than redeclared. Warm-up
    traffic is not pre-filtered: the bundle retains it and ``reduce_window`` excludes it
    itself, exactly as the harness's own reduction did. Declaration order is execution order
    for these windows, so the stable sort breaks ties the way the harness's does.

    One exception to exact comparison: a figure the manifest declares elided is skipped,
    because no recomputation can reach it. ``slo_pass`` is never skipped -- with the ITL
    family elided it may differ only toward strictness, never toward leniency.
    """
    run = bundle.report["run"]
    gate_block = run["slo_gates"]
    gates = SloGates(
        ttft_p95_max_s=gate_block["ttft_p95_max_s"],
        itl_p95_max_s=gate_block["itl_p95_max_s"],
        e2e_p95_max_s=gate_block["e2e_p95_max_s"],
        error_rate_max_pct=gate_block["error_rate_max_pct"],
    )
    # The declared repetition count is in the report's run block, not in the bundle config:
    # run_configs lists the windows that ran, and the section 5 confirmation window is one of
    # them. `repeats` is the only thing that says how many of them the rung is scored on.
    declared_repeats = run["repeats"]
    assert isinstance(declared_repeats, int), (
        "run.repeats is not an integer, so nothing separates the graded repetitions from "
        "the section 5 confirmation window that ran beside them; a bundle-backed example "
        "takes this figure straight from the harness and cannot honestly leave it unknown"
    )
    # The harness reduces with the workload's seed and so must the recomputation. No figure
    # compared below is drawn from it today -- it seeds the bootstrap -- but a reduction
    # carrying a different seed would start disagreeing the day an interval joins the row.
    seed = int(bundle.run_configs["workload"]["seed"])
    windows = bundle.run_configs["windows"]
    # Only a declared elision excuses a figure, and only the figures its key is known to
    # make irreproducible: an undeclared empty token_ts still produces an ITL-family
    # mismatch below, which is what keeps a quietly trimmed records file from reading as a
    # verified one.
    elided_figures = {
        figure for key in bundle.elisions for figure in _ELISION_DEPENDENTS.get(key, ())
    }
    token_ts_elided = "records.jsonl:token_ts" in bundle.elisions
    mismatches = []
    for row in run["results"]:
        concurrency = row["concurrency"]
        counted_windows = [
            window
            for window in windows
            if window["policy"]["concurrency"] == concurrency
            and window["policy"]["repetition"] < declared_repeats
        ]
        assert counted_windows, (
            f"the report publishes a rung at concurrency {concurrency} but no counted "
            "declared window owns it"
        )
        per_repetition = [
            reduce_window(
                _window_records(bundle.records, window),
                window_s=window["window_s"],
                t0=window["t0"],
                gates=gates,
                seed=seed,
            )
            for window in counted_windows
        ]
        median = _median_counted_repetition(per_repetition)
        for key in _RUNG_FIGURES:
            if key not in row:
                # The transport fields are optional in the schema; an absent key means the
                # run predates them, not that the reduction agrees. The core figures are
                # schema-required, so an absent one fails test_report_validates before it
                # could be waved through here.
                continue
            if key in elided_figures:
                continue
            published = row[key]
            actual = getattr(median, key)
            if _same_figure(published, actual):
                continue
            if key == "slo_pass" and token_ts_elided and published is False and actual is True:
                # Permission to differ runs one way only: published False beside a
                # recomputed True is the report holding itself to a stricter grade than
                # the elided records can confirm. The reverse -- published True beside a
                # recomputed False -- is the elision selling a rung the records grade as
                # failing as a gated pass, and falls through to the mismatch list.
                continue
            mismatches.append(
                f"concurrency {concurrency} {key}: the report publishes {published!r} "
                "but the lower-median counted repetition of the bundled records "
                f"reduces to {actual!r}"
            )
    return mismatches


def _assert_recomputes(bundle: _Bundle) -> None:
    mismatches = _rung_mismatches(bundle)
    assert not mismatches, (
        "figures in run.results are not the median counted repetition of the bundled "
        "records:\n  " + "\n  ".join(mismatches)
    )


def test_a_rung_row_is_the_median_repetition_and_not_a_pool_of_all_three(bundle):
    """A pooled row is a row no window exhibited; a fourth repetition shifts the median.

    The failure this test exists to catch is a rung row computed by concatenating the rung's
    repetitions and reducing the pool: pooled percentiles sit between the per-window ones,
    pooled ``output_tok_s`` is a rate no single window sustained, and such a row agrees with
    itself, its digests and its schema while quoting a rung nobody measured. The same
    address houses a second failure -- counting the section 5 confirmation window as a
    fourth repetition, which can move the chosen median at exactly the boundary rung the
    confirmation window guards. Only re-running the harness's actual rule sees either.
    """
    _assert_recomputes(bundle)


def test_the_recomputation_agrees_with_a_bundle_the_harness_has_just_written(tmp_path, monkeypatch):
    """The recomputation above skips until an example ships a bundle. This one never skips.

    A checker that has never run against a bundle is not yet a checker. Worse, the two ways
    it can be wrong are both silent: a rule that drifts from the harness's reports
    mismatches on every example at once and reads as a broken example, and a rule that is
    accidentally the harness's own -- pooling where the harness pools -- reports a clean
    match forever. Only a bundle whose provenance is known can tell those apart, and the
    offline adapter in ``test_cli_bench`` produces one on any machine, in under a second,
    with no GPU and no server.

    The ladder it drives ends with a section 5 confirmation window at the boundary rung, so
    the bundle written here carries one more window than the rung is scored on. That is the
    case the comparison most needs: counting it would move the median at exactly the rung
    whose grade the confirmation exists to defend.
    """
    pytest.importorskip("httpx", reason="ascep bench needs the [run] extra")
    # Imported inside the test, not at module scope: test_cli_bench skips itself when httpx
    # is absent, and a module-level import would turn that skip into a collection error for
    # every test in this file, including the ones that need no harness at all.
    import test_cli_bench as harness

    from ascep.cli import main

    config_path = harness._write(tmp_path, harness._config(tmp_path))
    harness._run_offline(monkeypatch)
    assert main(["bench", config_path]) == 0, "the offline ladder did not complete"

    report = _load(tmp_path / "report.json")
    bundle_dir = tmp_path / "bundle"
    with (bundle_dir / "records.jsonl").open(encoding="utf-8") as fp:
        records = read_records(fp)
    run_configs = _load(bundle_dir / "run_configs.json")
    assert len(run_configs["windows"]) > report["run"]["repeats"] * len(report["run"]["results"]), (
        "this ladder was expected to end with a confirmation window beyond the counted "
        "repetitions; without one the test is not exercising the case it exists for"
    )
    _assert_recomputes(
        _Bundle(
            example_dir=tmp_path,
            report=report,
            bundle_dir=bundle_dir,
            records=records,
            run_configs=run_configs,
            elisions=_elisions(bundle_dir),
        )
    )


def test_the_token_ts_elision_declaration_matches_exactly_what_the_records_carry(bundle):
    """A declaration is a license to skip figures, so it must be exactly true.

    The recomputation above trusts the manifest when it elides ``records.jsonl:token_ts``,
    and that trust is what this test grades. One record that kept its arrival stamps means
    the stream was cut unevenly -- a partial elision, which is a corruption wearing a
    justification, because the published ITL figures may then rest on records nobody can
    distinguish from the ones still carrying stamps. Conversely a bundle that declares no
    elision must still have stamps somewhere: one that dropped them all quietly would fail
    the re-reduction as an unexplained absence, and this test makes that failure say what
    actually happened.
    """
    stamped = sum(1 for record in bundle.records if record.token_ts)
    if "records.jsonl:token_ts" in bundle.elisions:
        assert stamped == 0, (
            "the manifest declares records.jsonl:token_ts elided but "
            f"{stamped:,} of {len(bundle.records):,} records still carry arrival stamps; "
            "a partially elided stream is a silent corruption, not an elision, and the ITL "
            "figures it half-supports are re-derivable by nobody"
        )
    else:
        assert stamped > 0, (
            "no record in this bundle carries a single token arrival stamp, yet the "
            "manifest declares no elision; an absence this total must be confessed in the "
            "manifest rather than surfacing downstream as an unexplained mismatch"
        )


def test_a_declared_elision_key_is_one_the_checks_know_how_to_honour(bundle):
    """An unknown elision key must fail loudly, on the first bundle that declares one.

    The honouring in ``_rung_mismatches`` works by explicit table, so a key nobody listed
    is either a typo or a new kind of gap the checks were never taught to excuse. Waved
    through, it exempts nothing while telling every reader the gap is handled -- or worse,
    teaches later code to treat the elisions mapping as a general license, where any
    reason-shaped string silences any figure.
    """
    unknown = sorted(set(bundle.elisions) - _ELISION_DEPENDENTS.keys())
    assert not unknown, (
        "the manifest declares elisions these tests do not know how to honour; add the "
        "dependent figures to _ELISION_DEPENDENTS or remove the declaration: " + ", ".join(unknown)
    )


def test_every_record_belongs_to_a_declared_window(bundle):
    """A bundle padded with records from another run passes every other check here.

    The manifest verifies bytes, not provenance, and the re-reduction selects records by
    the declared windows -- so records carrying a ``(concurrency, repetition)`` no window
    declares are invisible to both while bulk out of the bundle masquerades as supporting
    load. The only way to see them is to demand that every record name a window the run
    actually declared.
    """
    declared = {
        (window["policy"]["concurrency"], window["policy"]["repetition"])
        for window in bundle.run_configs["windows"]
    }
    # key=str because concurrency/repetition may legitimately be null in the record
    # contract, and a type-mixed sort would raise before reporting the real problem.
    stray = sorted({(r.concurrency, r.repetition) for r in bundle.records} - declared, key=str)
    assert not stray, (
        "records whose (concurrency, repetition) pair no declared window owns; "
        f"a bundle may not carry traffic from a run it does not declare: {stray}"
    )


def test_every_reproduction_path_resolves_to_a_file_inside_the_example(bundle):
    """A path that resolves nowhere is provenance by reference, not by value.

    Chapter 8 requires the path table to resolve from the report directory, and a null
    entry is a C1-declared gap that test_report_conformance.py already grades -- not this
    test's business. What must not pass here is a non-null path that escapes the example
    or names a file that does not exist: a reader who downloads the example follows the
    table and finds the evidence missing, which is worse than an honest gap labeled (U)
    because it looks like provenance until someone checks.
    """
    reproduction = bundle.report["reproduction"]
    problems = []
    for key in _REPRODUCTION_PATHS:
        rel = reproduction.get(key)
        if rel is None:
            continue
        target = (bundle.example_dir / rel).resolve()
        try:
            target.relative_to(bundle.example_dir.resolve())
        except ValueError:
            problems.append(f"reproduction.{key} ({rel}) escapes the example directory")
            continue
        if not target.is_file():
            problems.append(f"reproduction.{key} ({rel}) does not resolve to an existing file")
    assert not problems, "\n  ".join(problems)


def test_the_environment_capture_names_the_versions_that_produced_the_numbers(bundle):
    """Nothing in a bundle-backed example is typed by a human -- unless the versions are.

    The whole point of a bundle-backed example over a hand-written one is that every byte
    came out of the harness. The serving stack moves throughput by tens of percent between
    releases, and in a hand-written example the framework version is a bare declaration
    with no corroboration. The bundle's environment.json is the corroboration: probed from
    process metadata at run time, not copied from anyone's notes. A capture whose packages
    mapping is empty or absent throws that away, leaving the typed
    ``serving.framework_version`` as the only account of what served the requests.
    """
    # Read the bundle's own artifact rather than resolving the report's path table entry
    # for it: that entry is the previous test's ground, and the environment capture is
    # pinned by the manifest under the bundle's own fixed name.
    environment = json.loads((bundle.bundle_dir / "environment.json").read_text(encoding="utf-8"))
    packages = environment.get("packages")
    assert isinstance(packages, dict) and packages, (
        "environment.json has no non-empty packages mapping, so nothing independent "
        "corroborates the versions of the software that produced these numbers"
    )
