"""Tests for re-reducing a measurement bundle under today's reduction rules.

A bundle pins every byte the reduction reads, so a fixed reduction can be applied
without re-burning GPU hours: on one 14,000-request media ladder the measurement cost
more than 6,000 GPU-seconds, and the re-derivation costs seconds. The module's whole
value is that it refuses rather than guesses, so most of these tests are refusal tests,
and each docstring names the dishonest report the refusal prevents.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from ascep import conformance
from ascep.bench import persist, report, run
from ascep.bench.driver import Boundary, WindowPolicy, WindowRun
from ascep.bench.ladder import RepetitionResult, grade_ladder
from ascep.bench.metrics import reduce_window
from ascep.bench.records import read_records
from ascep.bench.rereduce import ReduceError, load_window_runs, rebuild_report
from ascep.cli import main

_T0 = 1_700_000_000.0
_SEED = 20240517

#: Figures today's reduction emits that the report pinned in the fixture predates. The
#: happy-path comparison ignores exactly these and nothing else.
_NEWLY_EMITTED_ROW_KEYS = ("dispersion", "dispersion_u_reason")

#: Keys mirror the published bench-config.json with the ladder scaled down to two
#: rungs; an invented key would make run.load_config reject the bundle in setup for
#: reasons that have nothing to do with the behaviour under test.
_BENCH_CONFIG = {
    "endpoint": {
        "base_url": "http://127.0.0.1:8000",
        "model": "qwen3-32b",
        "timeout_s": 300,
    },
    "declarations": {
        "hardware": "hardware.json",
        "model": "model.json",
        "serving": "serving.json",
        "workload": "workload.json",
    },
    "workload": {
        "corpus": "synthetic",
        "input_tokens": 128,
        "output_tokens": 5,
        "ignore_eos": True,
        "cache_policy": "unique-prefix",
        "seed": _SEED,
        "think_time_s": 1.5,
        "run_label": "qwen3-32b-gb200x4-tp4-synthetic",
    },
    "window": {"window_s": 30.0, "drain_deadline_s": 10.0, "warmup_requests": 1},
    # Three rungs, not two: run.py refuses a shorter ladder outright, because two points
    # cannot show where throughput stops scaling. A fixture the harness would reject is not
    # a bundle, so a test built on one proves nothing about re-reducing a real one.
    "ladder": {"concurrency": [2, 4, 6], "repetitions": 3, "throughput_collapse_ratio": 0.7},
    "slo_gates": {
        "ttft_p95_max_s": 5.0,
        "itl_p95_max_s": 0.2,
        "e2e_p95_max_s": 30.0,
        "error_rate_max_pct": 1.0,
        "declared_before_run": True,
    },
    "output": {
        "bundle_dir": "bundle",
        "report_path": "report.json",
        "engine_logs_path": "engine.log",
        "container_digest": None,
    },
}

#: The four pinned declaration documents, read from the repository's own worked example
#: rather than invented here. `rereduce` validates each one against its schema before it
#: will rebuild anything, so a hand-written stub would have to be re-hand-written every time
#: a schema gains a required field -- and until someone did, every test in this file would
#: fail on the fixture and say nothing about re-reduction. Reading the example keeps the
#: fixture valid by construction and keeps these tests about the rebuild.
_EXAMPLE_DECLARATIONS = Path(__file__).resolve().parents[1] / "examples" / "bench-config"
_DECLARATIONS = {
    layer: json.loads((_EXAMPLE_DECLARATIONS / f"{layer}.json").read_text(encoding="utf-8"))
    for layer in ("hardware", "model", "serving", "workload")
}

#: The run_configs.json workload block, keys in the writer's own order. rebuild_report
#: reads the seed from here, and a workload block missing the writer's keys would be a
#: pinned file the harness never wrote.
_RUN_WORKLOAD = {
    "run_label": "qwen3-32b-gb200x4-tp4-synthetic",
    "seed": _SEED,
    "sampler": "deterministic-generated",
    "cache_policy": "unique-prefix",
    "think_time_s": 1.5,
    "output_basis": "fixed",
    "ignore_eos": True,
    "corpus_name": "synthetic:128",
    "corpus_digest": hashlib.sha256(b"ascep-rereduce-test-corpus").hexdigest(),
    "corpus_size": None,
    "corpus_field": None,
    "media_placeholders_stripped": False,
    "prefix_adds_tokens": False,
    "temperature": None,
    "output_tokens": 5,
}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, doc) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(doc, fp, indent=2)
        fp.write("\n")


def _record_dict(concurrency, repetition, tag, t_issue, *, warmup=False):
    # The keys are RequestRecord's own field names, because read_records feeds the parsed
    # line straight into the dataclass. A convenience spelling here would make these tests
    # exercise a record shape the harness never writes.
    first_token = t_issue + 0.5
    return {
        "request_id": f"c{concurrency}-r{repetition}-{tag}",
        "issued_ts": t_issue,
        "outcome": "ok",
        "connect_ts": t_issue + 0.01,
        "first_token_ts": first_token,
        # token_ts holds every token after the first, so four entries plus the first
        # token make the five tokens output_tokens claims; a disagreement would make
        # the reconciliation figures nonsense.
        "token_ts": [first_token + 0.05 * step for step in range(1, 5)],
        "end_ts": first_token + 0.2,
        "input_tokens": 128,
        "output_tokens": 5,
        "concurrency": concurrency,
        "repetition": repetition,
        "session_id": None,
        # A warm-up record is one the driver issued before the window opened. reduce_window
        # excludes it itself, so it must survive reconstruction rather than be filtered here.
        "in_window": not warmup,
    }


def _window_specs():
    """Nine windows over three rungs, in execution order: the search ran 4, then 2, then 6.

    The order is deliberately neither ascending nor descending. Reconstruction that sorted the
    windows would still satisfy an ascending fixture, and the bug it would hide -- windows
    replayed in an order the run never had -- is the one that silently changes which repetition
    a rung publishes.
    """
    specs = []
    t0 = _T0
    for concurrency in (4, 2, 6):
        for repetition in range(3):
            records = [_record_dict(concurrency, repetition, "warmup", t0 - 0.8, warmup=True)]
            for index in range(3):
                records.append(_record_dict(concurrency, repetition, str(index), t0 + 1.0 + index))
            specs.append(
                {
                    "t0": t0,
                    "window_s": 30.0,
                    "drain_deadline_s": 10.0,
                    "warmup_count": 1,
                    "warmup_s_actual": 0.8,
                    "policy": {
                        "concurrency": concurrency,
                        "window_s": 30.0,
                        "drain_deadline_s": 10.0,
                        "think_time_s": 1.5,
                        "warmup_requests": 1,
                        "warmup_s": 0.0,
                        "loop": "closed",
                        "repetition": repetition,
                    },
                    "boundary": {
                        # offered splits into completed_in_window plus straddlers, and
                        # warmup restates the window's warmup_count; a boundary that
                        # contradicts its window is not a bundle the harness could have
                        # written, so a test built on one would prove nothing.
                        "t0": t0,
                        "window_s": 30.0,
                        "drain_deadline_s": 10.0,
                        "deadline": t0 + 30.0 + 10.0,
                        "offered": 3,
                        "warmup": 1,
                        "completed_in_window": 3,
                        "straddlers": 0,
                        "abandoned": 0,
                    },
                    "records": records,
                }
            )
            t0 += 100.0
    return specs


def _write_manifest(bundle_dir: Path) -> None:
    """Pin every file by sha256, the way the writer does.

    The fixture asserts verify_bundle passes on the result, so a drift in the manifest
    format fails here, loudly, instead of surfacing as thirteen unrelated refusals.
    """
    entries = {}
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries[path.relative_to(bundle_dir).as_posix()] = digest
    _write_json(bundle_dir / "manifest.json", {"sha256": entries})


def _write_bundle(bundle_dir: Path, specs) -> None:
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "declarations").mkdir()
    _write_json(bundle_dir / "bench-config.json", _BENCH_CONFIG)
    for layer, doc in _DECLARATIONS.items():
        _write_json(bundle_dir / "declarations" / f"{layer}.json", doc)
    _write_json(
        bundle_dir / "run_configs.json",
        {
            "workload": _RUN_WORKLOAD,
            "windows": [
                {key: value for key, value in spec.items() if key != "records"} for spec in specs
            ],
        },
    )
    with (bundle_dir / "records.jsonl").open("w", encoding="utf-8") as fp:
        for spec in specs:
            for record in spec["records"]:
                fp.write(json.dumps(record) + "\n")
    _write_json(bundle_dir / "environment.json", {"python": "3.12", "gpus": 4})
    (bundle_dir / "engine.log").write_text("engine started\n", encoding="utf-8")
    _write_manifest(bundle_dir)


def _edit_records(bundle_dir: Path, edit) -> None:
    path = bundle_dir / "records.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(records)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


def _window_runs(specs):
    """The WindowRun objects the original run held, rebuilt straight from the specs."""
    runs = []
    for spec in specs:
        text = "".join(json.dumps(record) + "\n" for record in spec["records"])
        runs.append(
            WindowRun(
                records=read_records(io.StringIO(text)),
                policy=WindowPolicy(**spec["policy"]),
                t0=spec["t0"],
                window_s=spec["window_s"],
                drain_deadline_s=spec["drain_deadline_s"],
                warmup_count=spec["warmup_count"],
                warmup_s_actual=spec["warmup_s_actual"],
                dephase_s=spec.get("dephase_s"),
                boundary=Boundary(**spec["boundary"]),
            )
        )
    return runs


def _original_report(bundle_dir: Path, runs):
    """Build the report the way the original run did: the harness's own assembly."""
    config, _raw = run.load_config(bundle_dir / "bench-config.json")
    declarations = {
        layer: _read_json(bundle_dir / "declarations" / f"{layer}.json")
        for layer in ("hardware", "model", "serving", "workload")
    }
    gates, policy = run.ladder_policy(config)
    repetitions = {}
    for window_run in runs:
        summary = reduce_window(
            window_run.records,
            window_s=window_run.window_s,
            t0=window_run.t0,
            gates=gates,
            seed=_SEED,
        )
        repetitions.setdefault(window_run.policy.concurrency, []).append(
            RepetitionResult(
                concurrency=window_run.policy.concurrency,
                repetition=window_run.policy.repetition,
                summary=summary,
                post_search=window_run.policy.repetition >= policy.repetitions,
            )
        )
    result = grade_ladder(repetitions, policy, censoring_cause=None)
    reproduction = {
        "run_configs_path": f"{bundle_dir.name}/run_configs.json",
        "raw_records_path": f"{bundle_dir.name}/records.jsonl",
        "engine_logs_path": f"{bundle_dir.name}/engine.log",
        "environment_capture_path": f"{bundle_dir.name}/environment.json",
        "container_digest": "sha256:" + "0" * 64,
    }
    return report.build_report(config, declarations, runs, repetitions, result, reproduction, None)


def _copy_bundle(bundle_dir: Path, tmp_path: Path) -> Path:
    # Keep the directory name: reproduction.raw_records_path resolves against the
    # bundle's parent, so a renamed copy would fail the citation check for reasons that
    # have nothing to do with the behaviour under test.
    dest = tmp_path / bundle_dir.name
    shutil.copytree(bundle_dir, dest)
    return dest


@pytest.fixture(scope="module")
def built_bundle(tmp_path_factory):
    """One real bundle and the report it backs, built once for the whole module.

    Every test copies this bundle into its own tmp_path and damages exactly one thing in
    the copy; damaging a copy is what keeps the refusal tests independent of each other.
    """
    bundle_dir = tmp_path_factory.mktemp("evidence") / "bundle"
    specs = _window_specs()
    _write_bundle(bundle_dir, specs)
    report = _original_report(bundle_dir, _window_runs(specs))
    # Play a report written before today's reduction: it carries none of the figures
    # the new rule newly emits. The rows live under "run", where build_report puts them --
    # a fixture reaching for a top-level "results" would be damaging a report shape the
    # harness has never written, and its refusal tests would fire on the wrong key.
    for row in report["run"]["results"]:
        for key in _NEWLY_EMITTED_ROW_KEYS:
            row.pop(key, None)
    assert persist.verify_bundle(bundle_dir) == []
    return bundle_dir, report


def test_a_bundle_rereduces_to_the_same_figures_it_was_built_from(built_bundle, tmp_path):
    """A rebuild that misread a WindowPolicy, dropped a warm-up record, or returned
    windows out of declaration order would publish figures the pinned evidence does not
    produce, and the report would read as the old run's measurement while being a new,
    different reduction nobody ran."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    rebuilt = rebuild_report(dest, previous_report=report)
    for old_row, new_row in zip(report["run"]["results"], rebuilt["run"]["results"], strict=True):
        for key, value in old_row.items():
            assert new_row[key] == value, key
        # A subset, not an equality: a rung publishes `dispersion` when it has repetitions to
        # disperse and `dispersion_u_reason` when it does not, never both, so demanding both
        # would fail a correct rebuild. What must hold is that the rebuild added nothing
        # else -- a new key nobody declared is a figure the pinned evidence never produced.
        assert set(new_row) - set(old_row) <= set(_NEWLY_EMITTED_ROW_KEYS)
        assert set(new_row) - set(old_row)


def test_windows_come_back_in_declaration_order(built_bundle, tmp_path):
    """Declaration order is execution order, and the median-repetition picker's stable
    sort depends on it: shuffled windows would silently publish a different repetition's
    figures per rung while the report still cited the same bundle."""
    bundle_dir, _report = built_bundle
    runs = load_window_runs(_copy_bundle(bundle_dir, tmp_path))
    order = [(window_run.policy.concurrency, window_run.policy.repetition) for window_run in runs]
    assert order == [
        (4, 0),
        (4, 1),
        (4, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (6, 0),
        (6, 1),
        (6, 2),
    ]
    assert order != sorted(order)


def test_warmup_records_stay_in_the_reconstructed_window(built_bundle, tmp_path):
    """reduce_window excludes warm-up records itself; pre-filtering them here would
    change the very reduction this module exists to repeat, and the error-rate
    denominators with it."""
    bundle_dir, _report = built_bundle
    runs = load_window_runs(_copy_bundle(bundle_dir, tmp_path))
    for window_run in runs:
        assert len(window_run.records) == window_run.warmup_count + 3
        assert window_run.records[0].request_id.endswith("-warmup")


def test_a_record_belonging_to_no_declared_window_is_refused_not_dropped(built_bundle, tmp_path):
    """Dropping the record would move the error-rate denominator of the window it
    belonged to, publishing a measured-looking error rate over evidence the rebuild had
    quietly amended."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    with (dest / "records.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(_record_dict(8, 0, "stray", _T0 + 0.5)) + "\n")
    with pytest.raises(ReduceError, match="group onto no declared window"):
        load_window_runs(dest)


def test_a_bundle_that_fails_manifest_verification_is_refused(built_bundle, tmp_path):
    """Re-reducing unpinned bytes is laundering: the rebuilt report would claim to
    derive from evidence nobody can identify."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    with (dest / "engine.log").open("a", encoding="utf-8") as fp:
        fp.write("a line the run never wrote\n")
    with pytest.raises(ReduceError, match="fails manifest verification"):
        rebuild_report(dest, previous_report=report)


def test_a_report_whose_ladder_was_censored_is_refused(built_bundle, tmp_path):
    """The censoring cause lived only in the process that died; a rebuild would promote
    a truncated ladder's lower bounds into figures that no longer say they are lower
    bounds."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    previous = copy.deepcopy(report)
    previous["conformance_note"] = (
        "The ladder was censored (the engine died at concurrency 8); "
        "the remaining figures are lower bounds."
    )
    with pytest.raises(ReduceError, match="censoring cause is not pinned"):
        rebuild_report(dest, previous_report=previous)


def test_a_report_citing_a_different_bundles_records_is_refused(built_bundle, tmp_path):
    """Otherwise one run's evidence stands behind another run's figures, and neither
    report can be checked against the bytes it cites."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    previous = copy.deepcopy(report)
    previous["reproduction"]["raw_records_path"] = "some-other-bundle/records.jsonl"
    with pytest.raises(ReduceError, match="cites raw records at"):
        rebuild_report(dest, previous_report=previous)


def test_a_report_naming_no_raw_records_path_is_refused(built_bundle, tmp_path):
    """With nothing to check the bundle against, any bundle at all could stand behind
    the rebuilt figures."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    previous = copy.deepcopy(report)
    previous["reproduction"]["raw_records_path"] = None
    with pytest.raises(ReduceError, match=r"names no reproduction\.raw_records_path"):
        rebuild_report(dest, previous_report=previous)


def test_run_configs_with_no_windows_is_refused(built_bundle, tmp_path):
    """A bundle with nothing to reduce is evidence of nothing; emitting a report from
    it would fabricate a measurement no run took."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    doc = _read_json(dest / "run_configs.json")
    doc["windows"] = []
    _write_json(dest / "run_configs.json", doc)
    with pytest.raises(ReduceError, match="declares no windows"):
        load_window_runs(dest)


def test_a_window_missing_a_field_the_reduction_needs_is_refused(built_bundle, tmp_path):
    """Defaulting the field would invent a declaration the run never made: the window
    would have been measured under one policy and reported under another."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    doc = _read_json(dest / "run_configs.json")
    del doc["windows"][0]["warmup_count"]
    _write_json(dest / "run_configs.json", doc)
    with pytest.raises(ReduceError, match="missing fields the reduction needs"):
        load_window_runs(dest)


def test_a_bundle_with_no_pinned_workload_seed_is_refused(built_bundle, tmp_path):
    """The reduction reuses one deterministic seed end to end; inventing one here would
    make the rebuild differ from the run it claims to repeat while presenting the
    figures as that run's own."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    doc = _read_json(dest / "run_configs.json")
    del doc["workload"]["seed"]
    _write_json(dest / "run_configs.json", doc)
    # Re-pin the edited file so manifest verification passes and the seed refusal is
    # the one that fires.
    _write_manifest(dest)
    with pytest.raises(ReduceError, match="pins no usable workload seed"):
        rebuild_report(dest, previous_report=report)


def test_session_records_without_persisted_session_counters_are_refused(built_bundle, tmp_path):
    """Distinct session ids cannot tell a session truncated at window close from a
    completed one, so the counts would have to be guessed, and the rebuilt report would
    publish completion figures the bundle does not pin."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)

    def give_session_ids(records):
        for record in records:
            if (record["concurrency"], record["repetition"]) == (4, 0):
                record["session_id"] = "session-1"

    # The fixture's windows carry no session counters, the way a bundle written before
    # counter persistence would not, so session ids on the records are the whole damage.
    _edit_records(dest, give_session_ids)
    with pytest.raises(ReduceError, match="predates session-count persistence"):
        load_window_runs(dest)


def test_a_window_with_no_session_ids_rebuilds_with_counters_of_zero(built_bundle, tmp_path):
    """A window whose records carry no session id has nothing to guess: refusing it
    would make every bundle that predates counter persistence un-re-reducible for no
    dishonesty, while inventing nonzero counters would fabricate sessions that never
    ran."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    # The fixture's windows carry neither sessions_started nor sessions_completed and
    # no record carries a session id, so the bundle already predates counter persistence.
    runs = load_window_runs(dest)
    assert all(
        window_run.sessions_started == 0 and window_run.sessions_completed == 0
        for window_run in runs
    )
    rebuilt = rebuild_report(dest, previous_report=report)
    assert rebuilt["run"]["results"]


def test_the_rebuild_does_not_modify_the_bundle(built_bundle, tmp_path):
    """A reducer that edited the evidence while deriving from it would make the
    manifest's promise -- these are the bytes that ran -- unkeepable, and the rebuilt
    report would cite bytes that no longer exist."""
    bundle_dir, report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    manifest_before = (dest / "manifest.json").read_bytes()
    rebuild_report(dest, previous_report=report)
    assert persist.verify_bundle(dest) == []
    assert (dest / "manifest.json").read_bytes() == manifest_before


def _graded_beside(bundle_dir: Path, report: dict) -> Path:
    """Write ``report`` where reduce looks for it, graded the way a published report is.

    The module fixture deliberately plays a report older than today's reduction, so it is
    not the input for a --check test: the rebuild would differ at `run` for a real reason
    and the test would pass while proving nothing about the grade. These two tests build
    the report from the same evidence under today's rules first.
    """
    report = copy.deepcopy(report)
    conformance.raise_claim(report, conformance.check(report))
    path = bundle_dir.parent / "report.json"
    _write_json(path, report)
    return path


def _current_report(bundle_dir: Path) -> dict:
    return _original_report(bundle_dir, _window_runs(_window_specs()))


def test_check_passes_on_a_graded_report_because_the_grade_is_not_a_figure(built_bundle, tmp_path):
    """Every published report is graded and every rebuild is an ungraded draft, so a
    --check that compared `conformance` would report a difference on every example in
    this repository -- and an operator who sees --check fail on the worked examples
    learns to ignore it, which is worse than not shipping the check at all. The grade is
    computed from the figures by a separate command, so figures matching is the whole
    claim --check is entitled to make."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    path = _graded_beside(dest, _current_report(dest))
    assert json.loads(path.read_text(encoding="utf-8"))["conformance"] != "non-conforming"
    assert main(["reduce", str(dest), "--check"]) == 0


def test_check_still_fails_when_a_published_figure_no_longer_reduces_from_the_evidence(
    built_bundle, tmp_path
):
    """Excluding the grade must not blunt the check. A report claiming a throughput its
    own records do not produce is exactly what --check exists to catch, and a fold that
    swallowed it would leave the command reporting success on a fabricated figure."""
    bundle_dir, _report = built_bundle
    dest = _copy_bundle(bundle_dir, tmp_path)
    tampered = _current_report(dest)
    tampered["run"]["results"][0]["output_tok_s"] += 1.0
    path = _graded_beside(dest, tampered)
    written = path.read_bytes()
    assert main(["reduce", str(dest), "--check"]) == 1
    # Refusing is not enough: the bundle and the report it grades must both be left alone,
    # or a failed check would itself be the edit that made the next one pass.
    assert persist.verify_bundle(dest) == []
    assert path.read_bytes() == written
