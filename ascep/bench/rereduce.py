"""Re-reduce a bundle: apply today's reduction to evidence a run already paid for.

The reduction that turns raw per-request records into published per-rung rows is code,
and code gets fixed. When it is, every report reduced under the old rule stops matching
what the new rule derives from the report's own bundle, and the honest ways forward are
to re-measure or to re-reduce. Re-measuring burns the GPU hours again; re-reducing is
possible at all only because the bundle pins every byte the reduction reads. This module
is the second path: a bundle must be sufficient to regenerate the report it backs.

The assembly is not reimplemented here. ``run.load_config``, ``run._ladder_policy`` and
``run._build_report`` are private names and are imported anyway, deliberately: a second
copy of the assembly would drift from the harness the first time either was edited, and
a rebuilt report that differs from a freshly measured one in anything but its
measurements is worse than no rebuild at all.

Nothing in this module writes to the bundle. A reducer that modified the evidence while
deriving from it would make the manifest's promise -- these are the bytes that ran --
unkeepable.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from ascep.bench import persist, run
from ascep.bench.driver import Boundary, WindowPolicy, WindowRun
from ascep.bench.ladder import RepetitionResult, grade_ladder
from ascep.bench.metrics import reduce_window
from ascep.bench.records import RequestRecord, read_records
from ascep.validation import validate


class ReduceError(Exception):
    """A refusal to re-reduce: the bundle or the report cannot honestly back a rebuild."""


#: The four layer documents a run is bound to. They are read from the bundle's own
#: declarations/ directory rather than through run.load_declarations, which resolves the
#: config's filenames against a config dir and would therefore validate today's files --
#: the operator's copies may have drifted since the run, and the pinned copies are the
#: ones the measurement was actually taken under.
_DECLARATION_LAYERS = ("hardware", "model", "serving", "workload")

#: The five reproduction keys write_bundle returns. They are carried forward from the
#: previous report rather than recomputed: the one that can be checked against the
#: bundle (raw_records_path) is checked before this mapping is built, and the other
#: four describe the same bundle. A missing key becomes None, which is how
#: _build_report reads them.
_C8_REPRODUCTION_KEYS = (
    "run_configs_path",
    "raw_records_path",
    "engine_logs_path",
    "environment_capture_path",
    "container_digest",
)

#: Boundary is persisted as a plain mapping whose shape the writer owns. Rebuilding
#: from exactly the declared fields keeps a writer-side addition from surfacing in this
#: read path as a TypeError an operator cannot act on.
_BOUNDARY_FIELDS = {f.name for f in fields(Boundary)}


def _read_run_configs(bundle_dir: Path) -> dict[str, Any]:
    """Read the bundle's pinned copy of the window declarations.

    This is the file that says where every window opened and how long it ran; without
    it the records are timings in search of their denominators, so anything unreadable
    or malformed is a refusal, never a default.
    """
    path = bundle_dir / "run_configs.json"
    try:
        with path.open("r", encoding="utf-8") as fp:
            doc = json.load(fp)
    except OSError as exc:
        raise ReduceError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReduceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ReduceError(f"{path} is not a JSON object; it cannot declare any windows")
    return doc


def _load_declarations(bundle_dir: Path) -> dict[str, Any]:
    """Read and schema-validate the bundle's pinned copies of the four declarations."""
    declarations: dict[str, Any] = {}
    for layer in _DECLARATION_LAYERS:
        path = bundle_dir / "declarations" / f"{layer}.json"
        try:
            with path.open("r", encoding="utf-8") as fp:
                doc = json.load(fp)
        except OSError as exc:
            raise ReduceError(f"cannot read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ReduceError(f"{path} is not valid JSON: {exc}") from exc
        problems = validate(layer, doc)
        if problems:
            raise ReduceError(f"{path} does not satisfy the {layer} schema: " + "; ".join(problems))
        declarations[layer] = doc
    return declarations


def load_window_runs(bundle_dir) -> list[WindowRun]:
    """Reconstruct the executed windows from the bundle's pinned files.

    Records group onto windows by the (concurrency, repetition) pair every record
    carries, and the windows come back in run_configs.json declaration order, because
    declaration order is execution order and the median-repetition picker's stable sort
    depends on it. Warm-up records stay in the WindowRun: reduce_window excludes them
    itself, and pre-filtering here would change the reduction this module exists to
    repeat.
    """
    bundle_dir = Path(bundle_dir)
    run_configs = _read_run_configs(bundle_dir)
    windows = run_configs.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ReduceError(
            f"{bundle_dir / 'run_configs.json'} declares no windows; "
            "a bundle with nothing to reduce is evidence of nothing"
        )
    records_path = bundle_dir / "records.jsonl"
    try:
        with records_path.open("r", encoding="utf-8") as fp:
            records = read_records(fp)
    except OSError as exc:
        raise ReduceError(f"cannot read {records_path}: {exc}") from exc

    by_window: dict[tuple[int, int], list[RequestRecord]] = {}
    for record in records:
        by_window.setdefault((record.concurrency, record.repetition), []).append(record)

    runs: list[WindowRun] = []
    for index, window in enumerate(windows):
        try:
            # Bundles written before de-phasing existed carry no `dephase` key, and the
            # dataclass default is True -- which would reconstruct a lock-step window as a
            # de-phased one and answer "was that plateau floor(W / C)?" with the wrong
            # word. An absent key is the pre-fix harness, so it reconstructs as False.
            policy_fields = dict(window["policy"])
            policy_fields.setdefault("dephase", False)
            policy = WindowPolicy(**policy_fields)
            boundary = Boundary(
                **{k: v for k, v in window["boundary"].items() if k in _BOUNDARY_FIELDS}
            )
            t0 = window["t0"]
            window_s = window["window_s"]
            drain_deadline_s = window["drain_deadline_s"]
            warmup_count = window["warmup_count"]
            warmup_s_actual = window["warmup_s_actual"]
            # .get, not [], for the same reason: an older bundle simply has no interval to
            # report, and None already means "entered in lock-step" downstream.
            dephase_s = window.get("dephase_s")
        except (KeyError, TypeError) as exc:
            raise ReduceError(
                f"window {index} in {bundle_dir / 'run_configs.json'} is missing fields "
                f"the reduction needs ({exc}); a window whose declarations are partial "
                "cannot be re-reduced, only re-measured"
            ) from exc
        window_records = by_window.pop((policy.concurrency, policy.repetition), [])
        started = window.get("sessions_started")
        completed = window.get("sessions_completed")
        if started is None or completed is None:
            # Older bundles never persisted the session counters. Zero is only safe when
            # no record carries a session id; otherwise the counts would have to be
            # guessed from distinct ids, which silently counts a session truncated at
            # window close as a completed one.
            if any(record.session_id is not None for record in window_records):
                raise ReduceError(
                    f"window concurrency={policy.concurrency} "
                    f"repetition={policy.repetition} contains session records but the "
                    "bundle predates session-count persistence, so this session run's "
                    "own report cannot be honestly rebuilt from it"
                )
            started = 0
            completed = 0
        runs.append(
            WindowRun(
                records=window_records,
                policy=policy,
                t0=t0,
                window_s=window_s,
                drain_deadline_s=drain_deadline_s,
                warmup_count=warmup_count,
                warmup_s_actual=warmup_s_actual,
                dephase_s=dephase_s,
                boundary=boundary,
                sessions_started=started,
                sessions_completed=completed,
            )
        )

    if by_window:
        leftover = ", ".join(
            f"(concurrency={c}, repetition={r})" for c, r in sorted(by_window, key=str)
        )
        raise ReduceError(
            f"records in {records_path} group onto no declared window: {leftover}; "
            "dropping them would move the error-rate denominators of the windows they "
            "belong to, so the bundle is refused rather than reduced around them"
        )
    return runs


def rebuild_report(bundle_dir, *, previous_report: dict) -> dict:
    """Derive the report this bundle backs under the current reduction rules.

    Every step that cannot be done honestly refuses with ReduceError rather than
    guessing, because every guess here produces a report that looks measured while
    being something else.
    """
    bundle_dir = Path(bundle_dir)

    # Re-reducing unpinned bytes is laundering: the whole value of the rebuilt report
    # is that it derives from evidence somebody can still identify.
    problems = persist.verify_bundle(bundle_dir)
    if problems:
        raise ReduceError(
            f"bundle {bundle_dir} fails manifest verification: " + "; ".join(problems)
        )

    # A censored ladder's censoring cause lived only in the process that died; the
    # bundle does not pin it, and a rebuild would quietly promote a truncated ladder's
    # lower bounds into figures that no longer say they are lower bounds.
    note = previous_report.get("conformance_note") or ""
    if "The ladder was censored (" in note:
        raise ReduceError(
            "the previous report's conformance_note says the ladder was censored, and "
            "the censoring cause is not pinned in the bundle; refusing to rebuild "
            "rather than promote a truncated ladder's lower bounds into unmarked "
            "figures"
        )

    # A report may only be rebuilt from the bundle it already cites; anything else
    # would let one run's evidence stand behind another run's figures.
    reproduction = previous_report.get("reproduction") or {}
    raw_records_path = reproduction.get("raw_records_path")
    records_path = bundle_dir / "records.jsonl"
    if not raw_records_path:
        raise ReduceError(
            "the previous report names no reproduction.raw_records_path, so there is "
            "nothing to check this bundle against"
        )
    cited = (bundle_dir.parent / raw_records_path).resolve()
    if cited != records_path.resolve():
        raise ReduceError(
            f"the previous report cites raw records at {raw_records_path} ({cited}), "
            f"not this bundle's {records_path.resolve()}"
        )

    # The bundle's copy of the config is the manifest-pinned one; the operator's
    # original may have drifted since the run, and the pinned copy is the one the
    # measurement was actually taken under.
    try:
        config, _raw = run.load_config(bundle_dir / "bench-config.json")
    except run.ConfigError as exc:
        raise ReduceError(f"the bundle's pinned bench config does not load: {exc}") from exc

    declarations = _load_declarations(bundle_dir)
    gates, policy = run._ladder_policy(config)

    run_configs = _read_run_configs(bundle_dir)
    try:
        seed = int(run_configs["workload"]["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReduceError(
            f"{bundle_dir / 'run_configs.json'} pins no usable workload seed ({exc}); "
            "the reduction reuses one deterministic seed end to end, and inventing one "
            "here would make the rebuild differ from the run it claims to repeat"
        ) from exc

    runs = load_window_runs(bundle_dir)

    repetitions: dict[int, list[RepetitionResult]] = {}
    for window_run in runs:
        summary = reduce_window(
            window_run.records,
            window_s=window_run.window_s,
            t0=window_run.t0,
            gates=gates,
            seed=seed,
        )
        # outcome and reason are not persisted in the bundle; they carry verdicts the
        # reduction itself cannot see, so the rebuild leaves them at their defaults
        # rather than inventing a verdict. post_search marks the section 5
        # confirmation window -- additional to the three repetitions, never instead of
        # them -- and grading partitions on it.
        repetitions.setdefault(window_run.policy.concurrency, []).append(
            RepetitionResult(
                concurrency=window_run.policy.concurrency,
                repetition=window_run.policy.repetition,
                summary=summary,
                post_search=window_run.policy.repetition >= policy.repetitions,
            )
        )

    result = grade_ladder(repetitions, policy, censoring_cause=None)

    c8 = {key: reproduction.get(key) for key in _C8_REPRODUCTION_KEYS}
    return run._build_report(config, declarations, runs, repetitions, result, c8, None)
