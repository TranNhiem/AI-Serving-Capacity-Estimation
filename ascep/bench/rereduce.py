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
from dataclasses import dataclass, fields
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

#: Per-rung figures each elided records field backs, keyed by the field half of an
#: elision key. token_ts is why the map exists: with the stamps emptied, reduce_window
#: finds no pooled gaps and silently falls back to the per-request means, which cannot
#: see a stall inside one request -- the exact failure an ITL gate exists to catch. On
#: the 26B campaign that substitution moved the concurrency-384 and concurrency-512
#: rungs from measured gate failures to passes, so slo_pass and the verdict fields
#: derived from it are listed beside the percentiles it corrupts.
_ELIDED_FIELD_IMPACT: dict[str, tuple[str, ...]] = {
    "token_ts": (
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
    ),
}

#: The report-level sections that rest on the per-rung verdicts above and are tainted
#: with them. capacity_tiers is computed from the graded ladder and
#: unmeasured_assumptions discloses what the grading was scored against, so an elided
#: bundle cannot confirm either even when every untainted figure still matches.
_ELIDED_REPORT_SECTIONS: tuple[str, ...] = (
    "capacity_tiers",
    "unmeasured_assumptions",
)

#: Per-rung keys holding a nested dict that is itself keyed by figure name. A tainted
#: figure is stripped from inside these too, because dispersion's itl_p95_s spread comes
#: from the same elided stamps as the row's own itl_p95_s. Only the tainted keys inside
#: are dropped, never the whole block: the ttft, e2e and throughput spreads beside them
#: reproduce exactly, and over-skipping is how a partial check becomes a false pass.
_FIGURE_KEYED_SUBBLOCKS: tuple[str, ...] = ("dispersion",)

_U_REASON_SUFFIX = "_u_reason"


def _grading_elisions(bundle_dir: Path) -> dict[str, str]:
    """The bundle's declared elisions that back a graded figure, mapped to their reasons.

    Only a ``records.jsonl:<field>`` key can taint: every figure the reduction
    publishes is derived from the records, so an elision naming any other file is
    acknowledged but cannot move a number the report rests on. The rule lives here
    exactly once because rebuild_report and check_report must never disagree about it —
    two copies of a safety rule drift the first time one is edited.

    Every records field taints, including one _ELIDED_FIELD_IMPACT has no entry for.
    Membership in that map is knowledge of the blast radius, not a condition for having
    one: treating an unmapped field as harmless would mean the next field somebody
    elides is silently exempt from the refusal this function exists to raise.
    """
    tainted: dict[str, str] = {}
    for key, reason in persist.load_elisions(bundle_dir).items():
        filename, sep, field_name = key.partition(":")
        if sep and filename == "records.jsonl" and field_name:
            tainted[key] = reason
    return tainted


def _unmapped_elisions(grading: dict[str, str]) -> tuple[str, ...]:
    """The tainting elisions whose effect on the report this version cannot scope.

    A partial check certifies that every figure it did NOT skip still matches. That
    claim needs the list of figures the elided field backs, so an elision of a field
    _ELIDED_FIELD_IMPACT does not describe makes the partial check unsound rather than
    merely narrower — it would certify figures that may well have moved.
    """
    return tuple(
        sorted(key for key in grading if key.partition(":")[2] not in _ELIDED_FIELD_IMPACT)
    )


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


def _verify_bundle_or_refuse(bundle_dir: Path) -> None:
    """Refuse unless the bundle's manifest verifies.

    Re-reducing unpinned bytes is laundering: the whole value of the rebuilt report
    is that it derives from evidence somebody can still identify. One copy serves both
    public entry points, because two copies of a safety rule drift the first time one
    is edited.
    """
    problems = persist.verify_bundle(bundle_dir)
    if problems:
        raise ReduceError(
            f"bundle {bundle_dir} fails manifest verification: " + "; ".join(problems)
        )


def rebuild_report(bundle_dir, *, previous_report: dict) -> dict:
    """Derive the report this bundle backs under the current reduction rules.

    Every step that cannot be done honestly refuses with ReduceError rather than
    guessing, because every guess here produces a report that looks measured while
    being something else.
    """
    bundle_dir = Path(bundle_dir)
    _verify_bundle_or_refuse(bundle_dir)

    # Checked here -- after verification, before the censored-ladder refusal inside
    # _derive_report -- so it reads as one more entry in the same refusal sequence. An
    # elision the reduction cannot see is worse than one it refuses: with token_ts
    # emptied, reduce_window finds no pooled gaps and silently falls back to the
    # per-request means, which cannot see a stall inside one request, so a rebuilt
    # ladder is graded against a substituted ITL population and can pass rungs the
    # measured run failed. This path writes that report, so it refuses rather than
    # publish a more permissive capacity from strictly less evidence; check_report
    # gets past the same evidence because it never lets the substituted values out.
    grading = _grading_elisions(bundle_dir)
    if grading:
        declared = "; ".join(
            f"{key} -- the manifest declares: {reason}" for key, reason in sorted(grading.items())
        )
        raise ReduceError(
            f"bundle {bundle_dir} declares elisions that back graded figures: {declared}; "
            "a rebuild would re-grade the ladder against the substituted "
            "per-request-mean ITL population and publish a more permissive capacity "
            "than was measured, so the published report keeps its measured figures"
        )
    return _derive_report(bundle_dir, previous_report=previous_report)


def _derive_report(bundle_dir, *, previous_report: dict) -> dict:
    """The derivation rebuild_report publishes and check_report only tests.

    The caller must already have run _verify_bundle_or_refuse. Every step that cannot
    be done honestly refuses with ReduceError rather than guessing, because every
    guess here produces a report that looks measured while being something else.
    check_report is allowed past the elision refusal rebuild_report enforces above: it
    never returns, writes or prints the values an elision taints — it only asks
    whether the untainted ones still agree — so the one honest question left on an
    elided bundle, "does everything the bundle still contains still reproduce?", is
    not refused alongside the dishonest one.
    """
    bundle_dir = Path(bundle_dir)

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


@dataclass(frozen=True)
class CheckResult:
    """What checking a published report against its bundle found.

    ``differing`` holds the top-level paths whose values no longer agree, plus
    ``run.results`` when the two sides disagree even on which rungs exist. ``skipped``
    maps each figure a bundle elision makes uncheckable to the manifest's own declared
    reason, so the operator reads why it could not be checked, not merely that it was
    not. ``elided`` is set only when a grading-tainting elision forced the partial
    comparison; a bundle whose elisions name files the reduction never reads still
    gets the full check and reports False.

    A figure named in ``skipped`` is skipped everywhere it appears in a rung row: as the
    value, as its ``_u_reason`` when the rebuild has to explain an absence the measured
    run never had, and as its entry inside the row's dispersion block. That is stated
    here because a scope wider than the printed list would otherwise be invisible to the
    operator reading the list.
    """

    differing: tuple[str, ...]
    skipped: dict[str, str]
    elided: bool


def _ungraded(report: dict) -> dict:
    """A copy of ``report`` with the grade folded back to how `ascep bench` writes it.

    A rebuild is an ungraded draft by construction, so every graded report differs from
    its own rebuild at `conformance` and at the note's opening paragraph -- which would
    make --check report a difference on every published example in this repository and
    teach operators that a failing --check is normal. The grade is not a figure: it is
    computed from the figures by a different command, so if the figures match,
    re-grading reproduces it. Folding rather than skipping keeps the rest of the note in
    the comparison, because everything after the opening paragraph is written by the run
    and is a figure.
    """
    from ascep.conformance import DRAFT_NOTE, GRADED_NOTE

    folded = dict(report)
    folded.pop("conformance", None)
    note = folded.get("conformance_note")
    if isinstance(note, str) and note.startswith(GRADED_NOTE):
        folded["conformance_note"] = DRAFT_NOTE + note[len(GRADED_NOTE) :]
    return folded


def _top_level_diffs(
    published: dict, rebuilt: dict, *, skip: frozenset[str] = frozenset()
) -> list[str]:
    """Top-level keys whose values disagree, absence on either side included.

    This is the comparison the CLI's --check message speaks for, so it lives beside the
    rebuild rather than beside the printing: a second copy in cli.py would drift from
    this one, and the two exclusions below are the ones the message promises the
    operator by name. report_generated_utc is excluded because a rebuild is genuinely
    generated at a new time, so comparing it would make --check fail by construction;
    the grade is excluded by _ungraded for the reason given there. Nothing else is
    excluded, because anything else that differs is the point of the check.
    """
    published, rebuilt = _ungraded(published), _ungraded(rebuilt)
    differing: list[str] = []
    for key in sorted(set(published) | set(rebuilt)):
        if key in skip or key == "report_generated_utc":
            continue
        if key not in published or key not in rebuilt or published[key] != rebuilt[key]:
            differing.append(key)
    return differing


def _is_tainted(field: str, figures: set[str]) -> bool:
    """Whether a per-rung key carries a tainted figure's value or its absence.

    A ``<figure>_u_reason`` is written by the same reduction step from the same missing
    evidence: with the stamps gone the rebuild explains an absence the measured run
    never had, so comparing the explanation would report a difference the elision
    already accounts for -- and report it under a name the skipped list never mentioned.
    """
    return field in figures or (
        field.endswith(_U_REASON_SUFFIX) and field[: -len(_U_REASON_SUFFIX)] in figures
    )


def _row_without_tainted(row: dict, figures: set[str]) -> dict:
    """Copy one rung row minus every tainted figure, its (U) reason, and its spread."""
    stripped = {field: value for field, value in row.items() if not _is_tainted(field, figures)}
    for block_name in _FIGURE_KEYED_SUBBLOCKS:
        block = stripped.get(block_name)
        if isinstance(block, dict):
            stripped[block_name] = {
                field: value for field, value in block.items() if not _is_tainted(field, figures)
            }
    return stripped


def _without_tainted(report: dict, figures: set[str]) -> dict:
    """Copy a report minus every per-rung figure and section an elision backs.

    Only the "run" branch is copied deeply enough to edit; every other value is shared
    with the caller's dict, which this module never mutates — mutating the caller's
    previous report would corrupt whatever error path the CLI runs next. The strip
    selects on keys without ever reading the tainted values, because those values are
    known to come from a substituted population and a number known to come from less
    evidence must not leave this function looking measured.
    """
    stripped = {key: value for key, value in report.items() if key not in _ELIDED_REPORT_SECTIONS}
    run_section = report.get("run")
    if isinstance(run_section, dict):
        run_copy = dict(run_section)
        results = run_section.get("results")
        # run.results is a LIST of rung rows, not a mapping keyed by concurrency -- a
        # dict comprehension here strips nothing, and the caller then reports "run" as
        # differing for the elision it just promised to skip.
        if isinstance(results, list):
            run_copy["results"] = [
                _row_without_tainted(row, figures) if isinstance(row, dict) else row
                for row in results
            ]
        stripped["run"] = run_copy
    return stripped


def _rung_sequence(report: dict) -> tuple[Any, ...]:
    """The concurrencies of a report's rung rows, in ladder order.

    Ordered, not a set: declaration order is execution order, so two ladders holding
    the same concurrencies in a different order are not the same ladder, and folding
    that difference away would let a reordered rebuild pass the check.
    """
    run_section = report.get("run")
    results = run_section.get("results") if isinstance(run_section, dict) else None
    if not isinstance(results, list):
        return ()
    return tuple(row.get("concurrency") if isinstance(row, dict) else row for row in results)


def check_report(bundle_dir, *, previous_report: dict) -> CheckResult:
    """Check whether the published report still follows from its bundle.

    With no grading-tainting elision this is the full rebuild and full comparison the
    check path has always done. With one, the internal rebuild still runs — the
    median-repetition picking and report assembly stay identical to the harness
    because _derive_report is the same function rebuild_report calls — and every field
    the elision backs is then removed from both sides before anything is compared. The
    tainted rebuilt values are never returned, written or printed: they are known to
    come from the substituted per-request-mean ITL population.
    """
    bundle_dir = Path(bundle_dir)
    _verify_bundle_or_refuse(bundle_dir)
    grading = _grading_elisions(bundle_dir)

    unmapped = _unmapped_elisions(grading)
    if unmapped:
        raise ReduceError(
            f"bundle {bundle_dir} declares elisions this version cannot scope: "
            + "; ".join(unmapped)
            + "; a partial check certifies every figure it did not skip, and without "
            "knowing which figures these fields back it would certify figures the "
            "elision may have moved"
        )

    rebuilt = _derive_report(bundle_dir, previous_report=previous_report)
    if not grading:
        return CheckResult(
            differing=tuple(_top_level_diffs(previous_report, rebuilt)),
            skipped={},
            elided=False,
        )

    # Insertion order into skipped is the CLI's listing order, and setdefault keeps
    # the first declared reason when two elisions back the same figure -- attributing
    # one figure to two elisions would invent a precision the manifest does not have.
    skipped: dict[str, str] = {}
    figures: set[str] = set()
    for key, reason in sorted(grading.items()):
        field_name = key.split(":", 1)[1]
        for figure in _ELIDED_FIELD_IMPACT[field_name]:
            figures.add(figure)
            skipped.setdefault(figure, reason)
        for section in _ELIDED_REPORT_SECTIONS:
            skipped.setdefault(section, reason)

    # An elision can blank values inside a rung row; it cannot mint or erase a
    # concurrency the harness measured at. A disagreement on the rung set is one the
    # elision cannot explain, so it belongs in differing even though the rows' tainted
    # fields are about to be stripped from the comparison.
    differing: list[str] = []
    skip: frozenset[str] = frozenset()
    if _rung_sequence(previous_report) != _rung_sequence(rebuilt):
        differing.append("run.results")
        # "run" itself would differ only because the results inside it did; naming
        # both paths would say one thing twice.
        skip = frozenset({"run"})

    published = _without_tainted(previous_report, figures)
    stripped = _without_tainted(rebuilt, figures)
    differing.extend(_top_level_diffs(published, stripped, skip=skip))
    return CheckResult(differing=tuple(differing), skipped=skipped, elided=True)
