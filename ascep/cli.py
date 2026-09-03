"""Command-line entry point for the ASCEP toolkit.

Every subcommand works on a bare ``pip install ascep``, with one exception. ``jsonschema``
lives in the optional ``run`` extra, so ``ascep.validation`` is imported lazily inside its
handler and its absence is reported as an install hint rather than a traceback; ``validate``
is the only command that genuinely needs it. ``conformance`` *uses* the schema check when it
is available and records a finding when it is not, and ``version``, ``size``, ``render`` and
``init`` never touch it at all. ``init`` does import ``ascep.validation``, but only for
``load_schema`` — reading the shipped JSON is plain stdlib; it is validating against it that
needs the extra.

That is deliberate, not incidental: the people most likely to reach for this tool are on a
locked-down benchmark cluster where adding a dependency is a ticket, and a capacity estimate
you cannot run is worth nothing. The ``bare-install`` CI job holds the line.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

from ascep import ASCEP_VERSION, __version__, capacity

# Mirrors validation.LAYERS; that module is only imported lazily inside handlers, so the
# layer choices cannot be read from it at parser-construction time.
_LAYERS = ("hardware", "model", "serving", "run", "workload", "capacity-report")

#: Derived from the dataclass rather than listed by hand. A hand-maintained allowlist silently
#: drops any field added to Workload later, and dropping one is not a crash — it is a wrong
#: number. Omitting requests_per_session, for instance, under-counts demand by exactly the
#: turns-per-session figure, which is the kind of quiet 5x error this protocol exists to stop.
_WORKLOAD_KEYS = tuple(f.name for f in dataclasses.fields(capacity.Workload))


def _eprint(message: str) -> None:
    """Write a diagnostic to stderr so stdout stays machine-parseable."""
    print(message, file=sys.stderr)


def _load_json(path: str) -> Any:
    """Parse a JSON file; decode and IO failures surface to the caller."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value: float) -> str:
    """Render a quantity without a meaningless trailing ``.0``."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _severity_rank(severity: str) -> int:
    """Sort key placing errors before warnings, robust to new levels being added later."""
    return 0 if severity == "error" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Schema-validate one JSON file against the chosen layer."""
    try:
        instance = _load_json(args.path)
    except (OSError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1
    try:
        from ascep import validation

        errors = validation.validate(args.layer, instance)
    except ImportError:
        # jsonschema and referencing are an optional extra; an install hint is more useful
        # than a traceback for a contributor on a fresh machine.
        _eprint("schema validation needs the optional dependencies: pip install 'ascep[run]'")
        return 2
    if errors:
        for message in errors:
            print(message)
        return 1
    print("OK")
    return 0


def _cmd_conformance(args: argparse.Namespace) -> int:
    """Check a report against rules C1-C12, print findings by rule, and with --raise save
    the computed level into the file so the artifact carries its own grade."""
    try:
        report = _load_json(args.path)
    except (OSError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1
    from ascep import conformance

    verdict = conformance.check(report)
    findings = sorted(
        verdict.findings,
        key=lambda f: (f.rule, _severity_rank(f.severity), f.path),
    )
    current_rule = None
    for finding in findings:
        if finding.rule != current_rule:
            print(f"{finding.rule}:")
            current_rule = finding.rule
        location = finding.path or "(report)"
        print(f"  [{finding.severity}] {location}: {finding.message}")
    if verdict.overstated:
        # A report claiming more than the checks support must never be quietly compared
        # against honest ones, so flag it as loudly as plain text allows.
        print(
            f"OVERSTATED: report claims {verdict.claimed!r} "
            f"but the checks support only {verdict.level!r}"
        )
    print(f"verdict: {verdict.level} (claimed: {verdict.claimed})")
    if args.raise_claim and not verdict.overstated:
        # The overstated case is deliberately excluded: a claim the checks do not support is
        # never lowered by a flag, quietly or otherwise. The OVERSTATED line above is the
        # whole response, and the author edits the file themselves.
        claimed = verdict.claimed
        if conformance.raise_claim(report, verdict):
            # Written exactly the way `ascep bench` writes a report, so a graded file diffs
            # against a fresh draft rather than against a third writer's idea of layout.
            pathlib.Path(args.path).write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            if claimed == verdict.level:
                # Grading a report that already claimed the right level still replaces the
                # draft paragraph, because "ungraded draft" is now false. Saying "raised"
                # here would misreport what moved.
                print(f"graded: {args.path} keeps {verdict.level!r}; draft note replaced")
            else:
                print(f"raised: {args.path} now claims {verdict.level!r} (was {claimed!r})")
        else:
            print(f"unchanged: {args.path} already claims {verdict.level!r}")
    if verdict.level == "non-conforming":
        return 1
    if args.strict and verdict.findings:
        return 1
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a report, writing to --out instead of stdout when it is given."""
    from ascep import render

    try:
        if args.out:
            render.render_file(args.path, out=args.out)
        else:
            print(render.render_file(args.path))
    except (OSError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1
    return 0


def _cmd_size(args: argparse.Namespace) -> int:
    """Compute the smallest whole-replica GPU count meeting the given workload."""
    try:
        data = _load_json(args.workload)
    except (OSError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1
    if not isinstance(data, dict):
        _eprint("error: workload file must contain a JSON object")
        return 1
    # A null value means "not given", not zero; drop it so Workload's defaults apply.
    kwargs = {key: data[key] for key in _WORKLOAD_KEYS if data.get(key) is not None}
    try:
        workload = capacity.Workload(**kwargs)
        cap = capacity.gpus_required(
            workload,
            kv_tokens_per_gpu=args.kv_tokens,
            throughput_tok_s_per_gpu=args.throughput_tok_s,
            headroom=args.headroom,
            gpus_per_replica=args.gpus_per_replica,
        )
        peak_users = workload.peak_concurrent_users()
        demand_tok_s = workload.demand_tok_s()
    except (TypeError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1
    print(f"peak_concurrent_users: {_fmt(peak_users)}")
    print(f"demand_tok_s: {_fmt(demand_tok_s)}")
    print(f"gpus: {cap.n_gpus}")
    print(f"binding_constraint: {cap.binding_constraint.value}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    """Write a schema-derived skeleton and name the choices it could not make."""
    from ascep import init as init_mod

    text = init_mod.render(args.layer)
    if args.out:
        path = pathlib.Path(args.out)
        # Refusing to clobber is not politeness. The file this would overwrite is a report
        # someone spent GPU-hours on, and `init` is the command a new user runs twice while
        # working out the flags.
        if path.exists() and not args.force:
            _eprint(f"error: {args.out} exists; pass --force to overwrite it")
            return 2
        path.write_text(text, encoding="utf-8")
        _eprint(f"wrote {args.out}")
    else:
        sys.stdout.write(text)

    # Straight to stderr so stdout stays a pipeable JSON document.
    for note in init_mod.decisions(args.layer):
        options = " OR ".join(" + ".join(fields) for fields in note["options"])
        _eprint(f"decide: {note['path']} needs {options} — no placeholder can be honest here")

    # A fresh skeleton deliberately fails validation, so say so here rather than letting the
    # first `ascep validate` read as a broken tool. The errors ARE the fill-in list.
    target = args.out or "the file"
    _eprint(f"every other value is null or {init_mod.TODO} — a skeleton is not yet valid.")
    _eprint(f"what is still unfilled:  ascep validate {target} --layer {args.layer}")
    if args.layer == "capacity-report":
        _eprint(f"how it grades:           ascep conformance {target}")
    return 0


def _bench_adapter(endpoint: dict):
    """Build the live adapter. Module-level so tests can replace it without a live server."""
    # httpx backs the OpenAI-compatible adapter and is an optional extra: ascep bench is the
    # only command that talks to a network, so the import failure has to read as an install
    # hint rather than a traceback. The import stays lazy so the rest of the CLI works
    # without the extra installed.
    import os

    try:
        from ascep.bench.adapters.base import AdapterConfig
        from ascep.bench.adapters.openai_compat import OpenAICompatAdapter
    except ImportError as exc:
        raise SystemExit(
            f"ascep bench needs the HTTP transport extra ({exc}); install it with: "
            "pip install ascep[run]"
        ) from exc
    return OpenAICompatAdapter(
        AdapterConfig(
            base_url=endpoint["base_url"],
            model=endpoint["model"],
            api_key=os.environ.get("ASCEP_API_KEY") or None,
            timeout_s=endpoint["timeout_s"],
        )
    )


def _cmd_bench(args: argparse.Namespace) -> int:
    from ascep.bench import run as bench_run

    # _bench_adapter is named through the module global on purpose: the acceptance tests
    # replace ascep.cli._bench_adapter with a fake, and resolving it at call time (rather
    # than closing over a local import) is what makes that substitution work.
    return bench_run.bench(args.config, dry_run=args.dry_run, adapter_factory=_bench_adapter)


#: Fields whose spread across sessions is worth a warning. A mean is only a workload if the
#: sessions behind it were alike; three sessions of 4, 40 and 400 turns average to a number no
#: session resembled, and a capacity estimate built on it is precise about a fiction.
_SPREAD_WATCH = (
    "turns_per_session",
    "requests_per_session",
    "input_tokens_per_request",
    "kv_residency",
)
#: Ratio of max to min above which the spread is called out. Two is deliberately loose: the
#: point is to catch sessions that are different *kinds* of work, not to police normal variance.
_SPREAD_FACTOR = 2.0

#: Workload fields the schema stores but the model can also derive. Overlaying a measurement
#: onto input_tokens_per_request or requests_per_session moves all four, and a declaration
#: whose stored derived values no longer follow from its stored inputs is the exact artifact
#: this protocol exists to stop -- it validates, it renders, and its GPU count is wrong.
_DERIVED_WORKLOAD_FIELDS = (
    "peak_concurrent_users",
    "active_sessions",
    "avg_context_tokens",
    "demand_tok_s",
)


def _merge_agent_workload(base: dict, fragment: dict) -> tuple[dict, list[str]]:
    """Overlay a measured agent block onto a workload declaration.

    Returns the merged declaration and the human-readable notes describing every change, so
    the caller can print them: a merge that edits a published declaration silently is worse
    than no merge at all.
    """
    merged = dict(base)
    notes: list[str] = []
    for key, value in fragment.items():
        if key == "_provenance":
            # Not a schema property. The workload layer sets additionalProperties false, so
            # carrying the provenance map into the declaration would make it fail validation.
            continue
        if key == "agent_loop" and isinstance(value, dict):
            existing = base.get("agent_loop") or {}
            value = dict(value)
            if value.get("session_max_context_tokens") is None:
                # The transcript cannot reveal the context limit, so a run of this command
                # without --session-max-context-tokens must not erase one the publisher
                # already declared: that would silently remove the compaction ceiling.
                inherited = existing.get("session_max_context_tokens")
                if inherited is not None:
                    value["session_max_context_tokens"] = inherited
                    notes.append("kept: agent_loop.session_max_context_tokens from the base file")
        if base.get(key) != value:
            notes.append(f"set: {key}: {json.dumps(base.get(key))} -> {json.dumps(value)}")
        merged[key] = value
        reason_key = f"{key}_u_reason"
        if reason_key in merged:
            # The field is measured now. Leaving the note that explains why it was not
            # measured makes the report claim a gap it has just closed, which costs the
            # publisher conformance level for work they actually did.
            merged.pop(reason_key)
            notes.append(f"dropped: {reason_key} — the field is measured now")
    return merged, notes


def _recompute_derived(merged: dict) -> tuple[dict, list[str]]:
    """Re-derive the stored derived fields from the merged inputs.

    Raises ValueError when a derived field is tagged as measured: replacing a measurement
    with an estimate is the one edit this command must never make quietly.
    """
    if merged.get("avg_context_tokens_tag") == "M":
        raise ValueError(
            "workload.avg_context_tokens is tagged (M): it was measured directly, and merging "
            "would overwrite that measurement with the estimator. Reconcile it by hand."
        )
    kwargs = {k: merged[k] for k in _WORKLOAD_KEYS if k in merged and merged[k] is not None}
    model = capacity.Workload(**kwargs)
    notes: list[str] = []
    out = dict(merged)
    for name in _DERIVED_WORKLOAD_FIELDS:
        if name not in out:
            continue
        value = getattr(model, name)()
        if out[name] != value:
            notes.append(f"recomputed: {name}: {json.dumps(out[name])} -> {_fmt(value)}")
        out[name] = value
    return out, notes


def _cmd_reduce(args: argparse.Namespace) -> int:
    """Re-derive a report from its bundle under the current reduction rules.

    The reduction is code and code gets fixed; without this command the only way to bring
    a published report up to a corrected rule is to re-measure it, which spends GPU hours
    to re-learn what the bundle already pins.
    """
    from ascep import validation
    from ascep.bench.rereduce import ReduceError, rebuild_report

    bundle_dir = pathlib.Path(args.bundle_dir)
    report_path = (
        pathlib.Path(args.report) if args.report else bundle_dir.parent / "report.json"
    )
    out_path = pathlib.Path(args.out) if args.out else report_path
    try:
        previous_report = _load_json(str(report_path))
    except (OSError, ValueError) as exc:
        _eprint(f"error: cannot read report {report_path}: {exc}")
        return 2
    if not isinstance(previous_report, dict):
        _eprint(f"error: {report_path} is not a JSON object")
        return 2
    try:
        rebuilt = rebuild_report(bundle_dir, previous_report=previous_report)
    except ReduceError as exc:
        _eprint(f"error: {exc}")
        return 2
    problems = validation.validate("capacity-report", rebuilt)
    if problems:
        # The bundle and the previous report were both accepted, so an invalid rebuild is
        # reduce assembling the report wrongly -- a defect here, not in the operator's
        # evidence, and nothing may be written on the strength of it.
        _eprint(
            "error: the rebuilt report fails capacity-report validation; this is a defect "
            "in reduce, not in the bundle -- please report it:"
        )
        for problem in problems:
            _eprint(f"  {problem}")
        return 2
    if args.check:
        differing = _report_top_level_diffs(previous_report, rebuilt)
        if not differing:
            _eprint(
                f"rebuilt report matches {report_path} (report_generated_utc and the "
                "conformance grade excluded; the grade is recomputed from the figures by "
                "`ascep conformance`, and the figures are what matched)"
            )
            return 0
        _eprint(f"rebuilt report differs from {report_path} at:")
        for path in differing:
            _eprint(f"  {path}")
        return 1
    try:
        out_path.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        _eprint(f"error: cannot write {out_path}: {exc}")
        return 2
    _print_result_changes(previous_report, rebuilt, out_path)
    # A rebuild is an ungraded draft by construction: the grade belongs to the figures,
    # and these figures are new. Carrying the old claim forward would let a report keep a
    # level earned by numbers it no longer contains, so reduce drops it and says so --
    # silently demoting a published `partial` to `non-conforming` would read as a
    # regression the operator caused rather than a grade waiting to be recomputed.
    claimed = previous_report.get("conformance")
    if claimed != rebuilt.get("conformance"):
        _eprint(
            f"note: the previous report claimed `{claimed}` and this rebuild is an "
            f"ungraded `{rebuilt.get('conformance')}`; run `ascep conformance --raise "
            f"{out_path}` to grade it from the figures it now carries"
        )
    return 0


#: Sentinel for "no value on this side", so an absent key is reported as absent rather than
#: conflated with an explicit null, which is a different statement in a report.
_ABSENT = object()

#: Cap on printed figure changes. A re-reduction that moves more figures than this is a rule
#: change the operator should diff in full, not a summary they should scroll.
_CHANGE_LINE_CAP = 40


def _ungraded(report: dict) -> dict:
    """A copy of ``report`` with the grade folded back to how `ascep bench` writes it.

    A rebuild is an ungraded draft by construction, so every graded report differs from its
    own rebuild at `conformance` and at the note's opening paragraph -- which would make
    --check report a difference on every published example in this repository and teach
    operators that a failing --check is normal. The grade is not a figure: it is computed
    from the figures by a different command, so if the figures match, re-grading reproduces
    it. Folding rather than skipping keeps the rest of the note in the comparison, because
    everything after the opening paragraph is written by the run and is a figure.
    """
    from ascep.conformance import DRAFT_NOTE, GRADED_NOTE

    folded = dict(report)
    folded.pop("conformance", None)
    note = folded.get("conformance_note")
    if isinstance(note, str) and note.startswith(GRADED_NOTE):
        folded["conformance_note"] = DRAFT_NOTE + note[len(GRADED_NOTE) :]
    return folded


def _report_top_level_diffs(old: dict, new: dict) -> list[str]:
    """Top-level keys whose values differ, excluding the generation timestamp and the grade.

    report_generated_utc is excluded because a rebuild is genuinely generated at a new
    time; comparing it would make --check fail by construction. `conformance` is excluded
    for the same reason and reported by _ungraded's fold instead. Nothing else is excluded,
    because anything else that differs is the point of the check.
    """
    old, new = _ungraded(old), _ungraded(new)
    diffs = []
    for key in list(old) + [k for k in new if k not in old]:
        if key == "report_generated_utc":
            continue
        if old.get(key) != new.get(key):
            diffs.append(key)
    return diffs


def _figure_diffs(old: Any, new: Any, path: str, out: list) -> None:
    """Collect leaf-level differences between two report fragments as (path, old, new).

    A side that is absent or null is treated as the empty container when the other side is
    a container, so a rung that appears or disappears reads as its figures, not as one
    opaque blob of JSON.
    """
    if isinstance(new, dict) and (old is _ABSENT or old is None):
        old = {}
    elif isinstance(new, list) and (old is _ABSENT or old is None):
        old = []
    elif isinstance(old, dict) and (new is _ABSENT or new is None):
        new = {}
    elif isinstance(old, list) and (new is _ABSENT or new is None):
        new = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in list(old) + [k for k in new if k not in old]:
            _figure_diffs(old.get(key, _ABSENT), new.get(key, _ABSENT), f"{path}.{key}", out)
    elif isinstance(old, list) and isinstance(new, list):
        for index in range(max(len(old), len(new))):
            old_item = old[index] if index < len(old) else _ABSENT
            new_item = new[index] if index < len(new) else _ABSENT
            _figure_diffs(old_item, new_item, f"{path}[{index}]", out)
    elif old is _ABSENT or new is _ABSENT or old != new:
        out.append((path, old, new))


def _format_figure(value: Any) -> str:
    """Render one figure for the change summary; absent reads as absent, not as null."""
    if value is _ABSENT:
        return "<absent>"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return json.dumps(value)


def _print_result_changes(previous_report: dict, rebuilt: dict, out_path: pathlib.Path) -> None:
    """Say which figures under run.results moved, and by how much.

    The operator should not have to diff two 30 KB files to learn what the new reduction
    rule did to their rungs.
    """
    old_results = (previous_report.get("run") or {}).get("results")
    new_results = (rebuilt.get("run") or {}).get("results")
    changes: list = []
    _figure_diffs(old_results, new_results, "run.results", changes)
    if not changes:
        _eprint(f"wrote {out_path}; no figures under run.results changed")
        return
    shown = changes[:_CHANGE_LINE_CAP]
    _eprint(f"wrote {out_path}; {len(changes):,} figure(s) under run.results changed:")
    for path, old, new in shown:
        line = f"  {path}: {_format_figure(old)} -> {_format_figure(new)}"
        if (
            isinstance(old, (int, float))
            and not isinstance(old, bool)
            and isinstance(new, (int, float))
            and not isinstance(new, bool)
        ):
            line += f" ({new - old:+,.6g})"
        _eprint(line)
    suppressed = len(changes) - len(shown)
    if suppressed:
        _eprint(f"  ... {suppressed:,} more suppressed")


def _cmd_agent_profile(args: argparse.Namespace) -> int:
    """Turn agent session transcripts into a measured ``code_agent`` workload block.

    The agent-loop numbers can otherwise only be declared -- provenance (U) or (I) -- and a
    declared turns_per_session is a guess that multiplies straight through to demand.
    """
    from ascep import agent_profile

    profiles = []
    for path in args.exports:
        try:
            profiles.append(agent_profile.parse_session(_load_json(path)))
        except (OSError, ValueError) as exc:
            # Name the file. A campaign profiles a directory of exports at once, and
            # "expected a session export" without a path sends the operator hunting.
            _eprint(f"error: {path}: {exc}")
            return 1

    if args.shapes:
        path = pathlib.Path(args.shapes)
        if path.exists() and not args.force:
            _eprint(f"error: {args.shapes} exists; pass --force to overwrite it")
            return 2
        try:
            shapes = agent_profile.to_replay_shapes(
                profiles, shared_prefix_tokens=args.shared_prefix_tokens
            )
        except ValueError as exc:
            _eprint(f"error: {exc}")
            return 1
        path.write_text(json.dumps(shapes, indent=2) + "\n", encoding="utf-8")
        summary = shapes["_summary"]
        _eprint(
            f"wrote {args.shapes}: {summary['sessions']} session(s), {summary['steps']} step(s), "
            f"{summary['prefix_resets']} prefix reset(s)"
        )
        if args.shared_prefix_tokens == 0:
            # Zero is the conservative default, not a measurement. Every real agent sends the
            # same system prompt and tool schemas on every request, and a replay that shares
            # nothing makes each session's first turn a cold prefill the deployment would not
            # pay -- understating capacity rather than overstating it, but still wrong.
            _eprint(
                "shared_prefix_tokens is 0: the replay will share no system prompt or tool "
                "schema across sessions, which under-counts prefix-cache hits"
            )

    stats = agent_profile.aggregate(profiles)
    try:
        workload = agent_profile.to_ascep_workload(
            profiles, session_max_context_tokens=args.session_max_context_tokens
        )
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 1

    document = workload
    if args.into:
        try:
            base = _load_json(args.into)
        except (OSError, ValueError) as exc:
            _eprint(f"error: {args.into}: {exc}")
            return 1
        if not isinstance(base, dict):
            _eprint(f"error: {args.into}: a workload declaration must be a JSON object")
            return 1
        document, notes = _merge_agent_workload(base, workload)
        try:
            document, derived_notes = _recompute_derived(document)
        except (TypeError, ValueError) as exc:
            _eprint(f"error: {exc}")
            return 1
        for note in notes + derived_notes:
            _eprint(note)

    text = json.dumps(document, indent=2) + "\n"
    if args.out:
        path = pathlib.Path(args.out)
        if path.exists() and not args.force:
            _eprint(f"error: {args.out} exists; pass --force to overwrite it")
            return 2
        path.write_text(text, encoding="utf-8")
        _eprint(f"wrote {args.out}")
    else:
        sys.stdout.write(text)

    # Everything below is advisory and goes to stderr, so stdout stays one JSON document.
    # Without --into that document is a fragment and will NOT validate on its own: the
    # workload layer sets additionalProperties false and requires two dozen fields a
    # transcript cannot supply.
    counted = len(profiles) - stats["skipped_sessions"]
    _eprint(f"profiled {counted} session(s); skipped {stats['skipped_sessions']} with no turns")
    for name in _SPREAD_WATCH:
        entry = stats.get(name)
        if not entry or entry["n"] < 2 or entry["min"] <= 0:
            continue
        if entry["max"] / entry["min"] >= _SPREAD_FACTOR:
            _eprint(
                f"spread: {name} ranges {_fmt(entry['min'])}..{_fmt(entry['max'])} across "
                f"{entry['n']} sessions — the mean this reports describes no single session"
            )
    if stats.get("compaction_resume_tokens") is None:
        # Null is honest, but a reader can mistake it for zero, which would price the KV
        # floor as though every session resumed from an empty prompt.
        _eprint(
            "compaction_resume_tokens is null: no session in this sample compacted, so the "
            "figure is not measured rather than zero"
        )
    if args.session_max_context_tokens is None:
        _eprint(
            "session_max_context_tokens is null: it is a serving choice, not a property of "
            "the transcript — pass --session-max-context-tokens to record the limit you ran at"
        )
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    """Print the package and protocol version."""
    print(f"ascep {__version__} (protocol {ASCEP_VERSION})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands wired to their handlers."""
    parser = argparse.ArgumentParser(
        prog="ascep",
        description="AI Serving Capacity Estimation Protocol tools.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_init = sub.add_parser("init", help="write a fillable report skeleton")
    p_init.add_argument("-o", "--out", help="write the skeleton here instead of stdout")
    p_init.add_argument(
        "--layer",
        default="capacity-report",
        choices=_LAYERS,
        help="which schema to scaffold (default: capacity-report, the whole report)",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out if it already exists",
    )
    p_init.set_defaults(handler=_cmd_init)

    p_validate = sub.add_parser("validate", help="schema-validate a JSON file")
    p_validate.add_argument("path", help="path to the JSON file")
    p_validate.add_argument(
        "--layer",
        default="capacity-report",
        choices=_LAYERS,
        help="schema layer to validate against (default: capacity-report)",
    )
    p_validate.set_defaults(handler=_cmd_validate)

    p_conf = sub.add_parser("conformance", help="check a report against rules C1-C12")
    p_conf.add_argument("path", help="path to the report JSON file")
    p_conf.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if there is any finding at all, even warnings",
    )
    p_conf.add_argument(
        "--raise",
        dest="raise_claim",
        action="store_true",
        help="write the computed level into the report when it is stronger than the claim",
    )
    p_conf.set_defaults(handler=_cmd_conformance)

    p_render = sub.add_parser("render", help="render a report as text")
    p_render.add_argument("path", help="path to the report JSON file")
    p_render.add_argument("-o", "--out", help="write output here instead of stdout")
    p_render.set_defaults(handler=_cmd_render)

    p_size = sub.add_parser("size", help="GPU count required to serve a workload")
    p_size.add_argument("--workload", required=True, help="path to a workload JSON file")
    p_size.add_argument(
        "--kv-tokens",
        type=float,
        required=True,
        metavar="N",
        help="per-GPU KV pool size in tokens; MUST come from a measurement at the SAME "
        "tensor-parallel width you will deploy at (see --gpus-per-replica), because "
        "per-GPU KV is not constant across topologies",
    )
    p_size.add_argument(
        "--throughput-tok-s",
        type=float,
        required=True,
        metavar="N",
        help="per-GPU decode throughput in tokens/s; MUST come from a measurement at the "
        "SAME tensor-parallel width you will deploy at (see --gpus-per-replica)",
    )
    p_size.add_argument(
        "--gpus-per-replica",
        type=int,
        default=1,
        metavar="N",
        help="tensor-parallel width; capacity is bought in whole replicas (default: 1)",
    )
    p_size.add_argument(
        "--headroom",
        type=float,
        default=1.15,
        metavar="F",
        help="factor dividing usable capacity for the recommended tier (default: 1.15)",
    )
    p_size.set_defaults(handler=_cmd_size)

    p_bench = sub.add_parser("bench", help="run a concurrency ladder and write a draft report")
    p_bench.add_argument("config", help="path to the bench config JSON")
    p_bench.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p_bench.set_defaults(handler=_cmd_bench)

    p_reduce = sub.add_parser(
        "reduce", help="re-derive a report from its bundle under the current reduction"
    )
    p_reduce.add_argument("bundle_dir", help="path to the bundle to reduce")
    p_reduce.add_argument(
        "--report",
        help="path to the existing report (default: BUNDLE_DIR/../report.json)",
    )
    p_reduce.add_argument(
        "--out",
        help="write the rebuilt report here (default: overwrite --report)",
    )
    p_reduce.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 0 if the rebuilt report matches the existing one, "
        "1 if it differs",
    )
    p_reduce.set_defaults(handler=_cmd_reduce)

    p_agent = sub.add_parser(
        "agent-profile",
        help="measure a code_agent workload from agent session transcripts",
    )
    p_agent.add_argument(
        "exports",
        nargs="+",
        metavar="EXPORT",
        help="session export JSON, as written by `opencode export <sessionID>`; pass several "
        "to average across sessions",
    )
    p_agent.add_argument("-o", "--out", help="write the workload block here instead of stdout")
    p_agent.add_argument(
        "--into",
        metavar="WORKLOAD",
        help="merge the measured block into this existing workload declaration and emit the "
        "whole document, re-deriving avg_context_tokens and demand_tok_s from the new inputs; "
        "without it the output is a fragment for you to merge by hand",
    )
    p_agent.add_argument(
        "--force", action="store_true", help="overwrite --out or --shapes if they exist"
    )
    p_agent.add_argument(
        "--shapes",
        metavar="FILE",
        help="also write the per-session, per-step shape file the closed-loop replay driver "
        "consumes; the workload block is a set of means and a mean cannot be replayed",
    )
    p_agent.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        metavar="N",
        help="tokens of system prompt and tool schema every session sends identically, for "
        "--shapes. The transcript records only the prompt total and cannot decompose it, so "
        "this is declared; 0 means the replay assumes no shared prefix (default: 0)",
    )
    p_agent.add_argument(
        "--session-max-context-tokens",
        type=int,
        default=None,
        metavar="N",
        help="the context limit the agent ran against; it governs when the loop compacts and "
        "is a serving choice the transcript cannot reveal, so it must be supplied by hand",
    )
    p_agent.set_defaults(handler=_cmd_agent_profile)

    p_version = sub.add_parser("version", help="print the protocol and package version")
    p_version.set_defaults(handler=_cmd_version)

    return parser


def main(argv=None) -> int:
    """Parse arguments, dispatch to a subcommand, and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
