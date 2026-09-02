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
    """Check a report against rules C1-C11, print findings by rule, and with --raise save
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

    p_conf = sub.add_parser("conformance", help="check a report against rules C1-C11")
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
