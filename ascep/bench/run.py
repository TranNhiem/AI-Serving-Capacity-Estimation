"""``ascep bench``: run a declared concurrency ladder and emit a draft capacity report.

Every other command in this toolkit grades evidence someone else produced; this one produces
the evidence. Its contract is therefore inverted relative to the rest of the CLI: refuse to
run rather than run under-specified, never invent a value the operator did not declare, and
never grade its own output. A wrong number emitted here is not caught downstream -- it *is*
the downstream.

What this module keeps is the decision-making: reading and refusing the config, building the
workload, climbing the ladder and deciding what to keep. Turning the kept windows into the
report JSON is :mod:`ascep.bench.report`, because a second caller re-derives a published
report from its bundle without running anything, and the two must not be separate
implementations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import signal
import sys
from pathlib import Path

from ascep import validation
from ascep.bench import driver, ladder, metrics, persist, report, sessions, workloads

__all__ = [
    "ConfigError",
    "bench",
    "ladder_policy",
    "load_config",
    "load_declarations",
    "plan_lines",
]


class ConfigError(Exception):
    """The bench config is missing, malformed, or under-specified; the run must not start."""


class _Terminated(KeyboardInterrupt):
    """A SIGTERM, raised as an interrupt so it takes the path Ctrl-C already takes."""


def _sigterm_raises_interrupt(_signum, _frame):
    raise _Terminated()


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """Make SIGTERM behave like Ctrl-C for the duration of a run.

    These runs are submitted as batch jobs, and the two ways they end early -- ``scancel``
    and the job's wall clock -- both arrive as SIGTERM, whose default action ends the process
    where it stands. Every completed window is in RAM at that moment and the bundle is
    written only at the end, so the default action turns hours of real measurement into
    nothing at all. The interrupt path already bundles what completed and marks the ladder
    censored, so the whole fix is to arrive on it. The SIGKILL that follows a grace period
    cannot be caught by anything; the answer to that one is a shorter ladder.
    """
    try:
        previous = signal.signal(signal.SIGTERM, _sigterm_raises_interrupt)
    except (AttributeError, OSError, ValueError):
        # No SIGTERM on this platform, or not the main thread: an embedded caller gets the
        # run, not an exception about signal handling.
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _cause_of(exc):
    """Name an interrupt in the words the operator will recognise from their scheduler."""
    if isinstance(exc, _Terminated):
        return "SIGTERM -- a scheduler kill: scancel, or the job's wall clock"
    return type(exc).__name__


#: Every permitted section and key, with the citation a refusal must carry when the key is
#: absent. Unknown keys and sections are refusals too: a typo such as ``drain_deadline_sec``
#: is otherwise indistinguishable from omitting the real key, and the run would proceed on a
#: default the operator believes they overrode.
_KEY_CITATIONS = {
    "endpoint": {
        "base_url": "section 7",
        "model": "section 7",
        "timeout_s": "section 7",
    },
    "declarations": {
        "hardware": "C3",
        "model": "C3",
        "serving": "C3",
        "workload": "C3",
    },
    "workload": {
        "corpus": "section 7",
        "input_tokens": "section 7",
        "output_tokens": "section 7",
        "ignore_eos": "section 7",
        "cache_policy": "section 7",
        "seed": "section 7",
        "think_time_s": "section 7",
        "run_label": "section 7",
    },
    "window": {
        "window_s": "section 7",
        "drain_deadline_s": "section 7.6",
        "warmup_requests": "section 7.3",
    },
    "ladder": {
        "concurrency": "section 7",
        "repetitions": "section 7",
        "throughput_collapse_ratio": "section 7",
    },
    "slo_gates": {
        "ttft_p95_max_s": "section 7",
        "itl_p95_max_s": "section 7",
        "e2e_p95_max_s": "section 7",
        "error_rate_max_pct": "section 7",
        "declared_before_run": "C7",
    },
    "output": {
        "bundle_dir": "section 7",
        "report_path": "section 7",
        "engine_logs_path": "section 7",
        "container_digest": "section 7",
    },
}

#: Keys a config may add but is not required to, in the same {section: {key: citation}}
#: shape as _KEY_CITATIONS. They live in a parallel mapping because a protocol that grows
#: capabilities cannot make every new key a breaking change to every operator's config:
#: adding the media keys to the required mapping would invalidate every published bench
#: config overnight, including configs whose runs are already cited. But silently accepting
#: unknown keys is how a typo becomes a run nobody declared, so _check_shape permits
#: exactly required | optional and still refuses anything outside the union by name.
_OPTIONAL_KEY_CITATIONS = {
    "workload": {
        # Directory the corpus's relative media paths resolve against; its presence is
        # what selects the multimodal corpus.
        "media_root": "section 9",
        # "base64" or "url", mirroring the serving layer's image_input_transport.
        "image_input_transport": "section 9",
        # Base URL the server fetches media from; required when transport is "url".
        "media_url_prefix": "section 9",
        # Cap on records, for a smoke run.
        "media_max_records": "section 9",
        # Where the prompt text lives in each record; default "conversations". Read by both
        # corpus readers, so it selects the same turn whether or not 'media_root' is set.
        "prompt_field": "section 9",
        # Send a media-bearing corpus as its text-only variant, markers removed. Opting in
        # from the config is the only way the reader's own error message can be acted on;
        # without it that message names a Python argument an operator cannot reach.
        "strip_media_placeholders": "section 9",
        # True when 'corpus' is a shapes file written by `ascep agent-profile --shapes`
        # rather than a prompt corpus: the ladder then replays whole captured sessions,
        # each step carrying the prompt growth, the generated length and the gap the
        # capture recorded. It is an explicit switch rather than something inferred from
        # the file's contents because the two modes measure different things, and a
        # config that fell into the wrong one would publish agent numbers for
        # independent-request traffic, or the reverse, with nothing in the report saying so.
        "replay_sessions": "section 10",
    },
}

#: Expected JSON types per key. bool is rejected anywhere a number is wanted, because
#: ``True`` silently satisfying ``input_tokens: int`` is exactly the kind of lie about what
#: was declared that this command exists to refuse.
_TYPES = {
    ("endpoint", "base_url"): (str,),
    ("endpoint", "model"): (str,),
    ("endpoint", "timeout_s"): (int, float),
    ("declarations", "hardware"): (str,),
    ("declarations", "model"): (str,),
    ("declarations", "serving"): (str,),
    ("declarations", "workload"): (str,),
    ("workload", "corpus"): (str,),
    # On a real corpus this key is consumed nowhere, so any int here is a prompt-length
    # claim the published config carries and nothing checks. Null is admitted so the
    # corpus-mode rule in _check_values can refuse every other value.
    ("workload", "input_tokens"): (int, type(None)),
    # Null is the declared uncapped mode. It is admitted here so the ignore_eos rule in
    # _check_values can tell it apart from the one combination that must be refused --
    # true with no length -- rather than dying of a bare None <= 1 TypeError downstream.
    ("workload", "output_tokens"): (int, type(None)),
    ("workload", "ignore_eos"): (bool,),
    ("workload", "cache_policy"): (str,),
    ("workload", "seed"): (int,),
    ("workload", "think_time_s"): (int, float),
    ("workload", "run_label"): (str,),
    ("workload", "media_root"): (str, type(None)),
    ("workload", "image_input_transport"): (str,),
    ("workload", "media_url_prefix"): (str, type(None)),
    ("workload", "media_max_records"): (int, type(None)),
    ("workload", "prompt_field"): (str,),
    ("workload", "strip_media_placeholders"): (bool,),
    ("workload", "replay_sessions"): (bool,),
    ("window", "window_s"): (int, float),
    ("window", "drain_deadline_s"): (int, float),
    ("window", "warmup_requests"): (int,),
    ("ladder", "concurrency"): (list,),
    ("ladder", "repetitions"): (int,),
    ("ladder", "throughput_collapse_ratio"): (int, float),
    ("slo_gates", "ttft_p95_max_s"): (int, float, type(None)),
    ("slo_gates", "itl_p95_max_s"): (int, float, type(None)),
    ("slo_gates", "e2e_p95_max_s"): (int, float, type(None)),
    ("slo_gates", "error_rate_max_pct"): (int, float, type(None)),
    ("output", "bundle_dir"): (str,),
    ("output", "report_path"): (str,),
    ("output", "engine_logs_path"): (str, type(None)),
    ("output", "container_digest"): (str, type(None)),
}


def load_config(path):
    """Read and fully validate the bench config, returning ``(config, raw_bytes)``.

    The raw bytes come back alongside the parsed document because the bundle must carry the
    operator's file verbatim: a re-serialisation records what this process understood, and
    the one failure worth catching is the harness understanding something other than what the
    operator wrote. Raises ConfigError for anything unreadable, malformed, unknown, or
    under-specified -- chapter 7's declarations have no honest defaults.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read the bench config '{path}': {exc}") from exc
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ConfigError(f"the bench config '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"the bench config '{path}' must be a JSON object at the top level")
    _check_shape(config)
    _check_values(config)
    return config, raw


def _check_shape(config):
    """Reject unknown sections and keys, and any missing key, naming what requires it."""
    for section in config:
        if section not in _KEY_CITATIONS:
            raise ConfigError(
                f"unknown section '{section}' in the bench config; permitted sections "
                f"are: {', '.join(sorted(_KEY_CITATIONS))}"
            )
    for section, keys in _KEY_CITATIONS.items():
        body = config.get(section)
        expected = ", ".join(keys)
        cite = keys[next(iter(keys))]
        if body is None:
            raise ConfigError(
                f"missing section '{section}' in the bench config; it must declare the "
                f"keys: {expected} ({cite})"
            )
        if not isinstance(body, dict):
            raise ConfigError(
                f"section '{section}' of the bench config must be a JSON object with the "
                f"keys: {expected} ({cite})"
            )
        # The permitted set is required | optional; the required loop below is exactly
        # the one that has always run, so a missing required key still raises the
        # message every published run was validated against.
        optional = _OPTIONAL_KEY_CITATIONS.get(section, {})
        permitted = ", ".join(list(keys) + [key for key in optional if key not in keys])
        for key in body:
            if key not in keys and key not in optional:
                raise ConfigError(
                    f"unknown key '{key}' in section '{section}' of the bench config; "
                    f"expected keys: {permitted} (section 7)"
                )
        for key, key_cite in keys.items():
            if key not in body:
                raise ConfigError(
                    f"bench config is missing '{key}' (section '{section}'): {key_cite} "
                    "requires it to be fixed before the first request is sent, and running "
                    "on a default would measure a run the operator never declared"
                )


def _check_values(config):
    """Reject values that parse but would silently change what is measured."""
    for (section, key), types in _TYPES.items():
        if key not in config[section]:
            # Optional and absent: there is no value to type-check, and the default is
            # applied in _build_workload, where it can be named rather than smuggled in.
            continue
        value = config[section][key]
        allows_bool = any(t is bool for t in types)
        if not isinstance(value, types) or (isinstance(value, bool) and not allows_bool):
            names = [t.__name__ for t in types if t is not type(None)]
            label = "/".join(names) + (" or null" if type(None) in types else "")
            cite = _KEY_CITATIONS[section].get(key) or _OPTIONAL_KEY_CITATIONS[section][key]
            raise ConfigError(
                f"config key '{key}' (section '{section}') must be {label}; got {value!r} ({cite})"
            )
    gates = config["slo_gates"]
    if gates["declared_before_run"] is not True:
        raise ConfigError(
            "'declared_before_run' must be literally true: C7 fixes the SLO gates before "
            "the first request is sent, and gates chosen after the numbers are in produce "
            "a sustainable tier that cannot be published"
        )
    base_url = config["endpoint"]["base_url"].rstrip("/")
    if base_url.endswith("/v1"):
        # The adapter appends the OpenAI route itself, so a base_url carrying one requests
        # /v1/v1/chat/completions. Every request 404s, which is scored as an error rather
        # than as a broken config, and the ladder climbs to the top measuring nothing but
        # the typo before publishing a 100% error rate as a property of the server.
        raise ConfigError(
            f"'base_url' must be the server root, not the API route; got {base_url!r}. "
            "The adapter appends /v1/chat/completions, so this one would request "
            "/v1/v1/chat/completions and score every 404 as a server error (section 7)"
        )
    cache_policy = config["workload"]["cache_policy"]
    if cache_policy not in workloads.CACHE_POLICIES:
        allowed = ", ".join(sorted(workloads.CACHE_POLICIES))
        raise ConfigError(
            f"'cache_policy' must be one of {allowed}; got {cache_policy!r} (section 7)"
        )
    rungs = config["ladder"]["concurrency"]
    bad_rung = any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in rungs)
    if not rungs or bad_rung:
        raise ConfigError(
            "'concurrency' must be a non-empty list of positive integers: the rungs are "
            "the measurement itself, and section 7 requires them declared in advance"
        )
    if any(later <= earlier for earlier, later in zip(rungs, rungs[1:])):
        # Grading keys a rung by its concurrency, so [4, 4] is not two searches of one
        # rung each -- it is six repetitions pooled into one, published as a single row
        # while the report's ladder declaration still lists two. Out of order is worse:
        # the collapse test holds each rung against the best *lower* COMPLETE rung, and a
        # descending ladder has none, so collapse silently stops being tested at all.
        raise ConfigError(
            "'concurrency' must be strictly increasing: section 7 climbs the ladder, and "
            f"got {rungs}. A repeated rung pools independent repetitions into one "
            "operating point; a descending one disables the collapse test"
        )
    if len(rungs) < 3:
        # run.schema.json puts minItems 3 on run.results, and there is one row per rung, so
        # a two-rung ladder cannot produce a report that validates. Caught here rather than
        # at the end because the alternative is what a two-rung smoke run actually did:
        # spend the windows, print the rung summaries, and then fail draft validation with
        # "is too short" and "this is a defect in bench" -- which sends the operator looking
        # for a bug in the writer rather than at the ladder they declared. run.single_point
        # is not an escape hatch for this: it labels a campaign at one *context length*,
        # which is a different axis and is set automatically.
        raise ConfigError(
            f"'concurrency' must declare at least 3 rungs; got {rungs}. Two points cannot "
            "show where throughput stops scaling, so the report they produce would not "
            "validate and the GPU hours would be spent before anything said so (section 7)"
        )
    # On anything but the synthetic corpus this number is consumed nowhere: the corpus's
    # own records fix the prompt length, so a declared value is validated and then used
    # for nothing, and the published config carries a prompt-length claim that can
    # disagree with the run by any margin without a single check firing.
    corpus = config["workload"]["corpus"]
    input_tokens = config["workload"]["input_tokens"]
    if corpus == "synthetic":
        # The synthetic corpus is generated to exactly this length and has no other
        # source for it; a null here is not deference to the corpus but a workload with
        # an unknowable prompt, and a default would measure a workload nobody declared.
        if input_tokens is None:
            raise ConfigError(
                "'input_tokens' must be a positive integer when 'corpus' is 'synthetic': "
                "the synthetic corpus has no other source for its length (section 7)"
            )
        if input_tokens <= 1:
            raise ConfigError("config key 'input_tokens' must be greater than 1 (section 7)")
    elif input_tokens is not None:
        raise ConfigError(
            f"'input_tokens' must be null when 'corpus' names a corpus file; got "
            f"{input_tokens!r} against corpus '{corpus}' (section 7). The corpus's own "
            "records fix the prompt length, so this number is read, checked and used for "
            "nothing. The measured prompt-token figure belongs in the workload "
            "declaration, which the report grades -- nothing grades a number in the bench "
            "config against it"
        )
    # Like input_tokens above, this pair is mode-conditional: false with a null length is a
    # legal declaration, the explicitly uncapped mode, but true turns off the model's only
    # way to stop on its own, so a length must travel with it. A null there is not
    # deference to the model; it asks the server to generate until the context limit on
    # every single request.
    ignore_eos = config["workload"]["ignore_eos"]
    output_tokens = config["workload"]["output_tokens"]
    if ignore_eos:
        if output_tokens is None:
            raise ConfigError(
                "'output_tokens' must be a positive integer when 'ignore_eos' is true: "
                "ignore_eos with no output length asks the server to generate until the "
                "context limit on every single request, which is a decode storm wearing a "
                "benchmark's name (section 7)"
            )
        if output_tokens <= 1:
            raise ConfigError("config key 'output_tokens' must be greater than 1 (section 7)")
    elif output_tokens is not None and output_tokens <= 1:
        raise ConfigError("config key 'output_tokens' must be greater than 1 (section 7)")
    numeric_floors = [
        ("ladder", "repetitions", 1, "section 7"),
        ("window", "window_s", 0, "section 7"),
        ("window", "drain_deadline_s", 0, "section 7.6"),
        ("window", "warmup_requests", -1, "section 7.3"),
        # A non-positive timeout aborts every request before the first token and reports
        # a 100% error rate as a property of the server.
        ("endpoint", "timeout_s", 0, "section 7"),
    ]
    for section, key, minimum, cite in numeric_floors:
        if config[section][key] <= minimum:
            raise ConfigError(f"config key '{key}' must be greater than {minimum} ({cite})")
    if config["workload"]["think_time_s"] < 0:
        # A negative think time asks a closed loop to arrive faster than its completions
        # allow; a quiet clamp to zero would turn the loop into an unthrottled one and
        # change the workload being measured.
        raise ConfigError("'think_time_s' must not be negative (section 7)")
    if config["output"]["engine_logs_path"] is None:
        # The bundle hashes the engine's own log so a reader can check the server's account
        # of the run against the client's. A null here is indistinguishable from a harness
        # that never asked for one, and it is the only C8 artifact bench cannot produce
        # itself.
        raise ConfigError(
            "'engine_logs_path' must name a readable file (C8): the reproduction bundle "
            "hashes the engine's own log, which is the only record of the run written by "
            "the server rather than by the load generator. If the engine wrote none, say "
            "so in a file and point at that"
        )
    _check_gates(config["slo_gates"])
    try:
        ladder_policy(config)
    except ValueError as exc:
        # LadderPolicy owns these rules, and it is constructed after the ladder has run.
        # Constructing a throwaway here moves the same refusal to before the first request:
        # a repetitions count of two is a defect in the declaration, and discovering it
        # when the grading call raises has already spent the GPU hours.
        raise ConfigError(f"the declared ladder cannot be graded as specified: {exc}") from exc


def _check_gates(gates):
    """Refuse an SLO declaration that satisfies C7 in shape while binding nothing."""
    named = ("ttft_p95_max_s", "itl_p95_max_s", "e2e_p95_max_s", "error_rate_max_pct")
    if all(gates[key] is None for key in named):
        # Four nulls and ``declared_before_run: true`` is the most dangerous config this
        # command accepts: every window passes, every rung grades COMPLETE, and the
        # sustainable tier it publishes is the measured tier under another name.
        raise ConfigError(
            "at least one SLO gate must be declared (C7): with all four null every window "
            "passes by definition, and the sustainable tier becomes the measured tier "
            "wearing an SLO label"
        )
    for key in ("ttft_p95_max_s", "itl_p95_max_s", "e2e_p95_max_s"):
        if gates[key] is not None and gates[key] <= 0:
            raise ConfigError(
                f"SLO gate '{key}' must be positive; got {gates[key]!r}. A latency bound "
                "at or below zero cannot be met by any response, so the run measures the "
                "gate rather than the server (section 7)"
            )
    error_rate = gates["error_rate_max_pct"]
    if error_rate is not None and not 0 <= error_rate <= 100:
        raise ConfigError(
            f"SLO gate 'error_rate_max_pct' must be a percentage between 0 and 100; got "
            f"{error_rate!r} (section 7)"
        )


def ladder_policy(config):
    """Build the grading policy from the config, so config validation refuses what it will."""
    gate_cfg = config["slo_gates"]
    gates = metrics.SloGates(
        ttft_p95_max_s=gate_cfg["ttft_p95_max_s"],
        itl_p95_max_s=gate_cfg["itl_p95_max_s"],
        e2e_p95_max_s=gate_cfg["e2e_p95_max_s"],
        error_rate_max_pct=gate_cfg["error_rate_max_pct"],
    )
    return gates, ladder.LadderPolicy(
        gates,
        repetitions=config["ladder"]["repetitions"],
        throughput_collapse_ratio=config["ladder"]["throughput_collapse_ratio"],
        cache_policy=config["workload"]["cache_policy"],
    )


def load_declarations(config, config_dir):
    """Read and schema-validate the four declared layer documents the run is bound to.

    Paths resolve relative to the directory holding the config file, so a bundle moved
    between machines still finds its declarations next to the config that named them. This
    check runs during ``--dry-run`` too: discovering four hours into a Slurm allocation that
    ``serving.json`` was malformed is discovering it too late, and C3 has no way to bind a
    measurement to a topology nobody wrote down.
    """
    documents = {}
    for layer in ("hardware", "model", "serving", "workload"):
        written = config["declarations"][layer]
        try:
            doc = json.loads((Path(config_dir) / written).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(
                f"cannot read the {layer} declaration '{written}' (C3): {exc}. The "
                "measurement cannot be bound to a declaration that does not parse"
            ) from exc
        problems = validation.validate(layer, doc)
        if problems:
            raise ConfigError(
                f"the {layer} declaration '{written}' failed schema validation (C3): "
                + "; ".join(problems)
            )
        documents[layer] = doc
    return documents


def _declaration_bytes(config, config_dir):
    """The four declaration files as bytes, to be hashed into the bundle beside the config.

    The report carries parsed copies of these documents, and a parsed copy is not evidence:
    a reader holding the report alone cannot tell whether the hardware block it claims to
    have been measured on is the one that was declared before the run. Their bytes go into
    the bundle so the manifest covers them, which is the only place C3's binding becomes
    checkable after publication.
    """
    files = {}
    for layer in ("hardware", "model", "serving", "workload"):
        written = config["declarations"][layer]
        try:
            files[f"declarations/{layer}.json"] = (Path(config_dir) / written).read_bytes()
        except OSError as exc:
            raise ConfigError(
                f"cannot read the {layer} declaration '{written}' for the bundle (C3): {exc}"
            ) from exc
    return files


def _engine_log_problem(path, report_dir, declared):
    """Say why the declared engine log cannot go into the bundle, or return None.

    Checked before the first request, because ``write_bundle`` hashes this file after the
    last one: a typo in the path, or a log outside the report directory, would otherwise
    surface at the one moment when the records exist nowhere but in RAM.

    ``declared`` is the string the operator wrote and ``path`` is where it resolved. A
    refusal that prints only one of them tells the operator nothing about which tree the
    harness looked in: ``calib/vllm_server.out is not a file`` was true of the working
    directory and false of the config's, and it refused a run that was correct in every
    respect except the directory it was launched from.

    Where the log lives only matters once it is too big to copy. ``write_bundle`` snapshots
    a log at or under its cap into the bundle, so /var/log and /tmp are fine and the bytes
    the report cites are frozen against a server that is still running. Above the cap the
    bundle can only name the file, and a name outside the report directory resolves on one
    machine, so that is the case this still refuses.
    """
    if not path.is_file():
        return (
            f"the declared engine log '{declared}', resolved against the config file's "
            f"directory to {path}, is not a file; every relative path in a bench config "
            "resolves against the config file's directory, not the working directory "
            "the command was invoked from"
        )
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        return f"the declared engine log '{declared}', resolved to {path}, cannot be read: {exc}"
    if path.stat().st_size > persist.ENGINE_LOG_SNAPSHOT_MAX_BYTES:
        try:
            path.resolve().relative_to(report_dir.resolve())
        except ValueError:
            return (
                f"the declared engine log '{declared}', resolved to {path}, is "
                f"{path.stat().st_size:,} bytes, above the "
                f"{persist.ENGINE_LOG_SNAPSHOT_MAX_BYTES:,}-byte cap for copying it into the "
                f"bundle, and it is outside the report directory {report_dir}; the "
                "reproduction table would have to name it with a path that resolves only "
                "on this machine. Trim it, or copy it under the report directory"
            )
    return None


def plan_lines(config, workload_obj=None):
    """Render the dry-run plan: what will be measured and the minimum wall clock it costs.

    The wall-clock figure is a floor, not an estimate of the mean: warm-up, request bodies,
    and the final drain all add to it, and the run that gets killed at a Slurm wall clock is
    the one that had the highest concurrency rung still in it.

    ``workload_obj`` is optional so that anything holding a config alone still gets a plan.
    Passing it adds the lines that need the corpus rather than the declaration, which for a
    session replay is the comparison an operator most needs before spending the hours: a
    window shorter than the sessions being replayed completes none of them.
    """
    endpoint = config["endpoint"]
    window = config["window"]
    lad = config["ladder"]
    rungs = lad["concurrency"]
    repetitions = lad["repetitions"]
    graded = len(rungs) * repetitions
    # Plus the section 5 confirmation repetition, which is run at the boundary rung once the
    # search is over. It is conditional -- a ladder where no rung passes its gates has no
    # boundary to confirm -- so the count is a range and the wall clock takes the high end.
    windows = graded + 1
    wall_s = windows * (window["window_s"] + window["drain_deadline_s"])
    lines = []
    if isinstance(workload_obj, workloads.SessionWorkload):
        plan = workload_obj.source
        shapes = plan.shapes
        clocks = sorted(shape.wall_clock_s for shape in shapes)
        steps = [len(shape.steps) for shape in shapes]
        longest = clocks[-1]
        median = clocks[len(clocks) // 2]
        lines += [
            f"replaying {plan.size} captured sessions "
            f"({min(steps)} to {max(steps)} steps each, {sum(steps)} steps in total), "
            f"drawn {plan.sampler_rule}",
            f"captured session wall clock: median {median:.1f} s, longest {longest:.1f} s, "
            f"against a window_s of {window['window_s']} s",
        ]
        if longest > window["window_s"]:
            # Not a refusal: a short window is a legitimate smoke run, and the operator may
            # know it. But it is the one thing about a replay that a report cannot recover
            # from silently -- the truncated sessions contribute only their opening turns,
            # which carry the short prompts, so the measured context and the prefill floor
            # both come out below the capture and in the flattering direction.
            lines.append(
                "at least one captured session is longer than the window, so some sessions "
                "will be cut off mid-conversation and contribute only their early, shorter "
                "turns; sessions_started and sessions_completed in the bundle are what to "
                "read that against"
            )
    return lines + [
        f"endpoint: {endpoint['base_url']} (model: {endpoint['model']})",
        f"concurrency rungs: {', '.join(str(r) for r in rungs)}",
        f"repetitions per rung: {repetitions}",
        f"windows: {graded} to {windows} ({len(rungs)} rungs * {repetitions} repetitions "
        "each, plus one section 5 confirmation repetition at the boundary rung if the "
        "ladder finds one)",
        f"warm-up requests per window: {window['warmup_requests']}",
        f"minimum wall-clock estimate: {wall_s:.0f} s "
        f"({windows} windows * (window_s {window['window_s']} s + drain_deadline_s "
        f"{window['drain_deadline_s']} s))",
        "the measured request count is not knowable in advance: this is a closed loop, "
        "so arrivals depend on completions rather than on a declared rate",
    ]


def _build_session_plan(wl, corpus_path, config_dir):
    """Load the captured shapes and refuse every declaration the replay would ignore.

    A key that has no effect is a key the operator misread, and here each one of them is a
    claim the published config makes about traffic the capture actually decides. All of it
    is checked before the first request, because an eight-hour ladder that turns out to
    have replayed the wrong thing has already spent the GPU hours.
    """
    if wl.get("media_root") is not None:
        raise ConfigError(
            "'replay_sessions' is true and 'media_root' is set (section 10): the captured "
            "shapes carry token counts and timings, not images, so this run would send no "
            "media at all while publishing a media label"
        )
    # No input_tokens rule here, deliberately. A shapes file is a corpus file, so
    # _check_values has already refused a non-null input_tokens against it by the time this
    # runs, and a second check would be unreachable -- a rule that cannot fire is one a
    # reader will eventually trust and a maintainer will eventually break without noticing.
    # Only the length is tested, not `or wl["ignore_eos"]`, and for the same reason as
    # input_tokens above: _check_values already refuses ignore_eos true with a null length,
    # so the only way to arrive here with ignore_eos set is to have declared a length too,
    # which the test below catches. The disjunct would read as the rule that stops a bare
    # ignore_eos and would never once have run.
    if wl["output_tokens"] is not None:
        raise ConfigError(
            "'replay_sessions' is true but an output length is declared (section 10): the "
            "captured step carries its own generated length and the replay forces it with "
            "ignore_eos. A single declared length would flatten the output side of the "
            "shape, which is half of what a session costs. Declare output_tokens null and "
            "ignore_eos false"
        )
    if float(wl["think_time_s"]) != 0.0:
        raise ConfigError(
            f"'replay_sessions' is true but 'think_time_s' is {wl['think_time_s']} "
            "(section 10): the capture already carries the measured gap after every step, "
            "so this would add a second idle period on top of the one being replayed and "
            "stretch every session past what was observed. Declare 0"
        )
    if wl["cache_policy"] == "unique-prefix":
        raise ConfigError(
            "'replay_sessions' is true but 'cache_policy' is 'unique-prefix' (section 10): "
            "a replayed session shares prefixes on purpose -- turn k's prompt begins with "
            "turn k-1's, which is the reuse an agent deployment actually gets -- so the "
            "declaration would deny in the config exactly what the run measures. "
            "'declared-workload' is the policy that says the reuse is the workload's own"
        )
    try:
        shapes, shared_prefix_tokens = sessions.load_shapes(corpus_path)
    except ValueError as exc:
        raise ConfigError(f"the captured shapes '{corpus_path}' cannot be replayed: {exc}") from exc
    return sessions.ReplaySessionPlan(
        shapes=shapes,
        seed=int(wl["seed"]),
        # Whitespace again, and for the same reason SyntheticCorpus uses it: the plan
        # refuses to publish a prompt that misses its target token count, so an oracle
        # that moves in jumps makes the shapes unreplayable rather than approximate.
        # What it costs is the same thing -- the counts are word counts, and the results
        # table publishes the server's own numbers instead.
        tokenizer=lambda text: len(text.split()),
        shared_prefix_tokens=shared_prefix_tokens,
        label=Path(corpus_path).stem,
    )


def _build_multimodal_corpus(wl, corpus_path, media_root, config_dir):
    """Construct the multimodal corpus, refusing every media misdeclaration while it is free.

    A media benchmark that degrades to text at request time publishes text numbers under a
    media label, so every check here happens before the first request is sent.
    """
    root = Path(media_root)
    if not root.is_absolute():
        root = Path(config_dir) / root
    if not root.is_dir():
        raise ConfigError(
            f"workload media_root is not an existing directory: '{root}' (section 9). "
            "The corpus's relative media paths resolve against it, and a run that "
            "cannot find its images measures a text workload under a media label"
        )
    transport = wl.get("image_input_transport", "base64")
    url_prefix = wl.get("media_url_prefix")
    if transport == "url" and not url_prefix:
        raise ConfigError(
            "'image_input_transport' is 'url' but 'media_url_prefix' is not set "
            "(section 9): the server fetches the media from that base URL, and without "
            "it every request would carry a URL that resolves to nothing"
        )
    if transport != "url" and url_prefix is not None:
        raise ConfigError(
            f"'media_url_prefix' is set but 'image_input_transport' is {transport!r} "
            "(section 9): with 'base64' the bytes travel in the request body and the "
            "prefix would be ignored, and a config whose keys do not all take effect "
            "is a config the operator misread"
        )
    try:
        return workloads.MultimodalJsonlCorpus(
            path=corpus_path,
            media_root=root,
            transport=transport,
            url_prefix=url_prefix,
            prompt_field=wl.get("prompt_field", "conversations"),
            max_records=wl.get("media_max_records"),
        )
    except ValueError as exc:
        raise ConfigError(
            f"the multimodal corpus '{corpus_path}' cannot be replayed: {exc}. Refusing "
            "before the first request is the point: a marker/reference mismatch "
            "discovered at request time has already spent the GPU hours"
        ) from exc


def _build_workload(config, config_dir):
    """Construct the Workload the ladder will replay, from the declared config only."""
    wl = config["workload"]
    corpus = wl["corpus"]
    media_root = wl.get("media_root")
    if media_root is not None and corpus == "synthetic":
        raise ConfigError(
            "'media_root' is set but 'corpus' is 'synthetic': a synthetic corpus has no "
            "media, so this config asks for a media run and would silently measure a "
            "text one (section 9)"
        )
    strip_media = bool(wl.get("strip_media_placeholders", False))
    if strip_media and (media_root is not None or corpus == "synthetic"):
        raise ConfigError(
            "'strip_media_placeholders' is true, but this config is not reading a corpus as "
            "text: it declares "
            + ("'media_root'" if media_root is not None else "the synthetic corpus")
            + " (section 9). The key exists to send a media-bearing corpus as its text-only "
            "variant, so accepting it here would let a report claim the media was stripped "
            "from a run that sent every image"
        )
    replay_sessions = bool(wl.get("replay_sessions", False))
    if replay_sessions and corpus == "synthetic":
        raise ConfigError(
            "'replay_sessions' is true but 'corpus' is 'synthetic' (section 10): there is "
            "nothing to replay. Point 'corpus' at the shapes file written by "
            "`ascep agent-profile --shapes`"
        )
    if corpus == "synthetic":
        # Whitespace, not characters-per-token. SyntheticCorpus pads one filler word at a
        # time and refuses to publish a prompt that misses the target exactly, so an oracle
        # that moves in jumps of two or three -- which every characters-per-token estimate
        # does -- makes the corpus unbuildable rather than approximate. One word per token
        # lands on the target by construction; what it costs is that the declared
        # input_tokens is a word count, which is why run.tokenizer stays null with a reason
        # and the results table publishes the server's own count instead.
        #
        # A word count is only a usable stand-in because SyntheticCorpus pads with common
        # English words. Measured against Gemma 4 on a GB200, the random-hex filler this
        # used to emit cost 7.98 tokens per word, so a config asking for 1,500 sent about
        # 12,000 and every context and KV figure downstream described a workload nobody
        # declared. The same 256 words of _FILLER_WORDS came to exactly 256 tokens.
        source = workloads.SyntheticCorpus(
            input_tokens=int(wl["input_tokens"]), tokenizer=lambda text: len(text.split())
        )
    else:
        corpus_path = Path(corpus)
        if not corpus_path.is_absolute():
            corpus_path = Path(config_dir) / corpus_path
        if not corpus_path.is_file():
            raise ConfigError(
                f"workload corpus file not found: '{corpus}' (section 7). Refusing to "
                "spend GPU hours against a corpus that cannot be replayed"
            )
        if replay_sessions:
            # Returns here rather than falling through to the output-plan block below,
            # because there is no single output length to plan: the capture carries one
            # per step. _build_session_plan has already refused every declaration that
            # would have fed that block.
            return workloads.SessionWorkload(
                source=_build_session_plan(wl, corpus_path, config_dir),
                # Unused -- workloads.SessionWorkload.spec takes each step's length from the
                # capture -- but the base dataclass requires one of the three plans, and
                # this is the one that claims the least: no length was declared here.
                output_plan=workloads.ModelDecidedOutput(),
                cache_policy=wl["cache_policy"],
                seed=int(wl["seed"]),
                think_time_s=0.0,
                run_label=wl["run_label"],
            )
        if media_root is None:
            # 'prompt_field' is read here as well as on the multimodal path. It used to be
            # honoured only when 'media_root' was set, and a text corpus was pinned to a
            # hardcoded "messages" no protocol section names and no published config ever
            # carried -- so a text campaign against any post-training corpus died at load,
            # and an operator who declared 'prompt_field' had it silently ignored.
            source = workloads.JsonlCorpus(
                corpus_path,
                field=wl.get("prompt_field", "conversations"),
                strip_media_placeholders=bool(wl.get("strip_media_placeholders", False)),
            )
        else:
            source = _build_multimodal_corpus(wl, corpus_path, media_root, config_dir)
    if wl["ignore_eos"]:
        output_plan = workloads.FixedOutput(int(wl["output_tokens"]), True)
    elif wl["output_tokens"] is not None:
        # EOS honoured, the declared number a ceiling rather than a commitment: the mode a
        # production latency claim is actually measured in. The two-state code read,
        # type-checked and range-checked this number and then built the uncapped plan, so a
        # declared 512 went out as no cap at all and one degenerate generation monopolised
        # an entire measurement window.
        output_plan = workloads.CappedOutput(int(wl["output_tokens"]))
    else:
        # An explicit null: the one mode where no length goes on the wire. The choice to
        # run uncapped is declared here, never fallen into because a number was quietly
        # dropped on the way.
        output_plan = workloads.ModelDecidedOutput()
    workload_cls = (
        workloads.MediaShapeWorkload
        if isinstance(source, workloads.MultimodalJsonlCorpus)
        else workloads.Workload
    )
    return workload_cls(
        source=source,
        output_plan=output_plan,
        cache_policy=wl["cache_policy"],
        seed=int(wl["seed"]),
        think_time_s=float(wl["think_time_s"]),
        run_label=wl["run_label"],
    )


async def _one_window(adapter, workload_obj, config, gates, *, concurrency, repetition):
    """Run and reduce a single window at one operating point, returning ``(run, summary)``."""
    window = config["window"]
    policy = driver.WindowPolicy(
        concurrency=concurrency,
        window_s=float(window["window_s"]),
        drain_deadline_s=float(window["drain_deadline_s"]),
        think_time_s=float(workload_obj.think_time_s),
        warmup_requests=int(window["warmup_requests"]),
        # The bundle labels every window from its policy, so a repetition left at the
        # default publishes nine windows all claiming to be repetition 0 -- and the
        # dispersion across repetitions is the whole point of section 7.5.
        repetition=repetition,
    )
    if isinstance(workload_obj, workloads.SessionWorkload):
        # The plan itself, not a generator bound to this window. One instance serves the
        # whole ladder: every render is a pure function of the session and step indices,
        # and the driver spaces those indices apart per operating point, so nothing here
        # needs rebinding and nothing can carry state from one rung into the next.
        run = await driver.run_window(
            adapter, policy=policy, reset=driver.no_reset, session_plan=workload_obj.source
        )
    else:
        run = await driver.run_window(
            adapter,
            workload_obj.for_repetition(repetition, concurrency=concurrency),
            policy=policy,
            reset=driver.no_reset,
        )
    summary = metrics.reduce_window(
        run.records,
        window_s=run.window_s,
        t0=run.t0,
        gates=gates,
        seed=workload_obj.seed,
    )
    return run, summary


def _keep(state, run, summary, concurrency, repetition, *, post_search=False):
    """File one finished window in ``state`` under the rung it was measured at."""
    state["runs"].append(run)
    state["repetitions"].setdefault(concurrency, []).append(
        ladder.RepetitionResult(
            concurrency=concurrency,
            repetition=repetition,
            summary=summary,
            post_search=post_search,
        )
    )


async def _confirm_boundary(adapter, workload_obj, config, gates, policy, state, err):
    """Re-run the boundary rung once more, after the search that selected it has finished.

    Section 5 will not publish a Sustainable figure on the strength of the search that found
    it. The boundary is the rung the stopping rule stopped at *because* it passed, so the
    three windows behind it are the very evidence that selection conditioned on. One further
    repetition, taken when no decision depends on its outcome, is what separates "the rung
    the search landed on" from "a rung this system sustains".

    Skipping it is not a smaller claim, it is no claim: ``sustainable_publishable`` stays
    false and the draft publishes no sustainable tier at all. That makes this window part of
    the procedure rather than an optional extra, which is why bench runs it itself instead of
    leaving it to an operator who would have to know section 5 to know it was missing.
    """
    provisional = ladder.grade_ladder(state["repetitions"], policy)
    concurrency = provisional.max_sustainable_concurrency
    if concurrency is None:
        print(
            "ascep bench: no rung passed its declared gates, so there is no boundary to "
            "confirm and no sustainable tier to publish",
            file=err,
        )
        return
    # One past the counted repetitions: grading partitions on post_search rather than on the
    # index, but the index is what labels the window in the bundle and what salts its request
    # ids, and a fourth window numbered 0 would collide with the first.
    repetition = int(policy.repetitions)
    try:
        run, summary = await _one_window(
            adapter,
            workload_obj,
            config,
            gates,
            concurrency=concurrency,
            repetition=repetition,
        )
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        state["censor"] = (
            f"stopped during the section 5 confirmation repetition at "
            f"concurrency={concurrency} ({_cause_of(exc)})"
        )
        return
    _keep(state, run, summary, concurrency, repetition, post_search=True)
    print(
        f"ascep bench: confirmation at concurrency={concurrency}: "
        f"{summary.n_completed} completed, slo_pass={summary.slo_pass!r}",
        file=err,
    )


async def _run_ladder(adapter, workload_obj, config, gates, policy, state, err):
    """Drive every (rung, repetition) window in this one coroutine, closing the adapter last.

    Results accumulate in ``state`` rather than in locals: when a real Ctrl-C lands, the
    asyncio runner cancels this coroutine and re-raises KeyboardInterrupt out of
    ``asyncio.run`` even if the cancellation was handled here, so the caller can only rely
    on mutated state, never on a return value.
    """
    lad = config["ladder"]
    try:
        for concurrency in lad["concurrency"]:
            for repetition in range(lad["repetitions"]):
                try:
                    run, summary = await _one_window(
                        adapter,
                        workload_obj,
                        config,
                        gates,
                        concurrency=int(concurrency),
                        repetition=repetition,
                    )
                except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                    # An interrupted ladder has real measurements for its lower rungs, and
                    # those are the rungs most likely to be sustainable anyway. A run that
                    # succeeded and left nothing behind is worse than a run that failed.
                    state["censor"] = (
                        f"stopped during concurrency={concurrency} "
                        f"repetition={repetition} ({_cause_of(exc)})"
                    )
                    return
                _keep(state, run, summary, int(concurrency), repetition)
                print(
                    f"ascep bench: concurrency={concurrency} repetition={repetition}: "
                    f"{summary.n_completed} completed, "
                    f"output_tok_s={summary.output_tok_s!r}, "
                    f"peak_in_flight={summary.peak_in_flight!r}",
                    file=err,
                )
                peak = summary.peak_in_flight
                # Half of the declaration is deliberately conservative, and the think-time
                # case is why: in a closed loop with think_time_s > 0 the virtual users
                # idle between requests, so expected in-flight is about concurrency times
                # service / (service + think), and warning there would cry wolf on every
                # healthy workload. Only a collapse well under half points at a throttle.
                if peak is not None and int(concurrency) > 1 and peak < 0.5 * concurrency:
                    print(
                        f"ascep bench: WARNING: peak_in_flight={peak} against the declared "
                        f"concurrency={concurrency}: less than half of the offered load was "
                        "ever in flight at once, so this rung may be throttled rather than "
                        "saturated. The usual causes are a client-side connection-pool or "
                        "file-descriptor cap, or a server-side admission limit; a peak "
                        "pinned at the same value across several rungs is the signature. "
                        "Check both limits before believing this rung as a capacity boundary",
                        file=err,
                    )
                completed = summary.n_completed
                # Fewer than three completions per user leaves the rate coarsely resolved:
                # over users whose cycles are near-identical the count is an integer, so
                # a rate estimated from k of them per user quantises in steps of about
                # 1/k. De-phasing removes the systematic bias of a phase-locked fleet;
                # it cannot sharpen what one window resolves.
                if (
                    completed is not None
                    and int(concurrency) > 0
                    and completed / int(concurrency) < 3.0
                ):
                    print(
                        f"ascep bench: WARNING: completions per user at this rung is "
                        f"{completed / int(concurrency):.2f} ({completed} completed at "
                        f"concurrency={concurrency}): a rate estimated from k cycles per "
                        "user quantises in steps of about 1/k, and de-phasing removes "
                        "the systematic bias of a phase-locked fleet but not this coarse "
                        "resolution. The remedy is a longer window_s (or a shorter "
                        "declared output length), not a re-read of this number",
                        file=err,
                    )
        await _confirm_boundary(adapter, workload_obj, config, gates, policy, state, err)
    finally:
        # The adapter's HTTP client belongs to this loop; closing it anywhere else risks a
        # cross-loop failure that only surfaces against a real server.
        await adapter.aclose()


def bench(config_path, *, dry_run=False, adapter_factory, out=None, err=None):
    """Run the concurrency ladder described by ``config_path`` and return the exit code.

    Every validation refusal returns 2 with a diagnostic on ``err``; ``--dry-run`` prints
    the plan to ``out``, writes nothing, and returns 0. A ladder that produced no completed
    window returns 1, because there is no evidence to bundle. Operator errors never
    propagate as exceptions: the whole point of this command is to refuse rather than to
    run under-specified.

    A run that wrote a bundle but did not complete as declared -- censored by an interrupt or
    an abort, or emitting a draft that fails its own schema -- returns 3. It is deliberately
    not 0: these runs are submitted as batch jobs, and the question a wrapper script asks
    afterwards is ``$?``. A truncated ladder that exits 0 gets swept into the results
    directory beside the complete ones, and the caveat that its concurrency figures are a
    lower bound survives only in a report nobody re-reads.
    """
    with _sigterm_as_interrupt():
        return _bench(
            config_path, dry_run=dry_run, adapter_factory=adapter_factory, out=out, err=err
        )


def _bench(config_path, *, dry_run, adapter_factory, out, err):
    """The body of :func:`bench`, minus the signal handling that has to wrap all of it."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    config_dir = Path(config_path).resolve().parent
    try:
        config, raw_config = load_config(config_path)
        declarations = load_declarations(config, config_dir)
        declared_bytes = _declaration_bytes(config, config_dir)
        workload_obj = _build_workload(config, config_dir)
    except ConfigError as exc:
        print(f"ascep bench: {exc}", file=err)
        return 2

    if dry_run:
        for line in plan_lines(config, workload_obj):
            print(line, file=out)
        return 0

    output = config["output"]
    # Outputs resolve against the config's directory exactly as inputs do; the process cwd
    # is otherwise part of the run's meaning, and a run launched from the wrong directory
    # splits the bundle from the declarations it was measured against. An absolute declared
    # value must not be dragged under the config directory either: Path.joinpath returns the
    # right operand when it is absolute, which is what saves it, but a reader will wonder,
    # so it is said here rather than discovered.
    bundle_dir = Path(config_dir) / output["bundle_dir"]
    if bundle_dir.exists():
        # The GPU hours behind an existing bundle are already spent and the records cannot
        # be regenerated, so overwrite is a refusal rather than an option.
        print(
            f"ascep bench: refusing to overwrite the existing bundle at {bundle_dir}; "
            "choose a new output.bundle_dir",
            file=err,
        )
        return 2

    # bundle_dir.parent is now anchored at the config, so a relative engine log moves with
    # the bundle instead of with the cwd: the old anchor let the C8 check hunt the log in
    # the tree the command was launched from while the bundle landed in another. Both sides
    # of _engine_log_problem's containment test sit in the same tree, so like is still
    # compared with like, and write_bundle below still receives the declared string, never
    # engine_logs: a reproduction table naming an absolute path from the machine that
    # produced the run cannot be checked by anyone else.
    engine_logs = Path(output["engine_logs_path"])
    if not engine_logs.is_absolute():
        engine_logs = bundle_dir.parent / engine_logs
    log_problem = _engine_log_problem(engine_logs, bundle_dir.parent, output["engine_logs_path"])
    if log_problem:
        print(f"ascep bench: {log_problem} (C8)", file=err)
        return 2

    gates, policy = ladder_policy(config)
    try:
        adapter = adapter_factory(config["endpoint"])
    except Exception as exc:
        print(f"ascep bench: could not build the endpoint adapter: {exc!r}", file=err)
        return 2

    state = {"runs": [], "repetitions": {}, "censor": None}
    try:
        # One event loop for the whole ladder: an httpx client built in one loop and awaited
        # from another raises only at runtime, against a real server, after the allocation
        # is spent.
        asyncio.run(_run_ladder(adapter, workload_obj, config, gates, policy, state, err))
    except KeyboardInterrupt:
        state["censor"] = state["censor"] or "interrupted (KeyboardInterrupt)"
    except asyncio.CancelledError:
        state["censor"] = state["censor"] or "cancelled before the ladder completed"
    except Exception as exc:
        # Unexpected failures get the same survivability rule as interrupts: bundle what
        # completed, and let the censoring cause carry why the ladder is truncated.
        state["censor"] = state["censor"] or f"aborted by {type(exc).__name__}: {exc}"
        print(
            f"ascep bench: the ladder aborted early ({exc!r}); bundling completed windows",
            file=err,
        )

    if not state["runs"]:
        print(
            "ascep bench: no window completed, so there is no evidence to bundle; "
            "nothing was written",
            file=err,
        )
        return 1

    result = ladder.grade_ladder(state["repetitions"], policy, censoring_cause=state["censor"])

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        c8 = persist.write_bundle(
            bundle_dir,
            relative_to=bundle_dir.parent,
            runs=state["runs"],
            workload=workload_obj,
            engine_logs_path=output["engine_logs_path"],
            container_digest=output["container_digest"],
            environment=persist.capture_environment(),
            extra_files={"bench-config.json": raw_config, **declared_bytes},
            overwrite=False,
        )
    except FileExistsError:
        print(
            f"ascep bench: refusing to overwrite the existing bundle at {bundle_dir}",
            file=err,
        )
        return 2
    except (OSError, KeyboardInterrupt) as exc:
        # A half-written bundle is worse than none: the records exist only in RAM, and the
        # directory left behind is exactly what the next attempt refuses to overwrite, so
        # the failure would also block the retry that might still have saved them.
        shutil.rmtree(bundle_dir, ignore_errors=True)
        print(
            f"ascep bench: writing the bundle failed ({exc!r}). The partial bundle at "
            f"{bundle_dir} was removed so a retry is not blocked; the measurements this "
            "run made are lost",
            file=err,
        )
        return 1

    draft = report.build_report(
        config,
        declarations,
        state["runs"],
        state["repetitions"],
        result,
        c8,
        state["censor"],
    )
    # Same anchor as bundle_dir above: the draft lands beside the bundle and the config, not
    # beside wherever the command happened to be launched, or a run's evidence splits across
    # two trees that can only name each other through cwd-relative paths.
    report_path = Path(config_dir) / output["report_path"]
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        # Unlike the bundle, this one is recoverable: the records and the config are on
        # disk and the draft is derived from them, so say where they are rather than
        # deleting anything.
        print(
            f"ascep bench: the draft report could not be written to {report_path} ({exc}), "
            f"but the measurements survive in the bundle at {bundle_dir}",
            file=err,
        )
        return 3

    # Schema validation here checks bench's own assembly, never its results: a draft that
    # fails the schema is a defect in this file, and the operator deserves to be told the
    # report they hold is not yet load-bearing rather than discovering it downstream.
    problems = validation.validate("capacity-report", draft)
    if problems:
        print(
            "ascep bench: the emitted draft does not validate; this is a defect in bench:",
            file=err,
        )
        for problem in problems:
            print(f"  - {problem}", file=err)
    print(f"ascep bench: bundle at {bundle_dir}; draft report at {report_path}", file=err)
    if state["censor"]:
        print(
            f"ascep bench: the ladder did not complete as declared ({state['censor']}); "
            "every concurrency figure in this draft is a lower bound",
            file=err,
        )
    return 3 if (problems or state["censor"]) else 0
