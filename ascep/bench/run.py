"""``ascep bench``: run a declared concurrency ladder and emit a draft capacity report.

Every other command in this toolkit grades evidence someone else produced; this one produces
the evidence. Its contract is therefore inverted relative to the rest of the CLI: refuse to
run rather than run under-specified, never invent a value the operator did not declare, and
never grade its own output. A wrong number emitted here is not caught downstream -- it *is*
the downstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from ascep import init, validation
from ascep.bench import driver, ladder, metrics, persist, workloads

__all__ = ["ConfigError", "load_config", "load_declarations", "plan_lines", "bench"]


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
        # Where the prompt text lives in each record; default "conversations".
        "prompt_field": "section 9",
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
        _ladder_policy(config)
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


def _ladder_policy(config):
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
    try:
        path.resolve().relative_to(report_dir.resolve())
    except ValueError:
        return (
            f"the declared engine log '{declared}', resolved to {path}, is outside the "
            f"report directory {report_dir}, so the reproduction table would have to name "
            "it with a path that resolves only on this machine; copy it under the report "
            "directory before the run"
        )
    return None


def plan_lines(config):
    """Render the dry-run plan: what will be measured and the minimum wall clock it costs.

    The wall-clock figure is a floor, not an estimate of the mean: warm-up, request bodies,
    and the final drain all add to it, and the run that gets killed at a Slurm wall clock is
    the one that had the highest concurrency rung still in it.
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
    return [
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


class _MediaShapeWorkload(workloads.Workload):
    """A Workload whose manifest also carries the corpus's measured media shape.

    C4 requires images_per_request, videos_per_request and their kin beside any throughput
    figure, and the only version of those numbers that is not someone's recollection is the
    one measured off the corpus itself. Text workloads keep the plain manifest with the key
    absent entirely: an absent key and a zeroed one say different things, and a run that
    carried no media has no media shape to declare.
    """

    def manifest(self):
        return {**super().manifest(), "media_shape": self.source.media_shape()}


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
    if corpus == "synthetic":
        # Whitespace, not characters-per-token. SyntheticCorpus pads one filler word at a
        # time and refuses to publish a prompt that misses the target exactly, so an oracle
        # that moves in jumps of two or three -- which every characters-per-token estimate
        # does -- makes the corpus unbuildable rather than approximate. One word per token
        # lands on the target by construction; what it costs is that the declared
        # input_tokens is a word count, which is why run.tokenizer stays null with a reason
        # and the results table publishes the server's own count instead.
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
        if media_root is None:
            source = workloads.JsonlCorpus(corpus_path, field="messages")
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
        _MediaShapeWorkload
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
                    f"output_tok_s={summary.output_tok_s!r}",
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
        for line in plan_lines(config):
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

    gates, policy = _ladder_policy(config)
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

    report = _build_report(
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
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    problems = validation.validate("capacity-report", report)
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


def _known(node, key, value):
    """Set ``node[key]`` and drop the ``_u_reason`` companion that the value makes false."""
    node[key] = value
    node.pop(f"{key}_u_reason", None)


def _unknown(node, key, reason):
    """Mark ``node[key]`` unknown, only where the skeleton already emitted a companion.

    The report schemas set ``additionalProperties: false`` on most objects, so inventing a
    ``_u_reason`` the skeleton did not emit fails validation for a cause the operator did
    not create -- exactly the laundering C2 exists to stop.
    """
    companion = f"{key}_u_reason"
    if companion in node:
        node[key] = None
        node[companion] = f"(U) {reason}"


def _measured(node, key, value, reason):
    """Set a measured value, or record why the measurement produced nothing."""
    if value is None:
        _unknown(node, key, reason)
    else:
        _known(node, key, value)


def _median_repetition(repetitions):
    """Pick the median repetition of a rung by output_tok_s (lower median on a tie).

    Averaging percentiles across windows would publish a row no window ever exhibited; the
    figures in one row have to be mutually consistent, so a single real repetition stands
    in for the rung.

    A repetition whose reduction produced no throughput figure at all is ranked with the
    others only if every repetition is in that state. ``None`` there means no completed
    record reported its output tokens, which is neither a fast window nor a zero one, and
    sorting it either way is a claim: at the top it becomes the median of a half-collapsed
    rung and publishes the best window as typical.
    """
    ranked = [rep for rep in repetitions if rep.summary.output_tok_s is not None] or list(
        repetitions
    )
    ordered = sorted(ranked, key=lambda rep: rep.summary.output_tok_s or 0.0)
    return ordered[(len(ordered) - 1) // 2]


def _counted(reps):
    """The repetitions a rung is graded on: the declared three, not the confirmation.

    The section 5 confirmation window is additional evidence about a boundary, never one of
    the repetitions the rung is scored from -- ``grade_rung`` partitions it out, and a row
    whose median came from a window the rung's own throughput median excluded would be a
    row no reader could reconcile with the grade beside it.
    """
    return [rep for rep in reps if not rep.post_search]


def _reason_for(summary, field, fallback):
    """Prefer the reducer's own account of a missing figure over a generic one.

    The reducer knows what a generic caller cannot: how many samples survived and which
    section sets the floor they missed. Its reasons already carry the ``(U)`` tag, which
    :func:`_unknown` adds, so it comes off here rather than being published twice.
    """
    stated = summary.reasons.get(field)
    if not stated:
        return fallback
    return stated[len("(U)") :].strip() if stated.startswith("(U)") else stated


def _window_of(runs, concurrency, repetition):
    """Find the WindowRun a graded repetition came from, or None if it is not in the bundle."""
    for run in runs:
        if run.policy.concurrency == concurrency and run.policy.repetition == repetition:
            return run
    return None


def _mean_reported(records, field):
    """Mean of the server's own token count over the records that carry one, else None.

    The adapter asks for ``include_usage`` and writes these straight from the response, so
    they are the only token counts in the run that were counted rather than declared. What
    the operator configured is a request, not a measurement: the synthetic corpus counts
    whitespace, so a declared 512 is 512 words and perhaps 650 tokens, and publishing it in
    a row tagged (M) understates the context every throughput figure in that row was
    produced at.
    """
    values = [
        getattr(record, field)
        for record in records
        if getattr(record, field, None) is not None and record.in_window
    ]
    return sum(values) / len(values) if values else None


def _mean_context(records):
    """Mean of the per-request context lengths the server counted, else None.

    C4 binds every throughput figure in a row to the context length beside it, so that
    length must be one some request actually occupied. mean(input_tokens) +
    mean(output_tokens) fails that: each mean is taken over whichever records happened
    to carry that count, so a record reporting only one side still votes in its half,
    and the sum of the two can be a context no request ever had -- yet it would be
    published in a row tagged (M). A record therefore contributes its input + output
    only when it carries both counts; a half-reported request contributes nothing.
    """
    contexts = [
        record.input_tokens + record.output_tokens
        for record in records
        if record.in_window and record.input_tokens is not None and record.output_tokens is not None
    ]
    mean = sum(contexts) / len(contexts) if contexts else None
    # An empty prompt with an empty completion can report 0 + 0, and a mean of those is
    # 0.0: a measured context of no tokens, which is not a context. The schema floor is
    # exclusive anyway, so it takes the same justified null as no measurement at all.
    return mean if mean else None


def _conformance_note(censor, result):
    """Write the note that stops ``non-conforming`` reading as a verdict on the hardware."""
    note = (
        "This report is an ungraded draft emitted by `ascep bench`. A load generator "
        "observes latency and throughput over HTTP and nothing else, so the four report "
        "sections it cannot observe -- the roofline comparison, the sizing result, the "
        "scaling table, and the theoretical and recommended capacity tiers -- are left "
        "unknown rather than estimated. `non-conforming` means this draft has not been "
        "graded, not that the hardware failed; `ascep conformance` is the command that "
        "may raise the claim."
    )
    if censor:
        note += (
            f" The ladder was censored ({censor}), so every concurrency figure in this "
            "report is a lower bound on the hardware, not a finding."
        )
    if result.is_lower_bound:
        # A ladder that ran out of declared rungs before anything failed found the highest
        # concurrency it was *asked* about. Reported without this sentence it understates
        # the hardware, and the next person orders GPUs against it.
        note += (
            " The ladder was exhausted without a failing rung"
            + (f" ({result.censoring_cause})" if result.censoring_cause else "")
            + ", so the concurrency figures are a lower bound -- at least this much, not "
            "this much at most. Extend the declared rungs to find the boundary."
        )
    if result.cache_caveat:
        note += f" {result.cache_caveat}"
    return note


_TIER_FIELDS = (
    "max_concurrent_users",
    "max_tokens_per_s",
    "max_requests_per_s",
    "daily_requests",
)


def _boundary_constraint(result, concurrency):
    """Name the floor the ladder actually hit above ``concurrency``, or say nothing.

    Chapter 5 settles the label: a rung that failed its declared gates has observed which
    floor binds, and ``slo`` overrides the constraint label exactly there. A rung that
    delivered nothing collapsed on throughput, not on a latency gate. ABORTED rungs are
    excluded deliberately: they are failure evidence by cause, not evidence that a floor
    binds. A ladder exhausted without a failing rung has shown no floor at all -- its
    figure is "at least this much", and naming a constraint there would print a lower
    bound as a measured maximum.
    """
    failures = [
        rung
        for rung in result.rungs.values()
        if rung.concurrency > concurrency and rung.outcome is ladder.RungOutcome.FAILED
    ]
    if not failures:
        return None
    boundary = min(failures, key=lambda rung: rung.concurrency)
    return "throughput" if boundary.zero_completions else "slo"


def _fill_tier(tier, concurrency, rung, median):
    """Fill one capacity tier from a graded rung and its median repetition."""
    _known(tier, "max_concurrent_users", concurrency)
    _measured(
        tier,
        "max_tokens_per_s",
        rung.throughput_tok_s,
        "the graded rung carried no sustainable throughput figure",
    )
    _measured(
        tier,
        "max_requests_per_s",
        median.summary.requests_per_s,
        "the median repetition for this rung produced no request rate",
    )
    rps = median.summary.requests_per_s
    _measured(
        tier,
        "daily_requests",
        None if rps is None else rps * 86400.0,
        "the median repetition for this rung produced no request rate to extrapolate",
    )
    tier["provenance"] = "M"


def _build_report(config, declarations, runs, repetitions, result, c8, censor):
    """Assemble the draft capacity report from measured rungs and declared documents.

    Bench fills only what it measured or was told; everything else stays null with a reason.
    Emitting keys absent would hand the operator a file that fails validation for causes
    they did not create, and filling them with estimates would launder guesses into
    measurements.
    """
    report = init.skeleton("capacity-report")
    # Bench measures; it does not normalise the operator's declarations on the way past.
    for layer in ("hardware", "model", "serving", "workload"):
        report[layer] = copy.deepcopy(declarations[layer])

    report["report_generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # The schema requires a claim, so bench makes the weakest one the enum offers; a
    # harness that grades its own results is the failure the negative corpus demonstrates.
    report["conformance"] = "non-conforming"
    report["conformance_note"] = _conformance_note(censor, result)

    window = config["window"]
    lad = config["ladder"]
    run_block = report["run"]
    _unknown(
        run_block,
        "engine_version",
        "bench observes an HTTP endpoint, not a build; the engine version was never "
        "visible to the load generator",
    )
    _measured(
        run_block,
        "container_digest",
        config["output"]["container_digest"],
        "the operator did not declare a container digest; bench pulls HTTP responses, "
        "not image registries",
    )
    warmups = [run.warmup_s_actual for run in runs]
    _measured(
        run_block,
        "warmup_seconds",
        sum(warmups) / len(warmups) if warmups else None,
        "no window ran, so there is no warm-up duration to average",
    )
    _known(run_block, "repeats", lad["repetitions"])
    _known(run_block, "sustained_window_seconds", window["window_s"])
    rung_list = [int(c) for c in lad["concurrency"]]
    _known(run_block, "concurrency_ladder", rung_list)
    _known(run_block, "drain_deadline_seconds", window["drain_deadline_s"])
    _known(run_block, "throughput_collapse_ratio", lad["throughput_collapse_ratio"])
    _known(run_block, "monotonic_across_ladder", result.monotone)
    # metrics.percentile implements the Hyndman-Fan type-7 estimator; naming any other
    # method here would make every percentile in this file unreproducible, so this string
    # is a statement about the code, not a guess.
    _known(run_block, "percentile_method", "hyndman-fan-type-7")
    _unknown(
        run_block,
        "tokenizer",
        "bench ships no tokenizer: the local token count required by chapter 4.7.1 was "
        "not taken, so the server's usage numbers are unchecked",
    )
    _known(run_block, "outlier_method", "none")
    _known(run_block, "open_loop", False)
    gates_node = run_block["slo_gates"]
    for gate_key in ("ttft_p95_max_s", "itl_p95_max_s", "e2e_p95_max_s", "error_rate_max_pct"):
        _measured(
            gates_node,
            gate_key,
            config["slo_gates"][gate_key],
            "no gate for this metric was declared in the bench config",
        )
    gates_node["declared_before_run"] = True
    for path_key in ("environment_capture_path", "raw_records_path", "engine_logs_path"):
        _measured(
            run_block,
            path_key,
            c8.get(path_key),
            "the reproduction bundle did not record this path",
        )

    # One row per rung, in ladder order: publishing only the winning rung would throw away
    # the shape of the curve, which is the part that says whether the tier is a plateau or
    # a cliff edge one request wide. A rung with no completed window has no evidence row
    # to publish, and the ladder's censoring cause explains its absence.
    row_template = copy.deepcopy(run_block["results"][0])
    rows = []
    # The chunk-gap fields sit with the ITL figures because they are the transport trace the
    # population label is about: a row whose ITL moved to per-request-mean is only auditable
    # if the factor and the observed chunk gaps ride on the same row.
    summary_fields = (
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
    )
    # Those three are optional in the schema, so `ascep init` does not emit them and neither
    # does the row template. _unknown fills only a companion that already exists -- right for
    # a hand-filled report, wrong here: a transport figure the reduction computed and found
    # empty would vanish from the row, and an absent key reads as "this rung never looked"
    # when the truth is "it looked and the stamps were not there". Seeded so the (U) has
    # somewhere to land; _known pops the companion on every rung that measured a value.
    for name in ("tokens_per_stream_chunk", "stream_chunk_gap_p50_s", "stream_chunk_gap_p95_s"):
        row_template.setdefault(f"{name}_u_reason", init.TODO)
    thin = []
    for concurrency in rung_list:
        rung = result.rungs.get(concurrency)
        reps = _counted(repetitions.get(concurrency, []))
        if rung is None or not reps:
            continue
        median = _median_repetition(reps)
        summary = median.summary
        row = copy.deepcopy(row_template)
        # The row is tagged (M), so its token counts have to be the server's, not the
        # config's. The declared figures are what was asked for; ignore_eos makes the output
        # side agree, but nothing makes the input side agree, and C4 binds every throughput
        # figure in this row to the context length beside it.
        window_run = _window_of(runs, concurrency, median.repetition)
        records = window_run.records if window_run is not None else []
        no_usage = (
            "the server returned no usage accounting for this rung, so the tokens these "
            "prompts actually cost were never counted; the configured value is a request, "
            "not a measurement, and publishing it in a row tagged (M) would launder it"
        )
        _measured(row, "context_tokens", _mean_context(records), no_usage)
        _measured(row, "input_tokens", _mean_reported(records, "input_tokens"), no_usage)
        _measured(row, "output_tokens", _mean_reported(records, "output_tokens"), no_usage)
        _known(row, "concurrency", concurrency)
        for field in summary_fields:
            _measured(
                row,
                field,
                getattr(summary, field),
                _reason_for(
                    summary,
                    field,
                    f"the window reduction produced no {field} for this rung; too few "
                    "completed samples survived exclusion",
                ),
            )
        # A published figure between the two section 4.3 floors is real but one straggler
        # wide. It cannot be flagged in the row -- the schema has no field for it -- so it
        # goes in the assumptions table, which is where a reviewer looks for what the
        # numbers cannot settle.
        thin.extend(
            f"{field} at concurrency {concurrency}"
            for field in sorted(summary.low_confidence & set(summary_fields))
        )
        _measured(
            row,
            "slo_pass",
            summary.slo_pass,
            "the reducer could not grade this rung against the declared gates",
        )
        _known(row, "outcome", rung.outcome.value.lower())
        # A non-COMPLETE row must arrive with the sentence that produced the verdict:
        # without it, a rung can publish "failed" beside a passing slo_pass -- two keys
        # answering different questions -- and read as a harness contradiction, leaving the
        # operator to re-derive from records.jsonl what grading already knew.
        if rung.reasons:
            _known(row, "reasons", list(rung.reasons))
        _unknown(
            row,
            "gpu_util_pct",
            "a load generator cannot see the GPU; this must come from the serving host",
        )
        _unknown(
            row,
            "gpu_mem_util_pct",
            "a load generator cannot see the GPU; this must come from the serving host",
        )
        row["provenance"] = "M"
        rows.append(row)
    run_block["results"] = rows

    # Belt and braces for skeleton drift: any surviving TODO companion in the block bench
    # owns is an unknown with an honest (if generic) reason, never a fabricated value.
    for key in list(run_block):
        companion = f"{key}_u_reason"
        if companion in run_block and "TODO" in str(run_block[companion]):
            _unknown(
                run_block,
                key,
                "bench did not measure this; it is not observable by a load generator over HTTP",
            )

    # Fewer than three measured context lengths is a point, not a scaling curve. Setting the
    # flag does not raise the grade -- it states the limit the grade already reflects, and an
    # unlabelled single point reads as a curve to whoever builds on the draft. single_point is
    # a plain boolean with no _u_reason companion, so it is set directly, after the sweep
    # above so nothing clobbers it.
    context_lengths = {
        row["context_tokens"]
        for row in rows
        if isinstance(row.get("context_tokens"), (int, float))
        and not isinstance(row["context_tokens"], bool)
    }
    run_block["single_point"] = len(context_lengths) < 3

    tiers = report["capacity_tiers"]
    # n_gpus is the topology the run was bound to in all four tiers, not a per-tier finding.
    n_gpus = declarations["serving"]["gpu_count"]
    for tier in tiers.values():
        _known(tier, "n_gpus", n_gpus)
        _unknown(
            tier,
            "binding_constraint",
            "bench exercises the throughput and latency floors only; the weights and KV "
            "floors are never evaluated, and a rung whose gates held still says nothing "
            "about whether KV would have bound first at another context length",
        )

    roofline_reason = (
        "the roofline needs the hardware's bandwidth and FLOPs and is computed by "
        "`ascep size`; bench measures latency, it does not model it"
    )
    for field in _TIER_FIELDS:
        _unknown(tiers["theoretical"], field, roofline_reason)
    policy_reason = (
        "a recommended tier derates a measurement by a headroom factor, and that factor "
        "is a policy choice, not a measurement; bench does not invent policy"
    )
    for field in _TIER_FIELDS:
        _unknown(tiers["recommended"], field, policy_reason)
    # Null provenance on an empty tier is a C1 error with no lawful answer: the schema has
    # no provenance_u_reason companion for the sibling fix, and "U" is the enum member that
    # means exactly that this row states nothing. The tag is the statement; inventing a
    # reason string beside it would only restate the tag less clearly.
    tiers["theoretical"]["provenance"] = "U"
    tiers["recommended"]["provenance"] = "U"

    complete_rungs = [
        c
        for c in rung_list
        if result.rungs.get(c) is not None
        and result.rungs[c].outcome is ladder.RungOutcome.COMPLETE
    ]
    if complete_rungs:
        top = max(complete_rungs)
        _fill_tier(
            tiers["measured"],
            top,
            result.rungs[top],
            _median_repetition(_counted(repetitions[top])),
        )
        # A ladder that stopped on a failed rung observed which floor binds, and chapter 5
        # settles the label there. Leaving the constraint null beside this number is a C5
        # error the run itself could have answered.
        constraint = _boundary_constraint(result, top)
        if constraint is not None:
            _known(tiers["measured"], "binding_constraint", constraint)
    else:
        why = "no rung completed its declared repetitions"
        if censor:
            why += f"; the ladder was censored ({censor})"
        for field in _TIER_FIELDS:
            _unknown(tiers["measured"], field, why)
        tiers["measured"]["provenance"] = "U"

    top_sustainable = result.max_sustainable_concurrency
    sustainable_rung = result.rungs.get(top_sustainable) if top_sustainable is not None else None
    # sustainable_publishable is the ladder's own gate on this figure -- confirmed by a
    # post-search repetition, monotone, and not harness-limited. Publishing the boundary
    # without it turns "the highest rung we happened to try" into "the highest rung that
    # works", which is the most flattering single sentence a capacity report can contain.
    if not result.sustainable_publishable:
        sustainable_rung = None
    if sustainable_rung is not None and _counted(repetitions.get(top_sustainable, [])):
        _fill_tier(
            tiers["sustainable"],
            top_sustainable,
            sustainable_rung,
            _median_repetition(_counted(repetitions[top_sustainable])),
        )
        # Same boundary, same rule as the measured tier: the first failed rung above is the
        # observed floor, and silence there is a C5 error the run could answer.
        constraint = _boundary_constraint(result, top_sustainable)
        if constraint is not None:
            _known(tiers["sustainable"], "binding_constraint", constraint)
    else:
        if censor is not None:
            why = f"the ladder was censored before a sustainable tier emerged ({censor})"
        elif top_sustainable is not None and not result.sustainable_publishable:
            why = (
                f"concurrency {top_sustainable} passed its gates but the ladder does not "
                "permit publishing it as a boundary"
                + ("" if result.confirmed else "; no post-search repetition confirmed it")
                + ("" if result.monotone else "; the ladder was not monotone")
                + (f"; {result.cache_caveat}" if result.cache_caveat else "")
            )
        elif result.terminated_at is not None:
            why = f"the ladder terminated at concurrency {result.terminated_at}"
        else:
            why = "no rung passed its declared SLO gates"
        for field in _TIER_FIELDS:
            _unknown(
                tiers["sustainable"],
                field,
                f"the ladder produced no sustainable tier: {why}",
            )
        tiers["sustainable"]["provenance"] = "U"

    roofline = report["roofline_comparison"]
    _unknown(roofline, "decode_tok_s_theoretical", roofline_reason)
    _unknown(
        roofline,
        "decode_tok_s_measured",
        "bench measures workload throughput, not an isolated decode rate at the "
        "roofline's operating point; the two are not interchangeable",
    )
    _unknown(
        roofline,
        "roofline_efficiency",
        "the ratio of two figures this run does not have: no theoretical roofline and no "
        "isolated decode measurement",
    )
    _unknown(roofline, "prefill_ttft_s_theoretical", roofline_reason)
    _unknown(
        roofline,
        "prefill_ttft_s_measured",
        "the TTFT bench measured mixes queueing with prefill; the roofline needs an "
        "isolated prefill measurement this run did not make",
    )

    sizing = report["sizing_result"]
    _unknown(
        sizing,
        "gpus_required",
        "sizing needs the declared demand targets and a headroom policy; `ascep size` "
        "computes it from those, not from this ladder",
    )
    _unknown(
        sizing,
        "replica_topology",
        "a replica topology is an output of sizing, not of measurement",
    )
    _unknown(
        sizing,
        "binding_constraint",
        "sizing binds only once demand and the roofline are known; this run has neither",
    )
    _unknown(
        sizing,
        "utilization_at_target_pct",
        "a bench run has no declared demand target to compute utilization against",
    )
    sizing["provenance"] = "U"

    # One topology is not a scaling curve, and the row shape requires a numeric
    # scaling_efficiency, so a single row could only be published as a fabricated 1.0.
    report["scaling"] = []

    assumption_template = copy.deepcopy(report["unmeasured_assumptions"][0])

    def _assumption(field, impact, cost):
        entry = copy.deepcopy(assumption_template)
        entry["field"] = field
        entry["impact_if_wrong"] = impact
        entry["cost_to_measure"] = cost
        return entry

    report["unmeasured_assumptions"] = [
        _assumption(
            "roofline_comparison",
            "Without a roofline, a measured throughput far from the hardware's ceiling "
            "is indistinguishable from a healthy one, and the real bottleneck goes "
            "uninvestigated.",
            "Run `ascep size` against the declared hardware block; it needs bandwidth "
            "and FLOPs from the declaration, not another bench run.",
        ),
        _assumption(
            "sizing_result (gpus_required and the headroom policy)",
            "A capacity decision made against the measured tier with no headroom leaves "
            "nothing for demand peaks; one made against an assumed headroom is policy "
            "dressed up as data.",
            "Declare the demand targets and run `ascep size`; the policy input is a "
            "capacity-planning decision, not a measurement.",
        ),
        _assumption(
            "scaling table (one topology only)",
            "A single topology says nothing about scaling efficiency, and extrapolating "
            "it to a node count this run never measured can halve or double a GPU order.",
            "Repeat the ladder at each tensor-parallel and pipeline-parallel degree of "
            "interest and fill one row per topology.",
        ),
        _assumption(
            "run.tokenizer (local token count not taken)",
            "Server-reported usage goes unchecked, so an engine that miscounts tokens "
            "inflates every tokens-per-second figure in this report.",
            "Count prompts and outputs locally with the served model's own tokenizer, "
            "as chapter 4.7.1 requires; bench ships no tokenizer to do it.",
        ),
        _assumption(
            "capacity_tiers.*.binding_constraint (weights and KV floors not evaluated)",
            "The named constraint is the floor observed under the declared gates. A "
            "weights or KV floor lower than it would not have been seen, so a deployment "
            "whose true floor is KV at another context length is still sized against the "
            "wrong resource, and the shortfall appears only in production.",
            "Inspect engine-reported KV cache occupancy and memory headroom on the serving "
            "host during the run, and repeat the ladder at longer context lengths; latency "
            "at one context length cannot decide the weights or KV floors.",
        ),
        _assumption(
            "run.results[].gpu_util_pct and gpu_mem_util_pct (not observed)",
            "Without utilization, a rung that failed its gates because the GPU was "
            "saturated is indistinguishable from one that failed because the client, the "
            "network or another tenant was, and the ladder's boundary gets attributed to "
            "the wrong component.",
            "Sample the serving host during each window -- nvidia-smi, DCGM, or the "
            "engine's own metrics endpoint -- and align the series to the window "
            "timestamps in the bundle; a load generator cannot see it over HTTP.",
        ),
        _assumption(
            "run.engine_version (not observed)",
            "Throughput moves by tens of percent between engine releases, so a report "
            "without the version cannot be compared with another run of the same model on "
            "the same hardware -- which is most of what these reports get used for.",
            "Read it from the serving host or the container image and record it; "
            "`/v1/models` does not carry it, so bench cannot ask.",
        ),
    ]
    if thin:
        report["unmeasured_assumptions"].append(
            _assumption(
                "figures published below the section 4.3 advisory sample floor: " + ", ".join(thin),
                "These percentiles are computed from enough samples to report but not "
                "enough to be stable: a single straggler moves them, so a tier boundary "
                "that turns on one of them can flip between two runs of the same config.",
                "Lengthen the window or raise the repetition count until the tail "
                "percentile clears its advisory floor; the sample count, not the "
                "estimator, is the limit.",
            )
        )

    reproduction = report["reproduction"]
    for key in (
        "run_configs_path",
        "raw_records_path",
        "engine_logs_path",
        "environment_capture_path",
    ):
        _measured(
            reproduction,
            key,
            c8.get(key),
            "the reproduction bundle did not record this path",
        )
    # Not the generic reason above. A missing path means the bundle failed to write an
    # artifact; a missing digest usually means there was no container to take one of -- a
    # bare-metal or conda run -- and bench cannot tell the two apart from a null in the
    # config. So it asserts no cause and states the consequence instead, which is the part
    # a reader needs: with no digest, nothing in the bundle pins the software except the
    # environment capture beside it, and that capture describes the host the harness ran on.
    _measured(
        reproduction,
        "container_digest",
        c8.get("container_digest"),
        "no container digest was declared for this run, so the software these figures were "
        "produced by is pinned only by the environment capture in the same bundle",
    )
    _unknown(
        reproduction,
        "analysis_script_path",
        "bench writes records and a manifest but no analysis script; every figure in "
        "this report is produced by `ascep bench` itself from the bundled records",
    )
    return report
