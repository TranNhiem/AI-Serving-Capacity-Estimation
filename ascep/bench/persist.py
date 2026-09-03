"""The reproduction bundle: what a run leaves behind after the GPU time is gone.

Rule C8 caps a report at ``partial`` when the bundle is missing, and chapter 8 requires the
path table to resolve. This module covers the half those rules cannot reach: that the bundle
a harness writes actually contains the run it just executed, that it says which facts it
could not capture instead of omitting them, and that it refuses to quietly overwrite
evidence that cannot be regenerated.

The failure this module exists to prevent is the bundle that looks complete. A directory
with five files in it satisfies every path check in the protocol while containing a records
file from yesterday's run, and no reader can tell. The manifest of content digests the
bundle carries is what separates "here are the records" from "here are records".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from importlib import metadata as _metadata
from pathlib import Path, PurePosixPath
from typing import Callable

from ascep import ASCEP_VERSION
from ascep import __version__ as _ascep_version
from ascep.bench.driver import WindowRun
from ascep.bench.records import write_records

#: The artifacts a bundle contains, with the C8 key each one answers to. The digest
#: manifest pins the bytes of each, so a bundle that verifies is the bundle that ran.
_RECORDS_NAME = "records.jsonl"
_CONFIGS_NAME = "run_configs.json"
_ENVIRONMENT_NAME = "environment.json"
_MANIFEST_NAME = "manifest.json"
_ENGINE_LOG_NAME = "engine.log"

#: Above this the engine log is hashed where it lies instead of being copied in. The cap
#: exists because a server that has been up for a week writes a log a bundle has no business
#: duplicating, and it is generous because the common case is a log from the one server this
#: run started. A GB200 ladder of eighteen windows produced 437 KB; a thousandfold margin on
#: that still leaves the bundle smaller than its own records file.
ENGINE_LOG_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024

#: The serving stack: the distributions whose versions move the numbers, not an inventory.
#: Both sides of the measurement are here -- the engine that generates the tokens and the
#: client that times them, because httpx and its scheduler sit inside every TTFT and every
#: inter-token gap this harness reports. A framework missing from this list is a gap to
#: close by adding it, not a reason to capture everything: a full freeze is thousands of
#: lines that bury the eight versions a reader came for.
_SERVING_STACK_PACKAGES = (
    "vllm",
    "vllm-flash-attn",
    "sglang",
    "tensorrt-llm",
    "torch",
    "transformers",
    "tokenizers",
    "flash-attn",
    "flashinfer-python",
    "xformers",
    "triton",
    "numpy",
    "httpx",
    "anyio",
)


def _default_runner(argv: list[str]) -> str:
    """Run ``argv`` and return its stdout, for the machine facts only a command knows."""
    result = subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def capture_environment(*, runner: Callable[[list[str]], str] | None = None, **extra) -> dict:
    """Capture the machine facts a reproduction bundle needs, naming any it cannot read.

    This runs after the measured window with the records still only in memory, so it must
    never raise: an exception here loses the run to a diagnostic nicety. Every fact it
    cannot read becomes ``None`` plus a sibling ``<key>_u_reason`` string explaining why --
    an omitted key reads as "not applicable", a null-with-reason reads as "we looked and
    could not tell", and only the second is true when nvidia-smi is not on the path.

    The ``packages`` mapping pins the serving stack, because the only other place a
    framework version appears is a declaration somebody typed, and nothing else in the
    bundle corroborates it. Its ``packages_source`` sibling records that the versions
    describe the harness process: the benchmark client and the model server need not be one
    environment, and a reader who assumes they are will read a pin for software that never
    ran the model.

    The harness pins itself too, from the imported module rather than from dist-info. A
    bundle that records the whole serving stack and not the tool that drove it cannot answer
    the first question a reader of an old report has -- whether the numbers predate the fix
    to the code that computed them.

    Caller-supplied ``extra`` facts merge in and win: the caller knows things the process
    cannot see, such as the Slurm allocation, and silently dropping them would make the
    capture look authoritative while being less complete than the operator's own notes.
    """
    if runner is None:
        runner = _default_runner
    env: dict = {}

    def blame(key: str, exc: BaseException) -> None:
        env[key] = None
        env[f"{key}_u_reason"] = f"(U) could not be captured: {type(exc).__name__}: {exc}"

    try:
        # The CSV query, not the banner. The banner is laid out for a human and its
        # spacing has moved between driver generations, so a parser written against one
        # release records None on the next -- and a null driver version in a bundle is
        # indistinguishable from a machine that had no GPU.
        env["driver_version"] = (
            runner(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            .splitlines()[0]
            .strip()
            or None
        )
        if env["driver_version"] is None:
            # A successful probe that says nothing is still "looked and could not tell";
            # treating it as a value would record an empty string as a driver version.
            env["driver_version_u_reason"] = "(U) nvidia-smi ran but reported no driver version"
    except Exception as exc:  # noqa: BLE001 -- must never raise, whatever the probe did
        blame("driver_version", exc)

    try:
        env["gpu_model"] = (
            runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            .splitlines()[0]
            .strip()
            or None
        )
        if env["gpu_model"] is None:
            env["gpu_model_u_reason"] = "(U) nvidia-smi reported no GPU name"
    except Exception as exc:  # noqa: BLE001
        blame("gpu_model", exc)

    try:
        env["python_version"] = _platform.python_version()
        env["platform"] = _platform.platform()
        env["implementation"] = _platform.python_implementation()
    except Exception as exc:  # noqa: BLE001
        blame("python_version", exc)

    try:
        # Versions come from dist-info metadata, never by importing the package: importing
        # torch costs seconds and gigabytes to read back a string, and on a client-only host
        # the serving framework is not installed at all, so the import would fail on a
        # machine where the metadata read succeeds.
        packages = {}
        for name in _SERVING_STACK_PACKAGES:
            try:
                packages[name] = _metadata.version(name)
            except _metadata.PackageNotFoundError:
                # Absent from the mapping IS the finding: this distribution cannot have
                # served the run. A null would carry the (U) "we looked and could not
                # tell", when the truth is simpler and stronger -- it is not installed.
                continue
        env["packages"] = packages
        env["packages_source"] = (
            "importlib.metadata in the process running ascep bench. The benchmark client "
            "and the model server can be separate environments on separate hosts; when "
            "they are, these versions describe the harness and say nothing about what "
            "served the requests."
        )
    except Exception as exc:  # noqa: BLE001
        # Deliberately all-or-nothing. Anything reaching here is the metadata layer itself
        # failing, not one package being absent, so a partial mapping would be a capture
        # that looks complete and silently omits whatever the loop had not reached yet.
        env["packages"] = {}
        env["packages_u_reason"] = f"(U) could not be probed: {type(exc).__name__}: {exc}"

    # The two versions dist-info cannot supply here. `ascep` is routinely run from a checkout
    # on PYTHONPATH with nothing installed -- the way every cluster run in this repository was
    # done -- and importlib.metadata would then record the harness as absent, which is the one
    # package in the mapping that provably ran. They are separate keys because they answer
    # separate questions: the package version identifies the code that produced the numbers,
    # and 0.3.0 reduces ITL over a population 0.2.0 got wrong, while the protocol version
    # identifies the rules the report is graded against. Read from the imported module, so
    # they cannot disagree with what the process is executing.
    env["ascep_package_version"] = _ascep_version
    env["ascep_protocol_version"] = ASCEP_VERSION

    # Extra facts merge in last: an operator's "slurm_job_id" must not be lost because a
    # probe happened to know a same-named fact, and a same-named extra is the caller
    # correcting the probe.
    env.update(extra)
    return env


def _relpath(target: Path, relative_to: Path) -> str:
    """Render ``target`` relative to ``relative_to``, refusing anything else.

    An absolute value in the returned table resolves on the machine that produced it and
    nowhere else, and chapter 8 requires the path table to resolve from the report
    directory.
    """
    try:
        return target.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        # Engine logs habitually live in /var/log, so this is the first thing a real harness
        # hits. Say what to do about it: the stock message names two paths and no remedy.
        raise ValueError(
            f"{target} is outside the report directory {relative_to}, so no relative path "
            "can name it. Copy or symlink it under the report directory before publishing; "
            "an absolute path in the reproduction table resolves only on this machine."
        ) from None


def _manifest_key(target: Path, bundle_dir: Path) -> str:
    """Render ``target`` relative to the bundle, walking up if it lives outside.

    Unlike :func:`_relpath` this tolerates ``..``, because the engine log is covered by the
    manifest without being copied into the bundle.
    """
    return PurePosixPath(os.path.relpath(target.resolve(), bundle_dir.resolve())).as_posix()


def _sha256(path: Path) -> str:
    """Hash a file in place, streaming so a gigabyte engine log stays cheap to cover."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    bundle_dir,
    *,
    relative_to,
    runs: list[WindowRun],
    workload,
    engine_logs_path,
    container_digest: str | None,
    environment: dict | None = None,
    extra_files: Mapping[str, bytes] | None = None,
    overwrite: bool = False,
) -> dict:
    """Write the reproduction bundle for ``runs`` and return its C8 reproduction block.

    The engine log is copied into the bundle when it is small enough to copy, and passed
    through by path when it is not. Passing it through was the original behaviour and the
    original reasoning still holds for a big log -- it can be gigabytes, and a harness that
    duplicated one would fill the disk the run was just measured on -- but the reasoning
    does not survive contact with a live log. The file a running server is still writing to
    changes the moment it logs another line, so its manifest digest is stale before the
    operator has finished reading the report, and ``verify_bundle`` calls an intact bundle
    modified. Copying below ``ENGINE_LOG_SNAPSHOT_MAX_BYTES`` freezes the bytes that were
    there when the run ended, which is the thing C8 actually wants, and it also lets the log
    live in /var/log: a copy lands inside the bundle, so ``_relpath`` has nothing to refuse.
    Above the cap the old behaviour returns, and the manifest records how many bytes were
    hashed so that a log which has only grown can still be told from one that was replaced.

    ``engine_logs_path`` and ``container_digest`` have no defaults: neither is knowable
    from inside the harness, and a default would put a confident value in the one field
    whose whole job is to be checkable. A null digest is allowed -- not every run happens
    in a container -- and is left for the checker to downgrade.

    ``extra_files`` maps a bundle-relative name to the exact bytes to drop in beside the
    four this function writes itself -- the caller's own config file, typically, whose bytes
    must survive the round trip through whatever parsed it. They go through here rather than
    being written afterwards because the manifest is written once, and an artifact a caller
    added later is an artifact nobody hashed: it can be edited after publication without
    ``verify_bundle`` saying a word, which is the one thing a bundle exists to prevent.

    An existing bundle is refused unless ``overwrite`` is set. The GPU hours are already
    spent; a second run that lands in the same directory destroys the first run's only
    copy, and the mistake is discovered at analysis time.
    """
    if not runs:
        raise ValueError("a bundle needs at least one run: an empty one is evidence of nothing")
    bundle_dir = Path(bundle_dir)
    relative_to = Path(relative_to)
    engine_logs_path = Path(engine_logs_path)
    if not engine_logs_path.is_absolute():
        engine_logs_path = relative_to / engine_logs_path

    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"bundle directory {bundle_dir} already exists; pass overwrite=True to "
                "replace it, knowing the run it holds cannot be regenerated"
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    all_records = [record for run in runs for record in run.records]
    ids = [r.request_id for r in all_records]
    if len(set(ids)) != len(ids):
        # One records file holds every repetition, so an id must identify a request across
        # the whole bundle; two requests sharing one makes the file unsplittable.
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"request_id values collide across repetitions: {dupes}")

    records_file = bundle_dir / _RECORDS_NAME
    with records_file.open("w") as fp:
        write_records(all_records, fp)

    workload_manifest = workload.manifest()
    # Present only on a session replay, and taken from the manifest rather than from the
    # workload's type so a third-party plan gets the same treatment. The two counts below
    # would be a truthful 0/0 on a text run, but "no sessions" and "sessions, none of them
    # finished" are different facts and a reader scanning a bundle should not have to know
    # which one a zero means.
    replayed_sessions = "session_plan" in workload_manifest

    config = {
        "workload": workload_manifest,
        "windows": [
            {
                "t0": run.t0,
                "window_s": run.window_s,
                "drain_deadline_s": run.drain_deadline_s,
                "warmup_count": run.warmup_count,
                "warmup_s_actual": run.warmup_s_actual,
                # null means the fleet entered this window in lock-step. It has to ship,
                # because a reviewer holding the bundle cannot otherwise tell whether a
                # throughput plateau in the ladder is the server saturating or
                # floor(window_s / cycle) stepping down by one for every user at once.
                "dephase_s": run.dephase_s,
                "policy": {
                    key: getattr(run.policy, key)
                    for key in (
                        "concurrency",
                        "window_s",
                        "drain_deadline_s",
                        "think_time_s",
                        "warmup_requests",
                        "warmup_s",
                        "dephase",
                        "loop",
                        "repetition",
                    )
                },
                # The boundary ships as data, not as a dataclass dump: asdict would
                # recurse silently into anything the dataclass later grows.
                "boundary": asdict(run.boundary),
                # A window ends on the clock, so the sessions still running are cut off
                # mid-conversation and their later, longer turns never happen. Started
                # minus completed is the size of that truncation, and it is the number a
                # reader needs to judge whether the window was long enough for the
                # sessions being replayed: a window that completes none of them measured
                # only the cheap first turns of every one.
                **(
                    {
                        "sessions_started": run.sessions_started,
                        "sessions_completed": run.sessions_completed,
                    }
                    if replayed_sessions
                    else {}
                ),
            }
            for run in runs
        ],
    }
    configs_file = bundle_dir / _CONFIGS_NAME
    configs_file.write_text(json.dumps(config, indent=2) + "\n")

    # Probe first, then let the caller's facts win. Writing the caller's dict alone would
    # make an operator who recorded one thing the process cannot see -- the Slurm
    # allocation, whether the node was exclusive -- silently discard everything the
    # process could, and the bundle would be less complete for having been annotated.
    env = capture_environment(**(environment or {}))
    env_file = bundle_dir / _ENVIRONMENT_NAME
    env_file.write_text(json.dumps(env, indent=2, default=str) + "\n")

    # Manifest keys are relative to the bundle, not to the report. verify_bundle is handed
    # a bundle directory and nothing else -- it is the check a third party runs on a
    # downloaded archive -- so a key it cannot resolve from there reports every intact
    # bundle as missing a file. The pass-through engine log therefore appears as a path
    # that walks out of the bundle, which is exactly what it is.
    files = {
        _RECORDS_NAME: _sha256(records_file),
        _CONFIGS_NAME: _sha256(configs_file),
        _ENVIRONMENT_NAME: _sha256(env_file),
    }
    hashed_prefix_bytes: dict[str, int] = {}
    if engine_logs_path.stat().st_size <= ENGINE_LOG_SNAPSHOT_MAX_BYTES:
        engine_log_file = bundle_dir / _ENGINE_LOG_NAME
        shutil.copyfile(engine_logs_path, engine_log_file)
        files[_ENGINE_LOG_NAME] = _sha256(engine_log_file)
        engine_logs_published = engine_log_file
    else:
        engine_log_key = _manifest_key(engine_logs_path, bundle_dir)
        files[engine_log_key] = _sha256(engine_logs_path)
        # Recorded only for the pass-through log, because it is the only artifact a bundle
        # names that something outside the bundle is still writing to. Without the length
        # there is no way to ask the one question worth asking of a changed log -- did the
        # server append to it, or did someone put a different file there -- and every
        # long-lived server turns an intact bundle into a failed verification.
        hashed_prefix_bytes[engine_log_key] = engine_logs_path.stat().st_size
        engine_logs_published = engine_logs_path
    for name, payload in (extra_files or {}).items():
        if name in files or name == _MANIFEST_NAME:
            # Silently overwriting records.jsonl with a caller's bytes would leave a bundle
            # whose manifest agrees with itself and describes a run that never happened.
            raise ValueError(f"extra file {name!r} would replace an artifact the bundle owns")
        extra_file = bundle_dir / name
        extra_file.parent.mkdir(parents=True, exist_ok=True)
        extra_file.write_bytes(payload)
        files[name] = _sha256(extra_file)
    manifest: dict = {"sha256": files}
    if hashed_prefix_bytes:
        manifest["hashed_prefix_bytes"] = hashed_prefix_bytes
    (bundle_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "run_configs_path": _relpath(configs_file, relative_to),
        "raw_records_path": _relpath(records_file, relative_to),
        "engine_logs_path": _relpath(engine_logs_published, relative_to),
        "environment_capture_path": _relpath(env_file, relative_to),
        "container_digest": container_digest,
    }


def verify_bundle(bundle_dir) -> list[str]:
    """Recompute the bundle's manifest digests; return one message per problem found.

    An empty list means intact. A missing manifest cannot be graded at all, and an edited
    or removed file is reported by name, because "the bundle is wrong" sends a reviewer to
    re-hash every artifact when one would do.

    A file the manifest records a hashed length for -- only ever a pass-through engine log,
    written by a server that outlived the run -- passes when it has merely grown and its
    first recorded bytes still hash to the manifest value. Those bytes are the evidence; the
    ones after them are the server's later life. Calling that a problem would mean no bundle
    from a still-running server ever verifies, which trains a reader to ignore the check.
    A log that shrank, or whose recorded prefix changed, is still reported: that is a
    different file where the evidence used to be.
    """
    bundle_dir = Path(bundle_dir)
    manifest_file = bundle_dir / _MANIFEST_NAME
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_file.read_text())
        entries = manifest["sha256"]
        prefixes = manifest.get("hashed_prefix_bytes") or {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"bundle manifest {_MANIFEST_NAME} is missing or unreadable: {exc}"]
    for name, expected in sorted(entries.items()):
        target = bundle_dir / name
        if not target.exists():
            # The engine log is stored by path relative to the report root, not inside
            # the bundle; an absent one still means the evidence is gone.
            problems.append(f"file {name} listed in the manifest is missing from the bundle")
            continue
        actual = _sha256(target)
        if actual == expected:
            continue
        hashed = prefixes.get(name)
        if hashed is not None and target.stat().st_size > hashed:
            with target.open("rb") as fp:
                if hashlib.sha256(fp.read(hashed)).hexdigest() == expected:
                    continue
        problems.append(
            f"file {name} has been modified since the bundle was written "
            f"(manifest sha256 {expected}, now {actual})"
        )
    return problems
