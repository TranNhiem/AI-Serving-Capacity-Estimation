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
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Callable

from ascep.bench.driver import WindowRun
from ascep.bench.records import write_records

#: The artifacts a bundle contains, with the C8 key each one answers to. The digest
#: manifest pins the bytes of each, so a bundle that verifies is the bundle that ran.
_RECORDS_NAME = "records.jsonl"
_CONFIGS_NAME = "run_configs.json"
_ENVIRONMENT_NAME = "environment.json"
_MANIFEST_NAME = "manifest.json"


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
    overwrite: bool = False,
) -> dict:
    """Write the reproduction bundle for ``runs`` and return its C8 reproduction block.

    The engine log is passed through, not copied: it can be gigabytes, and a harness that
    duplicated it would fill the disk the run was just measured on. Its path is echoed
    relative to ``relative_to`` and its bytes are hashed in place into the manifest.

    ``engine_logs_path`` and ``container_digest`` have no defaults: neither is knowable
    from inside the harness, and a default would put a confident value in the one field
    whose whole job is to be checkable. A null digest is allowed -- not every run happens
    in a container -- and is left for the checker to downgrade.

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

    config = {
        "workload": workload.manifest(),
        "windows": [
            {
                "t0": run.t0,
                "window_s": run.window_s,
                "drain_deadline_s": run.drain_deadline_s,
                "warmup_count": run.warmup_count,
                "warmup_s_actual": run.warmup_s_actual,
                "policy": {
                    key: getattr(run.policy, key)
                    for key in (
                        "concurrency",
                        "window_s",
                        "drain_deadline_s",
                        "think_time_s",
                        "warmup_requests",
                        "warmup_s",
                        "loop",
                        "repetition",
                    )
                },
                # The boundary ships as data, not as a dataclass dump: asdict would
                # recurse silently into anything the dataclass later grows.
                "boundary": asdict(run.boundary),
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
        _manifest_key(engine_logs_path, bundle_dir): _sha256(engine_logs_path),
    }
    (bundle_dir / _MANIFEST_NAME).write_text(json.dumps({"sha256": files}, indent=2) + "\n")

    return {
        "run_configs_path": _relpath(configs_file, relative_to),
        "raw_records_path": _relpath(records_file, relative_to),
        "engine_logs_path": _relpath(engine_logs_path, relative_to),
        "environment_capture_path": _relpath(env_file, relative_to),
        "container_digest": container_digest,
    }


def verify_bundle(bundle_dir) -> list[str]:
    """Recompute the bundle's manifest digests; return one message per problem found.

    An empty list means intact. A missing manifest cannot be graded at all, and an edited
    or removed file is reported by name, because "the bundle is wrong" sends a reviewer to
    re-hash every artifact when one would do.
    """
    bundle_dir = Path(bundle_dir)
    manifest_file = bundle_dir / _MANIFEST_NAME
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_file.read_text())
        entries = manifest["sha256"]
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
        if actual != expected:
            problems.append(
                f"file {name} has been modified since the bundle was written "
                f"(manifest sha256 {expected}, now {actual})"
            )
    return problems
