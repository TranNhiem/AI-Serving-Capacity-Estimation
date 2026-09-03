"""The reproduction bundle: what a run leaves behind after the GPU time is gone.

Rule C8 caps a report at ``partial`` when the bundle is missing, and chapter 8 requires the
path table to resolve. Both are enforced elsewhere. What is enforced here is the half those
rules cannot reach: that the bundle a harness writes actually contains the run it just
executed, that it says which facts it could not capture instead of omitting them, and that
it refuses to quietly overwrite evidence that cannot be regenerated.

The failure this module exists to prevent is the bundle that looks complete. A directory with
five files in it satisfies every path check in the protocol while containing a records file
from yesterday's run, and no reader can tell.
"""

import importlib.metadata
import json
import pathlib

import pytest

from ascep import ASCEP_VERSION, __version__
from ascep.bench import persist
from ascep.bench.driver import Boundary, WindowPolicy, WindowRun
from ascep.bench.persist import capture_environment, verify_bundle, write_bundle
from ascep.bench.records import Outcome, RequestRecord, read_records
from ascep.conformance import check


def _record(rid, issued, rep):
    return RequestRecord(
        request_id=rid,
        issued_ts=issued,
        outcome=Outcome.OK,
        first_token_ts=issued + 0.1,
        end_ts=issued + 1.0,
        output_tokens=32,
        input_tokens=64,
        concurrency=4,
        repetition=rep,
    )


def _run(rep, n=5, t0=0.0):
    records = [_record(f"r{rep}-{i}", t0 + i * 0.1, rep) for i in range(n)]
    policy = WindowPolicy(
        concurrency=4,
        window_s=10.0,
        drain_deadline_s=5.0,
        think_time_s=0.25,
        warmup_requests=8,
        repetition=rep,
    )
    boundary = Boundary(
        t0=t0,
        window_s=10.0,
        drain_deadline_s=5.0,
        deadline=t0 + 15.0,
        offered=n,
        warmup=0,
        completed_in_window=n,
        straddlers=0,
        abandoned=0,
    )
    return WindowRun(
        records=records,
        policy=policy,
        t0=t0,
        window_s=10.0,
        drain_deadline_s=5.0,
        warmup_count=8,
        warmup_s_actual=12.4,
        boundary=boundary,
    )


class _Manifest:
    """The smallest thing that looks like a workload manifest to the bundle writer."""

    def __init__(self, **extra):
        self._m = {"seed": 3, "sampler": "uniform-with-replacement", "corpus_size": 8, **extra}

    def manifest(self):
        return dict(self._m)


def _write(tmp_path, runs=None, **kw):
    # Small, so the bundle snapshots it. A real engine log can be gigabytes, and the
    # pass-through path that handles those is exercised below with a lowered cap.
    (tmp_path / "engine.log").write_text("KV cache size: 574798 tokens\n")
    base = dict(
        runs=runs if runs is not None else [_run(0), _run(1, t0=30.0)],
        workload=_Manifest(),
        engine_logs_path="engine.log",
        container_digest="sha256:" + "a" * 64,
        environment={"python_version": "3.12.4"},
    )
    base.update(kw)
    return write_bundle(tmp_path / "bundle", relative_to=tmp_path, **base)


# --- C8: the bundle is what the report points at --------------------------------------


def test_the_returned_paths_are_exactly_what_rule_c8_looks_for(tmp_path):
    """Asserted through the checker rather than against a literal list of keys.

    A hand-copied list here would keep passing after C8 grew a sixth required artifact, and
    the harness would go on writing bundles the checker downgrades.
    """
    reproduction = _write(tmp_path)
    verdict = check({"ascep_version": "0.1.0", "reproduction": reproduction})
    assert not [f for f in verdict.findings if f.rule == "C8"], verdict.findings


def test_every_path_it_returns_resolves_from_the_report_directory(tmp_path):
    """Chapter 8: the path table MUST resolve. A path relative to the harness's working
    directory resolves on the machine that produced it and nowhere else."""
    reproduction = _write(tmp_path)
    for key, value in reproduction.items():
        if key.endswith("_path"):
            assert not pathlib.Path(value).is_absolute(), f"{key} is absolute"
            assert (tmp_path / value).exists(), f"{key} does not resolve: {value}"


def test_the_engine_log_and_container_digest_have_no_defaults(tmp_path):
    """Neither is knowable from inside the harness, and both pin the software the other
    artifacts were produced by. A default would put a confident value in the one field whose
    whole job is to be checkable."""
    with pytest.raises(TypeError):
        write_bundle(
            tmp_path / "b",
            relative_to=tmp_path,
            runs=[_run(0)],
            workload=_Manifest(),
            environment={},
        )


def test_a_null_digest_is_allowed_and_left_for_the_checker_to_downgrade(tmp_path):
    """Not every run happens in a container. The honest bundle says so and accepts partial;
    inventing a digest to clear C8 would be the exact failure C8 exists to catch."""
    reproduction = _write(tmp_path, container_digest=None)
    assert reproduction["container_digest"] is None
    verdict = check({"ascep_version": "0.1.0", "reproduction": reproduction})
    assert any(f.rule == "C8" for f in verdict.findings)


# --- the records really are this run's ------------------------------------------------


def test_the_records_file_round_trips_every_record_from_every_repetition(tmp_path):
    reproduction = _write(tmp_path)
    with (tmp_path / reproduction["raw_records_path"]).open() as fp:
        back = read_records(fp)
    assert len(back) == 10
    assert {r.repetition for r in back} == {0, 1}
    assert [r.request_id for r in back] == [f"r{rep}-{i}" for rep in (0, 1) for i in range(5)]


def test_a_bundle_with_colliding_request_ids_is_refused(tmp_path):
    """One records file holds every repetition, so an id has to identify a request across
    the whole bundle. Two requests sharing one makes the file unsplittable, and the reader
    who tries gets a repetition with a request missing and no error to explain it."""
    with pytest.raises(ValueError, match="request_id"):
        _write(tmp_path, runs=[_run(0), _run(0, t0=30.0)])


def test_a_bundle_with_no_runs_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        _write(tmp_path, runs=[])


# --- the run config is what executed, not what was requested --------------------------


def test_the_run_config_records_the_declared_operating_point_of_each_repetition(tmp_path):
    reproduction = _write(tmp_path)
    config = json.loads((tmp_path / reproduction["run_configs_path"]).read_text())
    assert len(config["windows"]) == 2
    first = config["windows"][0]
    for key in ("concurrency", "window_s", "drain_deadline_s", "think_time_s", "repetition"):
        assert key in first["policy"], f"the config cannot be re-run without {key}"
    assert first["policy"]["drain_deadline_s"] == 5.0
    assert first["policy"]["think_time_s"] == 0.25


def test_the_run_config_records_the_warm_up_that_happened_not_the_one_requested(tmp_path):
    """Section 7.3 requires the number and duration to be declared. A run whose warm-up was
    cut short by a slow server is a different run, and only the actuals show it."""
    config = json.loads((tmp_path / _write(tmp_path)["run_configs_path"]).read_text())
    first = config["windows"][0]
    assert first["warmup_count"] == 8
    assert first["warmup_s_actual"] == pytest.approx(12.4)


def test_the_run_config_records_where_the_boundary_fell(tmp_path):
    """The cohort counts are the harness's ruling on section 7.6. Shipping them lets a
    reader re-run apply_boundary_rules over the records and compare, which is the only way
    to check a boundary rule from outside."""
    config = json.loads((tmp_path / _write(tmp_path)["run_configs_path"]).read_text())
    boundary = config["windows"][0]["boundary"]
    for key in ("t0", "deadline", "offered", "warmup", "completed_in_window", "straddlers"):
        assert key in boundary


def test_the_workload_manifest_travels_with_the_run_config(tmp_path):
    """The records say what the server was asked; only the manifest says why those prompts.
    Split across two artifacts, the pair goes missing one file at a time."""
    config = json.loads((tmp_path / _write(tmp_path)["run_configs_path"]).read_text())
    assert config["workload"]["seed"] == 3
    assert config["workload"]["sampler"] == "uniform-with-replacement"


# --- the environment capture names what it could not read -----------------------------


def test_an_unreadable_environment_fact_is_null_with_a_reason_not_absent():
    """Rule C1 in the one place nobody applies it: a machine-written file.

    An omitted key reads as "not applicable"; a null with a reason reads as "we looked and
    could not tell", and only the second is true when nvidia-smi is not on the path.
    """

    def runner(argv):
        raise FileNotFoundError(argv[0])

    env = capture_environment(runner=runner)
    assert env["driver_version"] is None
    assert "driver_version_u_reason" in env
    assert env["driver_version_u_reason"]


def test_the_environment_capture_never_raises_on_a_hostile_probe():
    """It runs after the measured window, when the records are already in memory and not yet
    on disk. An exception here loses the run to a diagnostic nicety."""

    def runner(argv):
        raise RuntimeError("boom")

    env = capture_environment(runner=runner)
    assert env["driver_version"] is None


def test_the_environment_capture_reports_what_it_could_read():
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert env["driver_version"] == "555.42.06"
    assert "driver_version_u_reason" not in env
    assert env["python_version"]
    assert env["platform"]


def test_the_environment_capture_pins_the_harness_that_produced_the_numbers():
    """The bundle pins fourteen packages of serving stack and, until this, not the tool that
    drove them. Every cluster run in this repository is launched from a checkout on
    PYTHONPATH with nothing installed, so importlib.metadata would report the harness absent
    -- the one distribution that provably ran. Without both numbers a reader of an old report
    cannot tell whether its ITL figures predate the release that fixed how ITL is reduced."""
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert env["ascep_package_version"] == __version__
    assert env["ascep_protocol_version"] == ASCEP_VERSION


def test_caller_supplied_environment_facts_are_merged_and_win(tmp_path):
    """The caller knows things the process cannot see -- the Slurm allocation, whether the
    node was exclusive. Silently dropping them would make the capture look authoritative
    while being less complete than the operator's own notes."""
    reproduction = _write(tmp_path, environment={"node_exclusivity": "exclusive"})
    env = json.loads((tmp_path / reproduction["environment_capture_path"]).read_text())
    assert env["node_exclusivity"] == "exclusive"
    assert env["python_version"]


# --- evidence is not overwritten, and tampering is detectable -------------------------


def test_writing_over_an_existing_bundle_is_refused(tmp_path):
    """The GPU hours are already spent. A second run that lands in the same directory
    destroys the first run's only copy, and the mistake is discovered at analysis time."""
    _write(tmp_path)
    with pytest.raises(FileExistsError):
        _write(tmp_path)


def test_overwriting_is_possible_but_has_to_be_asked_for(tmp_path):
    _write(tmp_path)
    reproduction = _write(tmp_path, overwrite=True)
    assert (tmp_path / reproduction["raw_records_path"]).exists()


def test_a_fresh_bundle_verifies(tmp_path):
    _write(tmp_path)
    assert verify_bundle(tmp_path / "bundle") == []


def test_an_edited_records_file_fails_verification(tmp_path):
    """The bundle's own digests are what separate "here are the records" from "here are
    records". Without them a hand-fixed outlier is undetectable by anyone downstream."""
    reproduction = _write(tmp_path)
    target = tmp_path / reproduction["raw_records_path"]
    # Appending a forged record rather than rewriting a field: a tamper test that matches on
    # the serialiser's whitespace passes for free the day the serialiser stops emitting it,
    # and reports the digest check as working when it was never exercised.
    with target.open("a") as fp:
        fp.write(json.dumps({"request_id": "forged", "outcome": "ok"}) + "\n")
    assert "records.jsonl" in " ".join(verify_bundle(tmp_path / "bundle"))


def test_a_file_removed_from_the_bundle_fails_verification(tmp_path):
    reproduction = _write(tmp_path)
    (tmp_path / reproduction["run_configs_path"]).unlink()
    assert verify_bundle(tmp_path / "bundle")


def test_an_extra_file_arrives_byte_for_byte(tmp_path):
    """The caller's config goes in as bytes, not as something re-serialised on the way past.
    A config round-tripped through a parser records what the harness understood, and the one
    difference worth catching is the harness understanding something else."""
    payload = b'{"window_s": 20.0,\n  "note": "trailing whitespace and all"   }\n'
    _write(tmp_path, extra_files={"bench-config.json": payload})
    assert (tmp_path / "bundle" / "bench-config.json").read_bytes() == payload


def test_an_edited_extra_file_fails_verification(tmp_path):
    """An artifact the manifest does not hash is one anyone can edit after publication
    without verify_bundle saying a word, and the config is the artifact where a single
    changed number rewrites what the run claims to have been."""
    _write(tmp_path, extra_files={"bench-config.json": b"{}\n"})
    (tmp_path / "bundle" / "bench-config.json").write_bytes(b'{"tampered": true}\n')
    assert "bench-config.json" in " ".join(verify_bundle(tmp_path / "bundle"))


def test_an_extra_file_may_not_take_the_name_of_an_artifact_the_bundle_owns(tmp_path):
    """Overwriting records.jsonl with the caller's bytes leaves a bundle whose manifest
    agrees with itself and describes a run that never happened -- the one failure a digest
    manifest cannot report, because it would be hashing the substitute."""
    with pytest.raises(ValueError):
        _write(tmp_path, extra_files={"records.jsonl": b"not the records\n"})


# --- the engine log: frozen when it is small, tolerated when it is still growing --------


def _manifest(tmp_path) -> dict:
    return json.loads((tmp_path / "bundle" / "manifest.json").read_text())


def test_a_small_engine_log_is_copied_into_the_bundle_rather_than_hashed_in_place(tmp_path):
    """Hashing it in place was the original behaviour and it cannot survive a live server.

    The engine writes another line the moment the bundle is closed, so the manifest digest is
    stale before the operator finishes reading the report and verify_bundle calls an intact
    bundle modified. A copy is the frozen evidence C8 is asking for, and it is the whole file
    -- an archive somebody downloads has to carry the log, not a path to one.
    """
    reproduction = _write(tmp_path)
    snapshot = tmp_path / "bundle" / "engine.log"
    assert snapshot.read_text() == (tmp_path / "engine.log").read_text()
    assert reproduction["engine_logs_path"] == "bundle/engine.log"
    assert "engine.log" in _manifest(tmp_path)["sha256"]
    assert "hashed_prefix_bytes" not in _manifest(tmp_path)


def test_a_snapshotted_log_verifies_after_the_server_writes_more_of_it(tmp_path):
    """The regression this pins is the one that failed a real GB200 bundle: the run ended,
    the server stayed up, and the next request it served invalidated the manifest."""
    _write(tmp_path)
    with (tmp_path / "engine.log").open("a") as fp:
        fp.write("Avg prompt throughput: 0.0 tokens/s\n")
    assert verify_bundle(tmp_path / "bundle") == []


def test_a_snapshotted_log_that_lives_outside_the_report_directory_is_still_publishable(
    tmp_path,
):
    """Engine logs habitually live in /var/log or /tmp, which _relpath has to refuse because
    an absolute path in the reproduction table resolves on one machine. Copying moots the
    refusal: what the table names is the copy, and the copy is inside the bundle."""
    elsewhere = tmp_path / "var"
    elsewhere.mkdir()
    (elsewhere / "vllm.out").write_text("Available KV cache memory: 104.88 GiB\n")
    reproduction = _write(tmp_path, engine_logs_path=elsewhere / "vllm.out")
    assert reproduction["engine_logs_path"] == "bundle/engine.log"
    assert (tmp_path / "bundle" / "engine.log").read_text().startswith("Available KV")


def test_an_engine_log_above_the_cap_is_hashed_where_it_lies_with_its_length(tmp_path, monkeypatch):
    """A server up for a week writes a log the bundle has no business duplicating. Above the
    cap the manifest key walks out of the bundle, which is what a pass-through log is, and
    the recorded length is what lets a grown log be told from a substituted one."""
    monkeypatch.setattr(persist, "ENGINE_LOG_SNAPSHOT_MAX_BYTES", 8)
    reproduction = _write(tmp_path)
    assert reproduction["engine_logs_path"] == "engine.log"
    assert not (tmp_path / "bundle" / "engine.log").exists()
    prefixes = _manifest(tmp_path)["hashed_prefix_bytes"]
    assert prefixes == {"../engine.log": (tmp_path / "engine.log").stat().st_size}


def test_verify_accepts_a_pass_through_log_the_server_only_appended_to(tmp_path, monkeypatch):
    """The bytes the manifest hashed are the evidence; the ones after them are the server's
    later life. Reporting that as tampering means no bundle from a still-running server ever
    verifies, which teaches a reader to ignore the check."""
    monkeypatch.setattr(persist, "ENGINE_LOG_SNAPSHOT_MAX_BYTES", 8)
    _write(tmp_path)
    with (tmp_path / "engine.log").open("a") as fp:
        fp.write("Engine iteration timed out\n" * 200)
    assert verify_bundle(tmp_path / "bundle") == []


def test_verify_refuses_a_pass_through_log_whose_recorded_prefix_changed(tmp_path, monkeypatch):
    """A longer file is not automatically an appended-to one. A log rotated and replaced by a
    different server's output is longer too, and the prefix check is the only thing between
    that and a bundle that verifies against evidence it never saw."""
    monkeypatch.setattr(persist, "ENGINE_LOG_SNAPSHOT_MAX_BYTES", 8)
    _write(tmp_path)
    (tmp_path / "engine.log").write_text("a different server said something else entirely\n")
    assert "engine.log" in " ".join(verify_bundle(tmp_path / "bundle"))


def test_verify_refuses_a_pass_through_log_that_was_truncated(tmp_path, monkeypatch):
    """Shorter than the length recorded means the evidence has been thrown away, and the
    prefix tolerance must not reach it: hashing fewer bytes than were hashed cannot match."""
    monkeypatch.setattr(persist, "ENGINE_LOG_SNAPSHOT_MAX_BYTES", 8)
    _write(tmp_path)
    (tmp_path / "engine.log").write_text("KV cache size:")
    assert "engine.log" in " ".join(verify_bundle(tmp_path / "bundle"))


# --- the packages block pins the serving stack, not just the platform -------------------


def _fake_version(name):
    # A hermetic stand-in in which only torch exists. Probing the real environment would
    # pass on a CI image that happens to have the packages and never exercise the absent
    # branch, which is the one the null-versus-absent rule below turns on.
    if name == "torch":
        return "2.3.1+cu121"
    raise importlib.metadata.PackageNotFoundError(name)


def test_installed_distributions_appear_in_packages_with_their_versions(monkeypatch):
    """What this replaces was 212 bytes naming no library version at all.

    "CPython 3.10.13 on Linux" pins nothing. The framework version that did exist lived in
    the serving declaration, typed by an operator, with nothing in the bundle corroborating
    it -- so a reproduction bundle whose whole purpose is to pin the software pinned none.
    """
    monkeypatch.setattr(importlib.metadata, "version", _fake_version)
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert env["packages"]["torch"] == "2.3.1+cu121"


def test_a_distribution_that_is_not_installed_is_absent_not_null(monkeypatch):
    """Absent means absent. Everywhere else in this module a null carries a (U) -- "we
    looked and could not tell" -- and a package that is simply not installed must not
    acquire one, because that reads as a gap in the capture rather than a fact about the
    machine."""
    monkeypatch.setattr(importlib.metadata, "version", _fake_version)
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert "vllm" not in env["packages"]
    assert env["packages"] == {"torch": "2.3.1+cu121"}


def test_a_package_probe_that_raises_does_not_propagate_and_leaves_a_u_reason(monkeypatch):
    """This runs after the measured window with the records still only in memory, so a
    raising probe must downgrade the packages block to a (U) rather than lose the run to a
    diagnostic nicety."""

    def boom(name):
        raise RuntimeError("metadata database corrupt")

    monkeypatch.setattr(importlib.metadata, "version", boom)
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert env["packages"] == {}
    assert env["packages_u_reason"].startswith("(U) ")


def test_the_client_side_of_the_measurement_is_pinned_too(monkeypatch):
    """httpx and its scheduler sit inside every TTFT and every inter-token gap this harness
    reports, so a capture that pins only the engine pins half the measurement."""
    seen = []

    def record(name):
        seen.append(name)
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", record)
    capture_environment(runner=lambda argv: "555.42.06\n")
    assert {"httpx", "anyio"} <= set(seen)


def test_packages_source_says_which_process_the_versions_describe():
    """The client and the server need not be one environment. A capture that does not say
    which side it saw will be trusted for both, and it only ever saw one."""
    env = capture_environment(runner=lambda argv: "555.42.06\n")
    assert "ascep bench" in env["packages_source"]


def test_a_caller_supplied_packages_fact_still_wins_over_the_probe(monkeypatch):
    """Extras merge last: an operator pinning versions the process cannot see -- the
    server's, when it runs elsewhere -- is correcting the capture, not colliding with it."""
    monkeypatch.setattr(importlib.metadata, "version", _fake_version)
    env = capture_environment(
        runner=lambda argv: "555.42.06\n", packages={"vllm": "0.11.0 (server host)"}
    )
    assert env["packages"] == {"vllm": "0.11.0 (server host)"}
