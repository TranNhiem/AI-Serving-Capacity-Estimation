"""Shared scaffolding for the ``ascep bench`` acceptance modules.

The command has one contract and several subjects, so the tests for it live in several
modules -- the contract itself, the workload declarations, and what the draft publishes --
and every one of them needs the same complete config to subtract from, the same four
declaration documents, and the same offline stand-in for a server. That is what is here
and all that is here: no test, no assertion about ``ascep bench``, nothing a reader has to
come looking for. A helper that grew an assertion would be a test nobody runs by name.

Not a ``conftest.py`` and not fixtures, deliberately. These are plain functions taking
explicit arguments, and a caller reading ``_config(tmp_path, **{"workload.corpus": ...})``
can see the whole config it is about to write; the same thing as a fixture would be a
config assembled somewhere the test does not name.
"""

from __future__ import annotations

import json
import pathlib

from ascep.cli import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The four layer documents bench reads rather than invents, borrowed from the negative
#: corpus baseline because those blocks are known to validate standalone. Any valid pair of
#: hardware/serving documents would do; what these tests care about is that bench refuses to
#: run without them and copies them through unchanged.
DECLARED = json.loads((REPO_ROOT / "examples" / "negative" / "baseline.json").read_text())


def _config(tmp_path: pathlib.Path, **overrides) -> dict:
    """A fully declared config. Every test that checks a refusal removes one key from this.

    The window is a fraction of a second because these tests run a fake adapter at memory
    speed; the dry-run tests that talk about wall clock override it with a realistic one.
    """
    config = {
        "endpoint": {
            "base_url": "http://127.0.0.1:9",
            "model": "test-model",
            "timeout_s": 30.0,
        },
        "declarations": {
            "hardware": "hardware.json",
            "model": "model.json",
            "serving": "serving.json",
            "workload": "workload.json",
        },
        "workload": {
            "corpus": "synthetic",
            "input_tokens": 512,
            "output_tokens": 128,
            "ignore_eos": True,
            "cache_policy": "unique-prefix",
            "seed": 11,
            "think_time_s": 0.01,
            "run_label": "acceptance",
        },
        "window": {
            "window_s": 0.4,
            "drain_deadline_s": 0.2,
            "warmup_requests": 2,
        },
        "ladder": {
            "concurrency": [1, 2, 4],
            "repetitions": 3,
            "throughput_collapse_ratio": 0.5,
        },
        "slo_gates": {
            "ttft_p95_max_s": 2.0,
            "itl_p95_max_s": 0.15,
            "e2e_p95_max_s": 60.0,
            "error_rate_max_pct": 1.0,
            "declared_before_run": True,
        },
        "output": {
            "bundle_dir": str(tmp_path / "bundle"),
            "report_path": str(tmp_path / "report.json"),
            "engine_logs_path": str(tmp_path / "engine.log"),
            "container_digest": "sha256:" + "b" * 64,
        },
    }
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if value is _DROP:
            config[section].pop(key, None)
        else:
            config[section][key] = value
    return config


_DROP = object()


def _write(tmp_path: pathlib.Path, config: dict, **layer_overrides) -> str:
    """Write the config, the four declarations it points at, and a stand-in engine log."""
    for layer in ("hardware", "model", "serving", "workload"):
        document = layer_overrides.get(layer, DECLARED[layer])
        (tmp_path / f"{layer}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    (tmp_path / "engine.log").write_text("fixture engine log\n", encoding="utf-8")
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(path)


def _dry_run(tmp_path: pathlib.Path, **overrides):
    path = _write(tmp_path, _config(tmp_path, **overrides))
    return main(["bench", path, "--dry-run"])


def _report(tmp_path: pathlib.Path) -> dict:
    return json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def _run_offline(
    monkeypatch,
    interrupt_after_s: float | None = None,
    reported_input_tokens: int = 512,
    token_gap_s: float = 0.0005,
    ttft_by_rung: dict[int, float] | None = None,
):
    """Replace the adapter with one that answers instantly, so the suite needs no server.

    Patched at the adapter boundary rather than at the transport, because the point of these
    tests is the assembly -- config to ladder to bundle to report -- and a fake transport
    would drag HTTP framing into tests that are not about it.

    ``interrupt_after_s`` is a deadline rather than a request count on purpose: a count that
    lands mid-ladder on one machine lands before the first window completes on a slower one,
    and the test would then assert that an empty bundle is a bundle.

    ``token_gap_s`` widens the simulated inter-token gap so a test can cross a declared ITL
    gate; it moves timestamps only, never the order or the outcome of a request, so the
    default keeps every existing run exactly as green as it was.

    ``ttft_by_rung`` maps a rung's concurrency to the first-token delay every request in that
    rung pays. Without it the fake answers every rung at exactly the same speed, so no
    declared latency gate can fail one rung and pass another, and a test about which graded
    rungs the measured tier is drawn from would have no mixed ladder to draw from. The rung
    is read back out of ``request_id``, which the workload builds as
    ``{run_label}-c{concurrency}-r{repetition}-i{index}``; a single-window run carries no
    rung and falls through to the default, as does any rung the mapping omits.
    """
    import re
    import time

    import ascep.cli as cli
    from ascep.bench.records import Outcome, RequestRecord

    started = time.monotonic()
    rung_of = re.compile(r"-c(\d+)-r\d+-i\d+$")

    def _ttft(request_id: str) -> float:
        if not ttft_by_rung:
            return 0.001
        found = rung_of.search(request_id)
        if found is None:
            return 0.001
        return ttft_by_rung.get(int(found.group(1)), 0.001)

    class _Fake:
        name = "fake"

        def __init__(self, *a, **k):
            pass

        async def aclose(self):
            pass

        async def issue(self, spec, *, clock, sink=None):
            if interrupt_after_s is not None and time.monotonic() - started > interrupt_after_s:
                raise KeyboardInterrupt
            t = clock()
            ttft = _ttft(spec.request_id)
            return RequestRecord(
                request_id=spec.request_id,
                issued_ts=t,
                outcome=Outcome.OK,
                first_token_ts=t + ttft,
                token_ts=[t + ttft + token_gap_s * i for i in range(8)],
                end_ts=t + ttft + 0.004 + token_gap_s * 7,
                output_tokens=spec.max_tokens or 128,
                input_tokens=reported_input_tokens,
            )

    monkeypatch.setattr(cli, "_bench_adapter", lambda config: _Fake(), raising=False)


def assert_draft_validates(tmp_path, monkeypatch, config: dict) -> None:
    """Run the offline ladder on ``config`` and assert the draft it wrote validates.

    Shared rather than duplicated because two subjects need exactly this check -- the text
    run and the session replay -- and they are in different modules. It used to be one test
    that the other reached into: the replay case rebound ``_config`` in the module globals
    and called the schema test function directly, to reuse its validation rather than keep a
    weaker copy beside it. The reuse was the right instinct and the mechanism was what kept
    a 2,251-line file from being split, because a test that patches another test's module
    state cannot move away from it. One helper both call keeps the instinct and drops the
    mechanism.
    """
    from ascep.validation import validate

    path = _write(tmp_path, config)
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0, "the offline ladder did not complete"
    assert validate("capacity-report", _report(tmp_path)) == []
