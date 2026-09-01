"""The OpenAI-compatible adapter, tested against a scripted server.

Every test here drives a fake transport and a fake clock, so the assertions are on exact
timestamps rather than on tolerances. The adapter is the only place in the harness that
decides what counts as a token and what counts as a failure, and both decisions are easy to
get wrong in a way that produces a plausible number instead of an error.
"""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx", reason="the adapter lives behind the `run` extra")

from ascep.bench.adapters.base import AdapterConfig, RequestSpec  # noqa: E402
from ascep.bench.adapters.openai_compat import OpenAICompatAdapter  # noqa: E402
from ascep.bench.records import Outcome  # noqa: E402

pytestmark = pytest.mark.asyncio


class FakeClock:
    """A clock the test advances by hand, so timings are exact and not merely close."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def chunk(content=None, *, role=None, finish=None, usage=None):
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    body = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return body


def usage_only(usage) -> dict:
    """The final chunk emitted under stream_options.include_usage has no choices at all."""
    return {"choices": [], "usage": usage}


def _adapter(handler, **cfg):
    config = AdapterConfig(base_url="http://fake", model="m", **cfg)
    return OpenAICompatAdapter(config, transport=httpx.MockTransport(handler))


def _script(clock, events, *, status=200, gap=0.1):
    """A transport that advances the clock before delivering each SSE event."""

    def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            for ev in events:
                clock.advance(gap)
                yield ev

        clock.advance(0.05)  # connect
        return httpx.Response(status, content=body())

    return handler


SPEC = RequestSpec(request_id="r1", messages=[{"role": "user", "content": "hi"}])


# --- what counts as the first token ---------------------------------------------------


async def test_the_leading_role_only_chunk_is_not_the_first_token():
    """Every OpenAI-compatible server sends one, and counting it flatters TTFT.

    The role chunk lands at +0.15 and the first real content at +0.25. Charging TTFT from
    the role chunk would report 0.15 s for a request the user waited 0.25 s to see.
    """
    clock = FakeClock()
    events = [
        sse(chunk(role="assistant")),
        sse(chunk("Hello")),
        sse(chunk("!")),
        b"data: [DONE]\n\n",
    ]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.ttft_s == pytest.approx(0.25)


async def test_an_empty_content_delta_is_not_a_token_either():
    clock = FakeClock()
    events = [sse(chunk(role="assistant")), sse(chunk("")), sse(chunk("x")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.ttft_s == pytest.approx(0.35)
    assert rec.token_ts == [], "one content chunk means no inter-token gap exists"


async def test_connect_time_is_recorded_separately_but_still_billed_into_ttft():
    """The user waits through connect; we still need to see how much of it was ours."""
    clock = FakeClock()
    events = [sse(chunk("a")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.client_overhead_s == pytest.approx(0.05)
    assert rec.ttft_s == pytest.approx(0.15)


# --- chunks are not tokens ------------------------------------------------------------


async def test_a_batched_stream_declares_that_its_itl_is_per_chunk():
    """Three chunks carrying ten tokens means the ITL series is not inter-token latency.

    The gaps are larger and fewer than the real ones, so a p95 ITL gate would be evaluated
    against a quantity that is not ITL. Silence here is the failure; the run is still OK.
    """
    clock = FakeClock()
    events = [
        sse(chunk("aaa")),
        sse(chunk("bbb")),
        sse(chunk("cccc")),
        sse(usage_only({"prompt_tokens": 5, "completion_tokens": 10})),
        b"data: [DONE]\n\n",
    ]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.outcome is Outcome.OK
    assert rec.error == "itl-granularity: 3 chunks for 10 tokens"
    assert rec.output_tokens == 10


async def test_a_token_per_chunk_stream_carries_no_granularity_note():
    clock = FakeClock()
    events = [
        sse(chunk("a")),
        sse(chunk("b")),
        sse(chunk("c")),
        sse(usage_only({"prompt_tokens": 5, "completion_tokens": 3})),
        b"data: [DONE]\n\n",
    ]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.error is None
    assert rec.itls_s == pytest.approx([0.1, 0.1])


async def test_a_chunk_count_never_becomes_a_token_count():
    """When the server omits usage, the counts stay unknown rather than becoming chunks."""
    clock = FakeClock()
    events = [sse(chunk("a")), sse(chunk("b")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.output_tokens is None
    assert rec.input_tokens is None
    assert rec.output_tokens_local is None


# --- failure classification -----------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503])
async def test_admission_refusal_is_a_refusal_not_a_transport_error(status):
    """A shedding server is at capacity, which is the thing being measured."""
    clock = FakeClock()

    def handler(request):
        return httpx.Response(status, text="queue full")

    rec = await _adapter(handler).issue(SPEC, clock=clock)
    assert rec.outcome is Outcome.REFUSED
    assert rec.http_status == status


async def test_a_server_error_is_recorded_with_its_status_and_body():
    def handler(request):
        return httpx.Response(500, text="CUDA out of memory")

    rec = await _adapter(handler).issue(SPEC, clock=FakeClock())
    assert rec.outcome is Outcome.ERROR
    assert rec.http_status == 500
    assert "CUDA out of memory" in rec.error


async def test_a_transport_failure_returns_a_record_instead_of_raising():
    """A raised exception is a request that silently never existed, which moves the
    error-rate denominator without anyone noticing."""

    def handler(request):
        raise httpx.ConnectError("connection refused")

    rec = await _adapter(handler).issue(SPEC, clock=FakeClock())
    assert rec.outcome is Outcome.ERROR
    assert rec.request_id == "r1"


async def test_a_timeout_is_its_own_outcome():
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    rec = await _adapter(handler).issue(SPEC, clock=FakeClock())
    assert rec.outcome is Outcome.TIMEOUT


async def test_a_failed_request_is_never_retried():
    """A retry hides the failure from the error rate and double-counts the throughput."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="nope")

    await _adapter(handler).issue(SPEC, clock=FakeClock())
    assert len(calls) == 1


# --- cancellation at window close -----------------------------------------------------


async def test_cancelling_in_flight_hands_back_a_partial_record_and_still_propagates():
    """Discarding in-flight requests removes exactly the slowest ones from every
    percentile. The record goes to the sink; the CancelledError still unwinds the task."""
    clock = FakeClock()
    captured = []

    def handler(request):
        async def body():
            clock.advance(0.1)
            yield sse(chunk("a"))
            await asyncio.sleep(30)  # cancelled here

        return httpx.Response(200, content=body())

    adapter = _adapter(handler)
    task = asyncio.ensure_future(adapter.issue(SPEC, clock=clock, sink=captured.append))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (rec,) = captured
    assert rec.outcome is Outcome.CANCELLED
    assert rec.first_token_ts is not None, "the timestamps collected so far are kept"


# --- defensive SSE parsing ------------------------------------------------------------


async def test_a_usage_only_final_chunk_with_no_choices_is_tolerated():
    clock = FakeClock()
    events = [sse(chunk("a")), sse(usage_only({"prompt_tokens": 2, "completion_tokens": 1}))]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.outcome is Outcome.OK
    assert rec.input_tokens == 2


async def test_a_malformed_chunk_does_not_abort_the_request():
    """One bad line should cost one chunk, not the whole request and its record."""
    clock = FakeClock()
    events = [sse(chunk("a")), b"data: {not json\n\n", sse(chunk("b")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.outcome is Outcome.OK
    assert len(rec.token_ts) == 1


async def test_keepalive_comments_and_blank_lines_are_ignored():
    clock = FakeClock()
    events = [b": keepalive\n\n", b"\n", sse(chunk("a")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.outcome is Outcome.OK
    assert rec.first_token_ts is not None


async def test_the_finish_reason_is_preserved_because_length_truncation_changes_the_result():
    """A run where every response hit max_tokens is measuring a different workload."""
    clock = FakeClock()
    events = [sse(chunk("a")), sse(chunk("b", finish="length")), b"data: [DONE]\n\n"]
    rec = await _adapter(_script(clock, events)).issue(SPEC, clock=clock)
    assert rec.finish_reason == "length"
