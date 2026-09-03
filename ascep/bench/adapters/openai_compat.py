"""Adapter for servers speaking the OpenAI streaming chat-completions API.

Covers vLLM, SGLang, TGI (with OpenAI routing), llama.cpp's server, and hosted endpoints.

On chunks versus tokens
-----------------------
These servers stream *chunks*, and a chunk is a transport batching decision, not a token. A
server under load may coalesce several tokens into one SSE event. If the resulting inter-chunk
gaps are reported as inter-token latency without declaring the substitution, the series has
fewer, larger gaps and the p95 ITL gate is being evaluated against a quantity that is not ITL
-- the metric has been silently redefined while keeping its name. This adapter therefore
records ``itl-granularity: ...`` in ``RequestRecord.error`` whenever the server's own usage
accounting proves the two counts diverged, and never writes a chunk count into
``output_tokens_local``, because a chunk count laundered into a tokenizer count is exactly the
kind of invisible substitution the reconciliation check (chapter 4 §7.1) exists to catch.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import httpx

from ascep.bench.adapters.base import Adapter, AdapterConfig, RequestSpec
from ascep.bench.records import Outcome, RequestRecord


class OpenAICompatAdapter(Adapter):
    def __init__(
        self,
        config: AdapterConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        headers = dict(config.headers)
        if config.api_key is not None:
            headers.setdefault("Authorization", f"Bearer {config.api_key}")
        # `transport` exists so the tests can script a server -- an exact byte stream against
        # an injected clock -- rather than assert tolerances against a real one. Timing rules
        # like "the role-only chunk is not the first token" are only checkable to the
        # microsecond, and a test that has to allow slop would pass on the bug it exists for.
        # The driver is the only thing allowed to decide how many requests are in flight.
        # httpx defaults to max_connections=100, and a default that silently binds before the
        # declared concurrency does is the worst kind of measurement bug: every rung above 100
        # offers exactly 100, so throughput flatlines, the knee lands on whatever rung first
        # crosses 100, and the report attributes a property of the HTTP client to the server.
        # Requests past the cap also queue inside the pool after issued_ts is stamped, so the
        # client's own queue is billed to the server as TTFT. None means the transport never
        # throttles and the closed loop in the driver is the sole limiter.
        self._client = httpx.AsyncClient(
            timeout=config.timeout_s,
            headers=headers,
            transport=transport,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self._url = f"{config.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def name(self) -> str:
        return f"openai-compat/{self._config.model}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def issue(
        self,
        spec: RequestSpec,
        *,
        clock: Callable[[], float],
        sink: Callable[[RequestRecord], None] | None = None,
    ) -> RequestRecord:
        record = RequestRecord(request_id=spec.request_id, issued_ts=clock())
        payload: dict = {
            "model": self._config.model,
            "messages": spec.messages,
            "stream": True,
            # Without include_usage the server never reports token counts, and substituting a
            # chunk count for them is forbidden by the record contract.
            "stream_options": {"include_usage": True},
        }
        if spec.max_tokens is not None:
            payload["max_tokens"] = spec.max_tokens
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        payload.update(spec.extra)

        chunk_count = 0
        malformed = 0
        completion_tokens: int | None = None

        # No retry loop exists here or anywhere above: a retried request removes a failure
        # from the chapter 7 section 6 error-rate denominator and double-counts throughput,
        # which is precisely how overload gets reported as health.
        try:
            async with self._client.stream("POST", self._url, json=payload) as response:
                # Headers have arrived: everything before this line is client-side connect
                # cost and must stay out of what we bill the server for.
                record.connect_ts = clock()
                record.http_status = response.status_code

                if response.status_code in (429, 503):
                    # Admission refusal is a capacity failure the user experienced, not a
                    # transport bug; miscounting it as ERROR would bury a shedding server.
                    record.outcome = Outcome.REFUSED
                    await response.aread()
                elif not 200 <= response.status_code < 300:
                    record.outcome = Outcome.ERROR
                    body = await response.aread()
                    record.error = body.decode("utf-8", errors="replace")[:200]
                else:
                    async for line in response.aiter_lines():
                        now = clock()
                        if spec.deadline_s is not None and now - record.issued_ts > spec.deadline_s:
                            # Accepted work the server could not finish in time: TIMEOUT, not
                            # CANCELLED, because the deadline belongs to the workload.
                            # Checked per arriving line, so it catches a slow stream but not
                            # a wedged one -- that is what AdapterConfig.timeout_s is for, and
                            # the two must both be set or a stalled socket hangs the rung.
                            record.outcome = Outcome.TIMEOUT
                            record.error = f"deadline: exceeded {spec.deadline_s}s mid-stream"
                            break
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            # Count, don't abort: dropping the request over one bad line
                            # would remove a slow in-flight request from the distribution.
                            malformed += 1
                            continue

                        usage = chunk.get("usage")
                        if usage:
                            # The ONLY source of token counts. Absent usage, the fields stay
                            # None -- never the chunk count.
                            record.input_tokens = usage.get("prompt_tokens")
                            record.output_tokens = usage.get("completion_tokens")
                            completion_tokens = record.output_tokens

                        choices = chunk.get("choices") or []
                        if not choices:
                            # The usage-only final chunk arrives with an empty choices list.
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason") is not None:
                            record.finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if not content:
                            # A role-only or empty-content delta is protocol framing, not a
                            # token; timestamping it here would make TTFT read better than
                            # anything the user experienced.
                            continue
                        if record.first_token_ts is None:
                            record.first_token_ts = now
                        else:
                            record.token_ts.append(now)
                        chunk_count += 1
        except httpx.TimeoutException:
            record.outcome = Outcome.TIMEOUT
        except asyncio.CancelledError:
            # Window close. The requests still in flight are the slowest ones; writing them
            # out before re-raising keeps every percentile honest.
            record.outcome = Outcome.CANCELLED
            record.error = "cancelled: in flight at window close"
            record.end_ts = clock()
            if sink is not None:
                sink(record)
            raise
        except Exception as exc:
            record.outcome = Outcome.ERROR
            record.error = repr(exc)[:200]

        record.end_ts = clock()

        # Notes accumulate rather than overwrite. Both of these describe a way this record's
        # timing series is less trustworthy than it looks, and keeping only the last one
        # written would hide the other from exactly the person auditing the run.
        notes = []
        if record.outcome is Outcome.OK and record.error is not None:
            notes.append(record.error)
        if malformed:
            notes.append(f"sse: skipped {malformed} malformed chunk(s)")
        if (
            record.outcome is Outcome.OK
            and completion_tokens is not None
            and completion_tokens != chunk_count
        ):
            # The server batched tokens per chunk: token_ts measures per-chunk delivery, and
            # an ITL gate reduced from it would pass judgement on the wrong quantity unless
            # the substitution is declared on the record itself.
            notes.append(f"itl-granularity: {chunk_count} chunks for {completion_tokens} tokens")
        if notes and record.outcome is Outcome.OK:
            record.error = "; ".join(notes)

        return record
