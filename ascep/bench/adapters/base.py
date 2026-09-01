"""The adapter contract: the smallest surface a serving framework must implement.

An adapter owns exactly one job -- turn one ``RequestSpec`` into one ``RequestRecord``.
Everything the protocol measures is derived from records, so the adapter is where a harness
most easily corrupts the data: by starting the clock late, by throwing away refusals, or by
dying on a server error instead of recording it. This module exists so that the driver, not
the adapter, owns the clock, the window, and the retry policy (there is none -- a retried
request disappears from the error-rate denominator and double-counts throughput).

The driver passes ``clock`` (normally ``time.perf_counter``) into every call. Adapters MUST
NOT import timekeeping of their own: a test that injects a fake clock and asserts exact
timings is how we keep TTFT honest across adapters, and an adapter that reads its own clock
defeats that test silently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable

from ascep.bench.records import RequestRecord


@dataclass
class RequestSpec:
    """Everything an adapter needs to issue one request, and nothing it does not.

    Prompt content lives in ``messages`` and nowhere else: an adapter that also accepted a
    rendered prompt string would let two drivers tokenize the same request differently and
    produce usage numbers that cannot be reconciled (chapter 4 §7.1).
    """

    request_id: str
    messages: list[dict]
    #: None means "server default" -- deliberately not coerced, because a coerced default is
    #: invisible in the record and changes the workload between runs.
    max_tokens: int | None = None
    temperature: float | None = None
    #: Per-request deadline measured on the driver's clock, so a fake clock in tests moves
    #: the deadline too.
    deadline_s: float | None = None
    #: Escape hatch for framework-specific parameters; anything here lands in the payload
    #: verbatim and is the caller's responsibility.
    extra: dict = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """Connection-level settings shared by every request an adapter issues."""

    base_url: str
    model: str
    api_key: str | None = None
    #: Transport deadline for the whole response. Distinct from ``RequestSpec.deadline_s``:
    #: this guards against a wedged socket, that one defines the measured workload.
    timeout_s: float = 600.0
    headers: dict = field(default_factory=dict)


class Adapter(abc.ABC):
    """One method, ``issue``. If a framework cannot be expressed through it, the record
    contract -- not the driver -- is the thing to revisit, because every published number is
    a reduction over records and a richer adapter interface cannot fix a thin record."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier recorded in the run manifest, so records outlive the code."""

    @abc.abstractmethod
    async def issue(
        self,
        spec: RequestSpec,
        *,
        clock: Callable[[], float],
        sink: Callable[[RequestRecord], None] | None = None,
    ) -> RequestRecord:
        """Issue exactly one request and return its record.

        Never raises for a server-side failure: a raised exception is a request that silently
        never existed, and the chapter 7 §6 denominator counts requests that existed. The
        single exception is ``asyncio.CancelledError`` at window close, which must propagate
        after the partial record has been handed to ``sink`` -- see the concrete adapters for
        why an in-flight request may not be dropped.
        """
