"""The per-request record: the only thing a conforming harness must retain.

Every figure ASCEP publishes is a reduction over these records, so the record is the
protocol's real contract with a benchmark harness. A summary statistic cannot be re-derived,
re-percentiled, or audited after the fact; a record can. Chapter 7 §8 therefore requires the
raw records to be persisted and cited in the reproduction bundle, not just the summary.

Four fields here exist because the private campaign that produced the first published report
got them wrong, and the report had to declare the results ``(U) not measured`` as a result:

``token_ts``      the campaign stored one ``tpot = (e2e - ttft) / (out_tokens - 1)`` per
                  request and took percentiles of *that*. A p95 over per-request means is not
                  the p95 inter-token latency: it cannot see a stall inside a request, which
                  is the exact event an ITL gate exists to catch. Keeping the timestamps costs
                  a few hundred floats per request and makes ITL a real distribution.
``connect_ts``    the campaign started its clock before the HTTP connection was established,
                  so client-side connect time was billed to the server as TTFT. Recording the
                  moment the request actually went out lets the reduction report both and
                  makes client overhead visible instead of silently inflating the server.
``admitted``      requests refused at admission were dropped from the record list entirely,
                  so a server shedding a third of its offered load scored a 0% error rate.
                  Chapter 7 §6 makes *issued* the normative denominator; that is only
                  computable if refusals are recorded rather than discarded.
``outcome``       an in-flight request cancelled at window close was discarded too, which
                  removes exactly the slowest requests and biases every percentile downward.
                  It is recorded with its own outcome instead, and the reduction decides.

Nothing in this module talks to a network or a GPU: it is a data contract, so that a harness
for a framework we have never seen can satisfy it without adopting our client.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import IO


class Outcome(str, Enum):
    """Terminal state of an issued request.

    Every issued request has exactly one, and all of them except ``OK`` count as failures in
    the chapter 7 §6 error rate. They are distinguished because the *cause* changes what the
    operator should do: refusal means the admission queue is full, timeout means the server
    accepted work it could not finish, and cancellation means our own window closed first.
    """

    OK = "ok"  # completed, response fully received
    REFUSED = "refused"  # rejected at admission (HTTP 429/503, queue full)
    ERROR = "error"  # transport failure or non-2xx that is not a refusal
    TIMEOUT = "timeout"  # exceeded the per-request deadline
    CANCELLED = "cancelled"  # still in flight when the measurement window closed


@dataclass
class RequestRecord:
    """One issued request, from the client's point of view.

    All timestamps are seconds from a single monotonic origin shared by the whole run
    (``time.perf_counter()``), never wall clock: wall clock is not monotonic and an NTP step
    mid-window silently produces negative latencies. The origin is recorded once per run in
    the run manifest so records from different processes can be aligned.
    """

    request_id: str
    #: When the driver decided to send. The error-rate denominator counts records, so a
    #: record must exist from this moment on -- including for a request that is refused.
    issued_ts: float
    outcome: Outcome = Outcome.OK

    #: When the request bytes were actually on the wire. ``issued_ts -> connect_ts`` is
    #: client-side cost and is never charged to the server.
    connect_ts: float | None = None
    #: Arrival of the first *content* token. A leading empty delta or a role-only chunk is
    #: not a token; counting it makes TTFT look better than the user's experience.
    first_token_ts: float | None = None
    #: Arrival of every subsequent content token, in order. ITL is the diff sequence.
    token_ts: list[float] = field(default_factory=list)
    #: Completion of the response body.
    end_ts: float | None = None

    #: Token counts as reported by the server's usage accounting. ``None`` when the server
    #: did not return usage -- never silently substituted with a chunk count, because a
    #: chunk is not a token and the substitution is invisible in the summary.
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Locally counted output tokens, when a tokenizer is available. Chapter 4 §7.1 asks for
    #: the reconciliation check; it can only run if both numbers are kept.
    output_tokens_local: int | None = None

    #: The operating point this record belongs to, so records from a whole ladder can live
    #: in one file and still be reduced per rung.
    concurrency: int | None = None
    repetition: int | None = None
    #: False for warm-up traffic. Warm-up records are retained, not deleted: an excluded
    #: record that is still on disk can be re-examined when a result looks wrong.
    in_window: bool = True

    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None

    # -- derived quantities -------------------------------------------------------------
    # Computed here rather than at reduction time so that every consumer -- ours or a third
    # party's -- derives them identically from the same fields.

    @property
    def ttft_s(self) -> float | None:
        """Time to first token, measured from the moment the request was issued.

        Deliberately from ``issued_ts`` and not ``connect_ts``: the user waits through
        connection setup too. :attr:`client_overhead_s` reports the part we caused, so a
        run where the load generator is the bottleneck is diagnosable rather than merely
        disappointing.
        """
        if self.first_token_ts is None:
            return None
        return self.first_token_ts - self.issued_ts

    @property
    def client_overhead_s(self) -> float | None:
        """Time spent before the request left the client."""
        if self.connect_ts is None:
            return None
        return self.connect_ts - self.issued_ts

    @property
    def e2e_s(self) -> float | None:
        if self.end_ts is None:
            return None
        return self.end_ts - self.issued_ts

    @property
    def itls_s(self) -> list[float]:
        """Every inter-token gap in this request.

        Empty for a request with fewer than two content tokens -- not zero, and not a single
        averaged value. A one-token response has no inter-token latency to report, and
        inventing one would drag the distribution toward whatever the prefill cost was.
        """
        stamps = ([self.first_token_ts] if self.first_token_ts is not None else []) + self.token_ts
        return [b - a for a, b in zip(stamps, stamps[1:])]

    @property
    def is_failure(self) -> bool:
        """Chapter 7 §6: everything that is not a completed response is a failure."""
        return self.outcome is not Outcome.OK

    def to_json(self) -> str:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return json.dumps(d, separators=(",", ":"))


def write_records(records: list[RequestRecord], fp: IO[str]) -> int:
    """Append records as JSON Lines. Returns the number written.

    JSONL and not a single JSON array because a run that dies at 80% must still leave a
    readable file -- a truncated array is unparseable, and losing an eight-hour campaign to
    a missing bracket is a failure mode this protocol should not have.
    """
    n = 0
    for r in records:
        fp.write(r.to_json())
        fp.write("\n")
        n += 1
    return n


def read_records(fp: IO[str]) -> list[RequestRecord]:
    """Read JSON Lines back. Blank lines are skipped; a malformed line raises.

    Malformed lines are not skipped: a silently dropped record changes the error-rate
    denominator, which is precisely the number that must not move without anyone noticing.
    """
    out: list[RequestRecord] = []
    for lineno, line in enumerate(fp, 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"record file is corrupt at line {lineno}: {exc}") from None
        d["outcome"] = Outcome(d.get("outcome", "ok"))
        out.append(RequestRecord(**d))
    return out
