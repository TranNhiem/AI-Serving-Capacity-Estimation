"""The record contract, tested at the four points where the private campaign got it wrong.

Each test below corresponds to a figure the first published report had to declare ``(U) not
measured`` because the harness discarded the evidence. They are regression tests for a
protocol defect, not for a code defect.
"""

import io

import pytest

from ascep.bench import Outcome, RequestRecord, read_records, write_records


def _rec(**kw):
    kw.setdefault("request_id", "r1")
    kw.setdefault("issued_ts", 100.0)
    return RequestRecord(**kw)


# --- inter-token latency is a distribution, not a per-request average ------------------


def test_itl_is_the_gap_between_consecutive_tokens_not_a_per_request_mean():
    """A stall inside one request must be visible in the ITL distribution.

    The campaign this protocol grew out of stored one averaged `tpot` per request. These
    timings average to 0.1 s/token, so a mean-based reduction reports a healthy 0.1 and the
    0.5 s stall -- the only number a user would notice -- disappears entirely.
    """
    r = _rec(first_token_ts=100.5, token_ts=[100.6, 101.1, 101.2], end_ts=101.2)
    assert r.itls_s == pytest.approx([0.1, 0.5, 0.1])
    assert max(r.itls_s) == pytest.approx(0.5)


def test_a_single_token_response_reports_no_inter_token_latency_at_all():
    """Not zero, and not one averaged value: there is no gap to measure.

    Inventing a value here would fold prefill cost into the ITL distribution, which is the
    one quantity ITL is defined to exclude.
    """
    assert _rec(first_token_ts=100.5, end_ts=100.5).itls_s == []


def test_a_request_that_never_produced_a_token_has_no_ttft_and_no_itl():
    r = _rec(outcome=Outcome.TIMEOUT, end_ts=130.0)
    assert r.ttft_s is None
    assert r.itls_s == []


# --- client overhead is separable from server latency ---------------------------------


def test_ttft_is_charged_from_issue_but_client_overhead_stays_visible():
    """The user waits through connection setup, so TTFT includes it.

    It is reported separately as well, because a load generator that has become the
    bottleneck otherwise looks exactly like a slow server.
    """
    r = _rec(issued_ts=100.0, connect_ts=100.2, first_token_ts=100.9)
    assert r.ttft_s == pytest.approx(0.9)
    assert r.client_overhead_s == pytest.approx(0.2)


# --- every issued request is a record, including the ones that never ran --------------


@pytest.mark.parametrize(
    "outcome", [Outcome.REFUSED, Outcome.ERROR, Outcome.TIMEOUT, Outcome.CANCELLED]
)
def test_every_non_ok_outcome_counts_as_a_failure(outcome):
    """Chapter 7 §6 denominates the error rate on requests *issued*.

    A refusal at admission is a capacity failure the user experiences; treating it as
    "not a request" is what let a server shedding a third of its load report 0% errors.
    """
    assert _rec(outcome=outcome).is_failure


def test_a_completed_request_is_not_a_failure():
    assert not _rec(outcome=Outcome.OK, first_token_ts=100.1, end_ts=100.5).is_failure


def test_a_cancelled_in_flight_request_survives_a_round_trip_and_is_still_counted():
    """Dropping requests still in flight at window close removes the slowest ones.

    That biases every latency percentile downward by exactly the amount that matters, so
    the record is kept and the reduction decides what to do with it.
    """
    buf = io.StringIO()
    write_records([_rec(outcome=Outcome.CANCELLED, first_token_ts=100.4)], buf)
    buf.seek(0)
    (back,) = read_records(buf)
    assert back.outcome is Outcome.CANCELLED
    assert back.is_failure


# --- persistence ----------------------------------------------------------------------


def test_records_round_trip_through_jsonl_without_losing_a_field():
    original = _rec(
        request_id="abc",
        issued_ts=1.0,
        connect_ts=1.1,
        first_token_ts=1.5,
        token_ts=[1.6, 1.7],
        end_ts=1.8,
        input_tokens=1024,
        output_tokens=3,
        output_tokens_local=3,
        concurrency=64,
        repetition=2,
        in_window=True,
        finish_reason="stop",
        http_status=200,
        session_id="s-7",
        turn_index=3,
    )
    buf = io.StringIO()
    assert write_records([original], buf) == 1
    buf.seek(0)
    (back,) = read_records(buf)
    assert back == original


def test_warm_up_records_are_retained_and_marked_rather_than_deleted():
    """An excluded record that is still on disk can be re-read when a result looks wrong."""
    buf = io.StringIO()
    write_records([_rec(in_window=False), _rec(request_id="r2", in_window=True)], buf)
    buf.seek(0)
    back = read_records(buf)
    assert [r.in_window for r in back] == [False, True]


def test_a_corrupt_line_raises_instead_of_being_skipped():
    """A silently dropped record moves the error-rate denominator with nobody noticing."""
    buf = io.StringIO('{"request_id":"r1","issued_ts":1.0}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        read_records(buf)


def test_blank_lines_are_tolerated():
    buf = io.StringIO('\n{"request_id":"r1","issued_ts":1.0}\n\n')
    assert len(read_records(buf)) == 1


# --- agent sessions: which requests shared a prefix, and which gaps were tool calls ----


def test_a_single_shot_request_carries_no_session_identity_at_all():
    """The agent fields must default to absent, not to a fabricated session of one.

    Every workload the protocol measured before agents existed would otherwise acquire a
    session id, and a reduction that groups by session would start charging warm-prefix
    economics to runs whose requests genuinely shared nothing.
    """
    r = _rec()
    assert r.session_id is None
    assert r.turn_index is None


def test_a_record_file_written_before_the_agent_fields_existed_still_loads():
    """Older bundles must stay readable, or every published record file becomes waste.

    Chapter 7 §8 requires raw records in the reproduction bundle; a schema addition that
    orphans the bundles already cited in the report defeats the point of keeping them.
    """
    buf = io.StringIO('{"request_id":"r1","issued_ts":1.0,"outcome":"ok","concurrency":8}\n')
    (back,) = read_records(buf)
    assert back.concurrency == 8
    assert back.session_id is None
    assert back.turn_index is None


def test_several_requests_in_one_turn_share_a_turn_index_and_order_by_issue_time():
    """A tool-calling turn is several requests, and the record must say so.

    Counting turns by counting records would report this two-step turn as two turns, which
    halves tool_calls_per_turn and doubles turns_per_session -- both in the direction that
    makes an agent loop look cheaper than it is.
    """
    turn = [
        _rec(request_id="a", issued_ts=10.0, session_id="s1", turn_index=0),
        _rec(request_id="b", issued_ts=14.0, session_id="s1", turn_index=0),
        _rec(request_id="c", issued_ts=20.0, session_id="s1", turn_index=1),
    ]
    assert len({r.turn_index for r in turn}) == 2
    assert len(turn) == 3
    by_turn_zero = sorted((r for r in turn if r.turn_index == 0), key=lambda r: r.issued_ts)
    assert [r.request_id for r in by_turn_zero] == ["a", "b"]


def test_missing_server_usage_stays_none_rather_than_becoming_a_chunk_count():
    """A streaming chunk is not a token, and the substitution is invisible downstream.

    The campaign fell back to counting chunks when the server omitted usage, which silently
    redefined every per-token figure for that run.
    """
    r = _rec(first_token_ts=100.1, token_ts=[100.2, 100.3], end_ts=100.3)
    assert r.output_tokens is None
