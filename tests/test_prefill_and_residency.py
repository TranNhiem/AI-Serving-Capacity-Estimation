"""Prefill, KV-residency and accumulating-context formulas, pinned to v0.4.0 compatibility.

A plausible capacity claim fails on two new axes in v0.5.0: it prices prompts as if they
were free, and it prices an idle agent as if it had released its KV. The cases here pin
both floors, the exact mix identity behind them, and the backward compatibility owed to
already published reports.
"""

from dataclasses import replace

import pytest

from ascep.capacity import Constraint, Workload, capacity_at, gpus_required

# A direct user count keeps every demand branch independent of Little's-law arithmetic.
# These values are illustrative: the properties under test are exact identities and
# errors, not measured server behaviour.
BASE_PROMPT_WORKLOAD = Workload(
    concurrent_users=100,
    avg_session_seconds=500,
    duty_cycle=0.5,
    input_tokens_per_request=200,
    output_tokens_per_request=100,
    requests_per_session=2,
)

IDENTITY_WORKLOAD = Workload(
    concurrent_users=64,
    duty_cycle=0.75,
    target_tok_s_per_user=12,
    input_tokens_per_request=300,
    output_tokens_per_request=100,
)

DETAIL_KEYS_V040 = [
    "avg_context_tokens",
    "users_kv_floor",
    "users_throughput_floor",
    "per_user_tok_s",
    "headroom",
    "kv_tokens",
    "throughput_tok_s",
]


def test_omitting_prefill_reproduces_the_three_floor_answer_and_detail_exactly():
    """Backward compatibility is the published-number guarantee, not an implementation
    detail.

    A fourth key or a changed floor would change every report produced by v0.4.0 while
    leaving its schema apparently valid. Pinning both the seven v0.4.0 keys and their
    insertion order catches a changed serialisation before it rewrites a published
    capacity claim.
    """
    workload = replace(
        BASE_PROMPT_WORKLOAD,
        concurrent_users=1_000,
        duty_cycle=0.5,
        avg_session_seconds=100,
        input_tokens_per_request=100,
        output_tokens_per_request=100,
        requests_per_session=1,
    )
    result = capacity_at(n_gpus=1, kv_tokens=60_000, throughput_tok_s=1_000, workload=workload)

    assert result.max_concurrent_users == pytest.approx(800.0)
    assert result.binding_constraint == Constraint.KV
    assert list(result.detail) == DETAIL_KEYS_V040
    # An all-or-nothing key check alone cannot catch the same keys carrying changed
    # values; capacity_at derives each entry above, so the values are asserted once here.
    assert result.detail == {
        "avg_context_tokens": 150.0,
        "users_kv_floor": 800.0,
        "users_throughput_floor": 1_000.0,
        "per_user_tok_s": 1.0,
        "headroom": 1.0,
        "kv_tokens": 60_000,
        "throughput_tok_s": 1_000,
    }
    # The published answer must be one of the floors verbatim, not a blend of them: a mean
    # or a product would still look plausible in the report and would name a constraint the
    # operator cannot act on.
    assert result.max_concurrent_users == result.detail["users_kv_floor"]


@pytest.mark.parametrize("measured_mix", [0.5, 1.0, 3.0])
def test_users_prefill_over_users_throughput_equals_measured_mix_over_declared_mix(measured_mix):
    """This identity is the reason the floor is predictable instead of merely cautious.

    Let rho_m be the mix measured on the rung and rho_w the mix declared by the workload.
    A capacity number without the fourth floor is wrong by exactly rho_w / rho_m whenever
    prefill binds. Approximate agreement is not enough: a loose tolerance would license a
    formula that loses either the duty cycle or the media term and still looks close on a
    chosen case.
    """
    workload = IDENTITY_WORKLOAD
    reported_mix = (workload.input_tokens_per_request + workload.media_tokens_per_request) / (
        workload.output_tokens_per_request + workload.reasoning_tokens_per_request
    )
    throughput = 1_000.0
    result = capacity_at(
        n_gpus=1,
        kv_tokens=1_000_000,
        throughput_tok_s=throughput,
        prefill_tok_s=throughput * measured_mix,
        workload=workload,
    )

    ratio = result.detail["users_prefill_floor"] / result.detail["users_throughput_floor"]
    assert ratio == pytest.approx(measured_mix / reported_mix, rel=1e-12)


def test_the_prefill_floor_can_bind_and_reports_the_string_prefill():
    """Without this path an input-heavy workload inherits capacity measured on an
    output-heavy run.

    Each of the 64 active sessions is pinned at 12 output tok/s, and the declared 3:1 mix
    makes that 36 input tok/s apiece -- 2,304 input tok/s at peak against the 360 the rung
    sustained, so ten users where the output side would have promised a hundred. Reporting
    anything except prefill would point the operator at decode throughput when prompt
    capacity is what has to be bought.
    """
    workload = replace(IDENTITY_WORKLOAD, duty_cycle=1.0, input_tokens_per_request=300)
    result = capacity_at(
        n_gpus=1,
        kv_tokens=1_000_000,
        throughput_tok_s=1_200,
        prefill_tok_s=360,
        workload=workload,
    )

    assert result.detail["users_prefill_floor"] == pytest.approx(10.0)
    assert result.detail["users_throughput_floor"] == pytest.approx(100.0)
    assert result.max_concurrent_users == pytest.approx(10.0)
    assert result.binding_constraint == "prefill"


@pytest.mark.parametrize(
    ("declared_input", "measured_mix"),
    [(100, 3.0), (300, 3.0)],
    ids=["declared-mix-one-third-of-measured", "declared-mix-equal-to-measured"],
)
def test_prefill_does_not_bind_when_the_declared_mix_is_not_heavier(declared_input, measured_mix):
    """A sound rung already paid the same or greater prompt cost while it was measured.

    Letting prefill bind there would double-count it and turn a sustainable output rate
    into an artificially low user count. The equal-mix case must also stay on throughput:
    the rung's measured output tokens/s already embeds exactly this prompt load.
    """
    workload = replace(IDENTITY_WORKLOAD, input_tokens_per_request=declared_input)
    result = capacity_at(
        n_gpus=1,
        kv_tokens=1_000_000,
        throughput_tok_s=900,
        prefill_tok_s=900 * measured_mix,
        workload=workload,
    )

    assert result.max_concurrent_users == pytest.approx(100.0)
    assert result.binding_constraint == Constraint.THROUGHPUT


def test_demand_prefill_tok_s_converts_a_generation_target_through_the_declared_shape():
    """The per-stream target fixes decoding speed, not request volume directly.

    Holding generation at 10 tok/s on a request that generates 100 tokens implies 0.1
    requests per second per active session, each carrying 500 prompt tokens. Treating
    the target itself as the request rate would price this workload five times too
    high; treating prompt tokens as irrelevant would price the axis at zero.
    """
    workload = Workload(
        concurrent_users=40,
        duty_cycle=0.5,
        target_tok_s_per_user=10,
        input_tokens_per_request=300,
        media_tokens_per_request=200,
        output_tokens_per_request=70,
        reasoning_tokens_per_request=30,
    )

    assert workload.demand_tok_s() == 200.0
    assert workload.demand_prefill_tok_s() == 1_000.0


def test_demand_prefill_tok_s_uses_session_demand_when_no_stream_target_is_set():
    """Duration-based demand must count prompts once per request, not once per session.

    Formula shape: users x prompt tokens per request x requests per session / session
    seconds, before duty cycle is applied. Applying duty here would be a second error:
    peak_concurrent_users is the number whose session lifetime spans avg_session_seconds,
    while duty_cycle is reserved for the in-flight generation branch.
    """
    workload = replace(
        BASE_PROMPT_WORKLOAD,
        concurrent_users=200,
        avg_session_seconds=500,
        duty_cycle=0.25,
        input_tokens_per_request=150,
        media_tokens_per_request=50,
        requests_per_session=5,
    )

    assert workload.demand_prefill_tok_s() == 400.0


def test_a_generation_target_without_generated_tokens_raises_instead_of_zeroing_prefill():
    """An embedding or reranking service serves requests and returns no tokens.

    Silently returning 0 would delete the prompt floor for exactly those services and
    let max_requests_per_s fall back to the broken zero-output path. Raising makes the
    impossible declaration visible before it becomes a conforming-looking capacity tier.
    """
    workload = Workload(
        concurrent_users=50,
        target_tok_s_per_user=10,
        input_tokens_per_request=1_000,
        output_tokens_per_request=0,
        reasoning_tokens_per_request=0,
    )

    with pytest.raises(ValueError, match="target_tok_s_per_user needs"):
        workload.demand_prefill_tok_s()


def test_a_zero_output_service_gets_requests_per_second_from_the_prompt_side():
    """Zero generated tokens cannot remain the denominator for request capacity.

    The old generated-token path reports 0 req/s for an embedding service that serves 30
    requests per second and simply returns no tokens. Dividing aggregate prompt demand by
    declared prompt size preserves the axis those services are actually sold on.
    """
    workload = Workload(
        concurrent_users=100,
        duty_cycle=1.0,
        target_tok_s_per_user=10,
        input_tokens_per_request=1_000,
        output_tokens_per_request=0,
        reasoning_tokens_per_request=1,
    )
    result = capacity_at(
        n_gpus=1,
        kv_tokens=1_000_000,
        throughput_tok_s=1_000,
        prefill_tok_s=30_000,
        workload=workload,
    )

    assert result.binding_constraint == Constraint.PREFILL
    assert result.max_requests_per_s == pytest.approx(30.0)


def test_avg_context_tokens_is_bit_identical_when_context_growth_is_zero():
    """Default zero must leave every pre-0.5.0 workload exactly where the protocol left
    it.

    This is a published-number property, not merely an arithmetic regression: changing
    the context by even one token changes the KV floor of reports that did not ask for
    the new estimator. The direct inputs make the old expression input + media +
    (reasoning + output) / 2 independently checkable.
    """
    workload = replace(
        BASE_PROMPT_WORKLOAD,
        input_tokens_per_request=211,
        media_tokens_per_request=37,
        output_tokens_per_request=25,
        reasoning_tokens_per_request=19,
        requests_per_session=7,
        context_growth_tokens_per_turn=0,
    )

    assert workload.avg_context_tokens() == 270.0


def test_avg_context_tokens_adds_the_mean_reread_over_an_accumulating_transcript():
    """An agent turn re-reads every earlier turn; chat arithmetic prices none of that
    growth.

    Four turns starting from a 300-token prompt average 300 x (4 - 1) / 2 = 450 added
    resident tokens before generated output is counted. Dropping the term would let the
    KV floor report capacity for sessions the pool cannot hold.
    """
    growth = replace(
        BASE_PROMPT_WORKLOAD,
        input_tokens_per_request=100,
        media_tokens_per_request=20,
        output_tokens_per_request=30,
        reasoning_tokens_per_request=10,
        requests_per_session=4,
        context_growth_tokens_per_turn=300,
    )

    assert growth.avg_context_tokens() == 590.0


def test_context_growth_is_not_charged_at_one_request_per_session():
    """There is no earlier turn to re-read on a single-turn session.

    Applying g x (N - 1) / 2 only where N > 1 keeps an isolated request on the old
    estimator even when the workload also declares growth for other shapes. Charging it
    anyway would invent context for a request the transcript has not accumulated.
    """
    workload = replace(
        BASE_PROMPT_WORKLOAD,
        input_tokens_per_request=100,
        output_tokens_per_request=20,
        requests_per_session=1,
        context_growth_tokens_per_turn=10_000,
    )

    assert workload.avg_context_tokens() == 110.0


def test_kv_residency_replaces_duty_cycle_and_changes_user_capacity_by_its_ratio():
    """A paused agent is not generating, but it is still holding its KV blocks.

    Reusing the 10% decode duty cycle as memory residency releases context the engine
    has not released and inflates user capacity by 0.3 / 0.1. The ratio pin prevents a
    refactor that preserves the field while continuing to divide by the decode value.
    """
    workload = replace(
        BASE_PROMPT_WORKLOAD,
        concurrent_users=1,
        duty_cycle=0.1,
        input_tokens_per_request=100,
        output_tokens_per_request=0,
        requests_per_session=1,
    )
    with_residency = replace(workload, kv_residency=0.3)

    duty_result = capacity_at(n_gpus=1, kv_tokens=30_000, throughput_tok_s=1, workload=workload)
    residency_result = capacity_at(
        n_gpus=1, kv_tokens=30_000, throughput_tok_s=1, workload=with_residency
    )

    assert duty_result.detail["users_kv_floor"] == pytest.approx(3_000.0)
    assert residency_result.detail["users_kv_floor"] == pytest.approx(1_000.0)
    assert duty_result.detail["users_kv_floor"] / residency_result.detail["users_kv_floor"] == (
        pytest.approx(3.0)
    )


def test_kv_residency_below_duty_cycle_raises_before_the_kv_floor_can_pass_it():
    """A session cannot hand back context while it is still spending part of its time
    generating from it.

    A residency below duty_cycle can only be a mix-up, and silently accepting it deflates
    the per-session footprint until the KV floor passes almost any workload. The divide
    that would produce that floor must never run.
    """
    workload = replace(BASE_PROMPT_WORKLOAD, duty_cycle=0.8, kv_residency=0.3)

    with pytest.raises(ValueError, match="kv_residency must be >= duty_cycle"):
        capacity_at(n_gpus=1, kv_tokens=1_000, throughput_tok_s=1_000, workload=workload)


@pytest.mark.parametrize(
    ("supply", "value"),
    [
        ("kv_tokens", -1.0),
        ("throughput_tok_s", -1.0),
        ("prefill_tok_s", -1.0),
    ],
)
def test_a_negative_supply_is_refused_rather_than_reported_as_a_capacity(supply, value):
    """A negative supply is not a small supply.

    Each of the three divides straight into a floor, min() then picks the negative one
    because it is the smallest, and the result is a Capacity announcing that the cluster
    serves minus four hundred users and is KV-bound. No reader treats that as a capacity;
    every downstream comparison does -- gpus_required's loop, the tier table, the report's
    binding_constraint field. Refusing at the door is the only place the sign is still
    attached to the input that carried it.
    """
    supplies = {"kv_tokens": 1_000_000.0, "throughput_tok_s": 5_000.0, "prefill_tok_s": 5_000.0}
    supplies[supply] = value
    with pytest.raises(ValueError, match=f"{supply} must be >= 0"):
        capacity_at(n_gpus=1, workload=BASE_PROMPT_WORKLOAD, **supplies)


def test_zero_kv_stays_a_measurable_answer_rather_than_an_error():
    """Zero is the honest report for weights that do not fit: kv_pool_bytes clamps to it.

    Folded into the negative guard it would become an exception, and a configuration that
    genuinely serves nobody would raise where it should publish a zero on the KV axis.
    """
    result = capacity_at(
        n_gpus=1, kv_tokens=0.0, throughput_tok_s=5_000.0, workload=BASE_PROMPT_WORKLOAD
    )
    assert result.max_concurrent_users == 0.0
    assert result.binding_constraint is Constraint.KV


def test_sizing_a_reranker_without_its_prefill_rate_buys_a_tenth_of_the_cluster():
    """gpus_required returns the first replica count that clears the floors it was handed,
    so an axis omitted there is not an unpriced axis -- it is a wrong purchase order.

    The reranker is the sharpest case because it is the one the fourth floor was added for:
    output_tokens_per_request is zero, so the throughput floor is infinite and only KV binds.
    Sized blind, 500 concurrent users fit on one GPU whose memory could hold them; sized with
    the measured input rate beside it, the same workload needs ten, and the constraint the
    report names changes from memory to compute. Both answers claim 500 users. Only one of
    them can serve 4,000 prompt tokens apiece at the rate the declaration asks for.
    """
    reranker = Workload(
        concurrent_users=500,
        duty_cycle=1.0,
        avg_session_seconds=10,
        input_tokens_per_request=4_000,
        output_tokens_per_request=0,
        requests_per_session=1,
    )
    per_gpu = {"kv_tokens_per_gpu": 2_000_000.0, "throughput_tok_s_per_gpu": 10_000.0}

    blind = gpus_required(reranker, headroom=1.0, **per_gpu)
    priced = gpus_required(reranker, headroom=1.0, prefill_tok_s_per_gpu=20_000.0, **per_gpu)

    assert blind.n_gpus == 1
    assert blind.binding_constraint is Constraint.KV
    assert priced.n_gpus == 10
    assert priced.binding_constraint is Constraint.PREFILL


def test_sizing_without_the_new_argument_is_byte_identical_to_the_v040_answer():
    """The pass-through has to be inert when nothing was measured, for the same reason
    capacity_at's fourth floor is: a sizing that moved on upgrade would silently rewrite
    every cluster recommendation already published, and C11 -- not a changed number -- is
    the protocol's way of saying an axis went unpriced."""
    per_gpu = {"kv_tokens_per_gpu": 500_000.0, "throughput_tok_s_per_gpu": 4_000.0}

    default = gpus_required(BASE_PROMPT_WORKLOAD, **per_gpu)
    explicit_none = gpus_required(BASE_PROMPT_WORKLOAD, prefill_tok_s_per_gpu=None, **per_gpu)

    assert default == explicit_none
    assert list(default.detail) == DETAIL_KEYS_V040
