"""The MoE decode roofline, which prices a step by the experts the whole batch selected.

Per-token active params are the right weight read at batch 1 and only at batch 1. Above it a
step reads the union of every routed expert, and the union saturates long before a production
batch size -- so the flat-active formula understates weight traffic by up to
``total_params / active_params`` and pushes ``roofline_efficiency`` down by the same factor.
The cases here pin both endpoints of the union, the guards against a half-declared geometry,
and the dense behaviour that every already-published report depends on being unchanged.
"""

import math

import pytest

from ascep.capacity import moe_decode_weight_params, roofline_decode_tok_s

# The 26B/3.8B MoE measured on GB200: 128 experts, top-8, ~183M parameters each over a
# ~2.36B shared trunk. Real geometry rather than round numbers, because the point of these
# cases is the size of the correction on a model somebody actually served.
ACTIVE = 3_822_530_590
TOTAL = 25_805_936_206
EXPERTS = 128
TOP_K = 8


def test_a_single_token_step_reads_exactly_the_active_parameters():
    # The identity that makes the correction safe to adopt: at batch 1 the union is the one
    # token's top_k, so the new formula reproduces the old one and no batch-1 figure moves.
    assert moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, 1) == pytest.approx(ACTIVE)


def test_a_large_batch_step_reads_essentially_every_stored_parameter():
    # The fact the flat-active formula denies. 128 experts and top-8 means a token misses a
    # given expert with probability 0.9375, and 0.9375 ** 1024 is about 1e-28.
    assert moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, 1024) == pytest.approx(TOTAL)


def test_the_expert_union_is_monotonic_and_never_leaves_the_active_to_total_band():
    # A step can never read less than one token's experts nor more than the whole checkpoint.
    # A formula that left the band would produce a roofline above the theoretical ceiling or
    # below the batch-1 floor, either of which reads as a plausible number.
    previous = 0.0
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        params = moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, batch)
        assert ACTIVE <= params <= TOTAL, f"batch {batch} left the band at {params:,.0f}"
        assert params >= previous, f"batch {batch} read less than batch {batch // 2}"
        previous = params


def test_the_union_matches_the_closed_form_for_independent_uniform_routing():
    # Pins the model itself, not just its endpoints: an expert is skipped only when all
    # batch_size tokens miss it. Rewriting this as a per-token average would pass both
    # endpoint cases above and still be wrong everywhere between them.
    per_expert = (TOTAL - ACTIVE) / (EXPERTS - TOP_K)
    trunk = ACTIVE - TOP_K * per_expert
    for batch in (3, 32, 100):
        touched = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** batch)
        assert moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, batch) == pytest.approx(
            trunk + touched * per_expert
        )


def test_the_union_saturates_well_below_a_production_batch_size():
    # The operational claim behind the correction. If saturation needed batch 10,000 the
    # flat-active formula would be a fair approximation in practice; it needs batch 32.
    at_32 = moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, 32)
    assert at_32 / TOTAL > 0.88, f"batch 32 read only {at_32 / TOTAL:.2%} of the checkpoint"
    at_192 = moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, 192)
    assert at_192 / TOTAL > 0.999


def test_a_dense_model_declared_as_moe_reads_its_parameters_once_at_any_batch():
    # total == active means there is no expert bank to union, and the per-expert size solved
    # from the two would be zero. Returning something batch-dependent here would make a dense
    # model's roofline fall as concurrency rose.
    for batch in (1, 64, 4096):
        assert moe_decode_weight_params(ACTIVE, ACTIVE, EXPERTS, TOP_K, batch) == ACTIVE


def test_routing_every_token_to_every_expert_is_not_a_division_by_zero():
    # moe_experts == moe_top_k is a legal declaration for a model with no sparsity, and it is
    # the exact input that makes the (experts - top_k) denominator vanish.
    assert moe_decode_weight_params(ACTIVE, TOTAL, TOP_K, TOP_K, 512) == ACTIVE


def test_a_checkpoint_smaller_than_its_active_count_is_rejected_not_extrapolated():
    # An MoE stores every expert and runs top_k, so total below active is a declaration error.
    # Left unchecked it makes per_expert negative and the step's weight read *shrink* as the
    # batch grows, which inflates the roofline in the direction nothing else would catch.
    with pytest.raises(ValueError, match="never the smaller"):
        moe_decode_weight_params(TOTAL, ACTIVE, EXPERTS, TOP_K, 8)


@pytest.mark.parametrize("top_k", [0, -1, EXPERTS + 1])
def test_a_top_k_outside_the_expert_bank_is_rejected(top_k):
    with pytest.raises(ValueError, match="moe_top_k must be in"):
        moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, top_k, 8)


def test_omitting_the_moe_geometry_leaves_the_dense_roofline_exactly_as_it_was():
    # Backward compatibility for every report published before the correction. A dense model
    # passes none of the three new arguments and must get the pre-existing arithmetic.
    got = roofline_decode_tok_s(
        ACTIVE, "bf16", 8e12, batch_size=476, avg_context_tokens=903.35, kv_per_token=20_480
    )
    bytes_per_step = ACTIVE * 2 + 476 * 903.35 * 20_480
    assert got == pytest.approx(8e12 / bytes_per_step * 476)


@pytest.mark.parametrize(
    "partial",
    [
        {"total_params": TOTAL},
        {"moe_experts": EXPERTS},
        {"moe_top_k": TOP_K},
        {"total_params": TOTAL, "moe_experts": EXPERTS},
        {"moe_experts": EXPERTS, "moe_top_k": TOP_K},
    ],
)
def test_a_half_declared_moe_geometry_raises_instead_of_falling_back_to_dense(partial):
    # The dangerous alternative is silence: the caller asked for the expert-union model, got
    # the flat one because one argument was missing, and the understated roofline it returns
    # is indistinguishable from a correct number.
    with pytest.raises(ValueError, match="together"):
        roofline_decode_tok_s(ACTIVE, "bf16", 8e12, batch_size=64, **partial)


def test_the_moe_roofline_is_the_lower_one_and_the_gap_is_the_correction():
    # End to end at the rung that exposed the defect: concurrency 512, effective batch 476.
    # The published efficiency was 0.043 against the flat-active bound; against the union
    # bound the same measurement is 0.158. C6 reads an efficiency near 1.0 as evidence of
    # measurement error, and it cannot do that job against a denominator inflated 3.7x.
    shared = dict(
        precision="bf16",
        hbm_bandwidth_bytes_s=8e12,
        batch_size=476,
        avg_context_tokens=903.35,
        kv_per_token=20_480,
    )
    flat = roofline_decode_tok_s(ACTIVE, **shared)
    union = roofline_decode_tok_s(
        ACTIVE, **shared, total_params=TOTAL, moe_experts=EXPERTS, moe_top_k=TOP_K
    )
    assert union < flat
    assert flat / union == pytest.approx(3.67, abs=0.02)
    measured = 9984.0
    assert measured / flat == pytest.approx(0.043, abs=0.001)
    assert measured / union == pytest.approx(0.158, abs=0.001)


def test_the_correction_vanishes_as_the_batch_returns_to_one():
    # The two formulas must agree exactly where the flat one was right, so adopting the
    # correction cannot move a batch-1 roofline anybody already published.
    shared = dict(precision="bf16", hbm_bandwidth_bytes_s=8e12, batch_size=1)
    flat = roofline_decode_tok_s(ACTIVE, **shared)
    union = roofline_decode_tok_s(
        ACTIVE, **shared, total_params=TOTAL, moe_experts=EXPERTS, moe_top_k=TOP_K
    )
    assert union == pytest.approx(flat)


def test_the_union_never_underflows_to_a_zero_weight_read_at_extreme_batch():
    # (1 - k/E) ** batch underflows to 0.0 for a large enough batch, which is the correct
    # limit rather than a failure -- but only if the surrounding arithmetic still lands on
    # total_params instead of on nan or a negative.
    params = moe_decode_weight_params(ACTIVE, TOTAL, EXPERTS, TOP_K, 10_000_000)
    assert math.isfinite(params)
    assert params == pytest.approx(TOTAL)
