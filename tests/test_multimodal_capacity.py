"""Media-token and reasoning-mode formulas, pinned against measured server behaviour.

A plausible-but-wrong media formula passes every sanity check and then mis-sizes the
cluster; the regression cases here are real measurements, not invented numbers, and they
are the only thing standing between this predictor and that failure.
"""

from dataclasses import replace

import pytest

from ascep.capacity import (
    Workload,
    capacity_at,
    image_tokens,
    media_arrival_check,
    media_token_cap_check,
    video_frames,
    video_tokens,
    vision_encoder_bytes,
)

# Geometry of the Qwen3-VL-class checkpoint behind the measured runs: 16 px patches, 2x2
# spatial merge, 2x temporal merge, 2 fps sampling of 768x432 frames.
CLIP_KWARGS = dict(
    frame_policy="uniform-fps",
    sampling_fps=2.0,
    temporal_merge=2,
    patch_px=16,
    spatial_merge=2,
)
DEFAULT_PIXEL_BUDGET = 25_165_824
RAISED_PIXEL_BUDGET = 251_658_240  # 10x the default; high enough that the cap never binds

# The published examples/chatbot-10k-dau workload. Its numbers are in a report, so they
# are load-bearing.
CHATBOT_10K_DAU = Workload(
    daily_active_users=10_000,
    sessions_per_user_per_day=2,
    avg_session_seconds=600,
    peak_to_mean=4.0,
    duty_cycle=0.4,
    input_tokens_per_request=1_000,
    output_tokens_per_request=400,
    requests_per_session=5,
)

DECLARED_TABLE = [
    {"max_width": 512, "max_height": 512, "tokens": 64},
    {"max_width": 1024, "max_height": 1024, "tokens": 256},
    {"max_width": 2048, "max_height": 2048, "tokens": 1024},
]


@pytest.mark.parametrize(
    ("duration_s", "longest_edge_px", "measured_tokens"),
    [
        (39.1, DEFAULT_PIXEL_BUDGET, 12_090),
        (47.4, DEFAULT_PIXEL_BUDGET, 12_345),
        (94.8, DEFAULT_PIXEL_BUDGET, 12_333),
        (39.1, RAISED_PIXEL_BUDGET, 13_533),
        (47.4, RAISED_PIXEL_BUDGET, 16_293),
        (94.8, RAISED_PIXEL_BUDGET, 32_853),
    ],
    ids=[
        "39.1s-default-budget",
        "47.4s-default-budget",
        "94.8s-default-budget",
        "39.1s-raised-budget",
        "47.4s-raised-budget",
        "94.8s-raised-budget",
    ],
)
def test_video_tokens_reproduces_the_six_measured_server_counts(
    duration_s, longest_edge_px, measured_tokens
):
    """These six numbers are measured on an 8x H100 vLLM deployment, not invented.

    They are the only thing standing between this predictor and a plausible formula that
    is wrong: one that scales linearly with duration, or that drops the temporal_merge
    factor from the budget ceiling and predicts 24,576 tokens where the server truncates
    at 12,288. Either version reads fine in a report and mis-sizes the cluster by 2x.
    """
    predicted = video_tokens(duration_s, 768, 432, longest_edge_px=longest_edge_px, **CLIP_KWARGS)
    rel_err = abs(predicted - measured_tokens) / measured_tokens
    assert rel_err <= 0.05, (
        f"{duration_s}s clip at budget {longest_edge_px}: predicted {predicted} tokens vs "
        f"measured {measured_tokens} (relative error {rel_err:.1%}, tolerance 5%)"
    )


def test_the_default_budget_caps_all_three_clips_at_the_same_token_count():
    """Under the default budget the capped column must be flat, because that flatness IS
    the cap.

    The 5% tolerance above would hide a predictor that let the capped column drift with
    duration - 12,090 and 12,345 both pass - but such a predictor is modelling something
    the server does not do, and it will invent a duration dependence that does not exist.
    """
    predicted = {
        video_tokens(d, 768, 432, longest_edge_px=DEFAULT_PIXEL_BUDGET, **CLIP_KWARGS)
        for d in (39.1, 47.4, 94.8)
    }
    assert predicted == {12_288}, (
        f"capped predictions must be identical across durations, got {sorted(predicted)}"
    )


@pytest.mark.parametrize(
    ("width_px", "height_px", "expected_tokens"),
    [(1024, 1024, 1024), (640, 480, 300)],
)
def test_a_16_px_patch_with_spatial_merge_2_costs_one_token_per_32x32_px(
    width_px, height_px, expected_tokens
):
    """One token per 32x32 px is the conversion every prompt budget for this model class
    is built on; getting it wrong scales the error into every media workload."""
    tokens = image_tokens(
        width_px, height_px, policy="dynamic-resolution", patch_px=16, spatial_merge=2
    )
    assert tokens == expected_tokens, (
        f"{width_px}x{height_px}: expected {expected_tokens} tokens, got {tokens}"
    )


def test_a_partial_patch_rounds_up_rather_than_disappearing():
    """A partial patch is still a patch.

    Rounding 1000 px down to 31 patches instead of up to 32 under-counts this image by 63
    tokens, and the same silent under-count applies to every prompt and every video frame
    the workload sends.
    """
    tokens = image_tokens(1000, 1000, policy="dynamic-resolution", patch_px=16, spatial_merge=2)
    assert tokens == 1024, f"1000x1000 must round up to 32 patches per side: got {tokens}"


def test_fixed_grid_charges_every_image_the_same_regardless_of_resolution():
    """A fixed-grid model that started scaling with resolution would break the prompt
    floor of every workload sized on its declared per-image cost."""
    kwargs = dict(policy="fixed-grid", fixed_tokens=256)
    assert image_tokens(64, 64, **kwargs) == 256
    assert image_tokens(4096, 4096, **kwargs) == 256


def test_declared_table_charges_the_smallest_row_that_covers_the_image():
    """Picking a larger covering row would overstate every prompt; the table is a
    declaration of what the preprocessor does, not an approximation of it."""
    assert image_tokens(300, 200, policy="declared-table", table=DECLARED_TABLE) == 64
    assert image_tokens(800, 600, policy="declared-table", table=DECLARED_TABLE) == 256


def test_declared_table_raises_when_no_row_covers_the_image():
    """Falling back to the nearest row would report a confident token count for an image
    the preprocessor will actually reject or rescale - precise and wrong is worse than
    an exception."""
    with pytest.raises(ValueError, match="no declared-table row"):
        image_tokens(3000, 100, policy="declared-table", table=DECLARED_TABLE)


def test_an_unknown_image_policy_raises_naming_the_policy():
    """A silent default here becomes a wrong token count in every downstream floor."""
    with pytest.raises(ValueError, match="magic-grid"):
        image_tokens(100, 100, policy="magic-grid", patch_px=16)


def test_an_unknown_frame_policy_raises_naming_the_policy():
    """Same failure, one level up: a misspelled policy in a config must not quietly
    become a frame count."""
    with pytest.raises(ValueError, match="magic-fps"):
        video_frames(10.0, policy="magic-fps", sampling_fps=2.0)


def test_uniform_fps_rounds_a_partial_final_second_up():
    """Rounding down would drop the partial second's frame from every clip; over a corpus
    of short clips that is a systematic under-count of the prefill floor."""
    frames = video_frames(10.1, policy="uniform-fps", sampling_fps=2.0)
    assert frames == 21, f"10.1 s at 2 fps must ceil to 21 frames, got {frames}"


def test_uniform_count_ignores_duration_entirely():
    """If duration leaked into a uniform-count policy, long clips would be budgeted for
    frames the model never samples."""
    short = video_frames(5.0, policy="uniform-count", frame_count=8)
    long = video_frames(500.0, policy="uniform-count", frame_count=8)
    assert short == long == 8, f"uniform-count must not see duration: {short} vs {long}"


def test_max_frames_clamps_last_and_the_clamp_says_nothing():
    """The clamp is where a long clip silently becomes a sparse one.

    A 240 s clip clamped to 12 frames is sampled at an effective 0.05 fps - forty times
    sparser than the 2 fps the workload declared - and nothing in the return value says
    so. Pin the order (sample, then clamp) so this stays a visible, deliberate behaviour
    rather than something a refactor 'fixes'.
    """
    frames = video_frames(240.0, policy="uniform-fps", sampling_fps=2.0, max_frames=12)
    assert frames == 12, f"max_frames must clamp after sampling: got {frames}"


def test_the_three_capped_measurements_are_detected_as_capped():
    """If this check stopped firing, a capped run would size the cluster as an uncapped
    one and the cap would be discovered in production."""
    result = media_token_cap_check([(39.1, 12_090), (47.4, 12_345), (94.8, 12_333)])
    assert result.capped, f"expected capped=True, got: {result.explanation}"


def test_the_three_uncapped_measurements_are_detected_as_uncapped():
    """A false positive here is also expensive: it sends the reader hunting for a cap
    that does not exist and discrediting measurements that are fine."""
    result = media_token_cap_check([(39.1, 13_533), (47.4, 16_293), (94.8, 32_853)])
    assert not result.capped, f"expected capped=False, got: {result.explanation}"


def test_cap_check_refuses_fewer_than_two_samples():
    """One measurement cannot distinguish a cap from a scaling curve; pretending it can
    turns a guess into a reported pass."""
    with pytest.raises(ValueError, match="at least two samples"):
        media_token_cap_check([(39.1, 12_090)])


def test_cap_check_refuses_a_sample_set_that_spans_too_little():
    """This is the important case: an inconclusive check must raise, not return
    capped=False.

    These two points are flat in tokens, exactly like the capped runs - but they span
    only 1.2x in duration, so the flatness proves nothing. 'We looked and found no cap'
    and 'we could not look' size a cluster differently, and an inconclusive check
    reported as a pass is the worse failure.
    """
    with pytest.raises(ValueError, match="inconclusive"):
        media_token_cap_check([(39.1, 12_090), (47.4, 12_345)])


def test_a_prompt_matching_the_text_only_prediction_means_the_media_never_arrived():
    """The real cause this guards against: an AV1 corpus that the container's decoder
    turned into zero frames, with no error raised.

    Every request went through as text, and the run published a media capacity figure
    measured on no media. A prompt-token count within tolerance of the text-only
    prediction is the only signal that catches it.
    """
    result = media_arrival_check(1_010, 1_000)
    assert result.capped, f"expected the missing-media verdict, got: {result.explanation}"


def test_a_prompt_well_above_the_text_only_prediction_means_the_media_arrived():
    """The check must not cry wolf on a healthy run, or operators learn to ignore it -
    which is the same as not having it."""
    result = media_arrival_check(13_288, 1_000)
    assert not result.capped, f"expected media-arrived, got: {result.explanation}"


def test_a_replicated_vision_tower_costs_exactly_tp_times_the_sharded_one():
    """Assuming the vision tower shards with the language model is how it goes missing
    from a memory budget entirely.

    On a small language model the replicated encoder is a double-digit percentage of the
    weight floor; dividing it by TP because the LM divides is not a small error, it is
    the whole encoder.
    """
    params = 400_000_000
    replicated = vision_encoder_bytes(params, "bf16", tensor_parallel=8, replicated_per_rank=True)
    sharded = vision_encoder_bytes(params, "bf16", tensor_parallel=8, replicated_per_rank=False)
    assert replicated == params * 2
    assert replicated == sharded * 8, (
        f"replicated ({replicated}) and sharded ({sharded}) must differ by exactly the TP width"
    )


def test_default_media_and_reasoning_fields_leave_the_published_numbers_untouched():
    """examples/chatbot-10k-dau is in a published report; a silent change here rewrites
    its numbers.

    With the two new fields at their defaults, avg_context_tokens() and demand_tok_s()
    must return exactly what they returned before the fields existed.
    """
    assert CHATBOT_10K_DAU.avg_context_tokens() == 1200.0
    demand = CHATBOT_10K_DAU.demand_tok_s()
    assert demand == pytest.approx(1851.85, abs=0.01), (
        f"published demand is 1851.85 tok/s, got {demand}"
    )


def test_media_tokens_are_resident_in_full_while_reasoning_tokens_are_halved():
    """The asymmetry is deliberate: prompt tokens sit in KV for the whole request, while
    generated tokens accumulate from zero, so a mid-generation session holds half of them.

    Halving media would understate the KV floor of every multimodal workload; failing to
    halve reasoning would overstate it.
    """
    with_media = replace(CHATBOT_10K_DAU, media_tokens_per_request=800)
    with_reasoning = replace(CHATBOT_10K_DAU, reasoning_tokens_per_request=200)
    media_delta = with_media.avg_context_tokens() - CHATBOT_10K_DAU.avg_context_tokens()
    reasoning_delta = with_reasoning.avg_context_tokens() - CHATBOT_10K_DAU.avg_context_tokens()
    assert media_delta == 800.0, f"media tokens must add in full, got delta {media_delta}"
    assert reasoning_delta == 100.0, (
        f"reasoning tokens must be halved like output, got a delta of {reasoning_delta}"
    )


def test_reasoning_tokens_raise_demand_while_media_tokens_do_not():
    """Media sits in the prompt: it costs KV, not decode throughput. Reasoning is
    generated: it costs both. Conflating them misstates which floor binds."""
    base = CHATBOT_10K_DAU.demand_tok_s()
    with_media = replace(CHATBOT_10K_DAU, media_tokens_per_request=8_000).demand_tok_s()
    with_reasoning = replace(CHATBOT_10K_DAU, reasoning_tokens_per_request=200).demand_tok_s()
    assert with_media == base, f"media tokens must not move demand: {with_media} vs {base}"
    assert with_reasoning > base, f"reasoning must raise demand: {with_reasoning} vs {base}"


def test_reasoning_mode_multiplies_the_throughput_floor_by_over_50x():
    """This is why reasoning mode is a required declaration rather than a footnote.

    The same workload at 120 output tokens per request versus 120 output plus 33,709
    reasoning tokens is not a tuning detail; it is a different cluster. A report that
    omits the reasoning term understates the throughput floor by the full ratio.
    """
    visible = replace(CHATBOT_10K_DAU, output_tokens_per_request=120)
    thinking = replace(visible, reasoning_tokens_per_request=33_709)
    ratio = thinking.demand_tok_s() / visible.demand_tok_s()
    assert ratio > 50, f"reasoning mode must multiply demand by over 50x, got {ratio:.1f}x"


def test_a_request_is_priced_at_every_token_it_generates_including_the_reasoning_ones():
    """The requests/s denominator must count the same tokens the numerator does.

    ``capacity_at`` derives requests/s by dividing generated tokens/s by tokens per request,
    and the numerator comes from ``demand_tok_s()``, which counts reasoning as output. Pricing
    the request at its visible tokens alone therefore divides a thinking-mode rate by a
    non-thinking-mode request, and the quotient is not requests/s at all -- it is requests/s
    times the reasoning ratio, 282x on this checkpoint. The error is invisible in review
    because every term is individually right, and it inflates ``daily_requests()`` in the
    direction that makes a cluster look like it can serve traffic it cannot.
    """
    thinking = replace(
        CHATBOT_10K_DAU, output_tokens_per_request=120, reasoning_tokens_per_request=33_709
    )
    capacity = capacity_at(
        n_gpus=1, kv_tokens=40_000_000, throughput_tok_s=200_000.0, workload=thinking
    )
    generated_per_request = (
        thinking.output_tokens_per_request + thinking.reasoning_tokens_per_request
    )
    expected = capacity.max_tokens_per_s / generated_per_request
    assert capacity.max_requests_per_s == pytest.approx(expected), (
        f"requests/s must be generated tok/s over generated tokens per request: "
        f"{capacity.max_requests_per_s} vs {expected}"
    )
    # The visible-only divisor is what shipped, so name the number it produced: a test that
    # only checked the identity above would still pass if the two terms drifted together.
    visible_only = capacity.max_tokens_per_s / thinking.output_tokens_per_request
    assert capacity.max_requests_per_s < visible_only / 280, (
        f"the visible-output divisor overstates requests/s by ~282x and must not return: "
        f"{capacity.max_requests_per_s} vs {visible_only}"
    )


def test_a_workload_that_declares_no_reasoning_keeps_the_requests_per_second_it_published():
    """The denominator fix must be inert wherever reasoning is zero.

    Every capacity report published before this change declared no reasoning tokens, so if
    the corrected divisor moved those numbers the fix would silently restate history as well
    as correct it.
    """
    capacity = capacity_at(
        n_gpus=1, kv_tokens=40_000_000, throughput_tok_s=200_000.0, workload=CHATBOT_10K_DAU
    )
    expected = capacity.max_tokens_per_s / CHATBOT_10K_DAU.output_tokens_per_request
    assert capacity.max_requests_per_s == pytest.approx(expected)
