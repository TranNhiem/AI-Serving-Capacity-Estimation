"""The three attention families scale KV differently, and confusing them is a sizing disaster.

These tests pin the distinctions that `kv_bytes_per_token` alone cannot express.
"""

import pytest

from ascep.capacity import (
    GIB,
    effective_layer_frac,
    kv_bytes_per_token,
    kv_bytes_per_token_mla,
    kv_capacity_sessions,
    kv_heads_per_rank,
)

# DeepSeek-V3 geometry, verbatim from its config.json.
DSV3 = dict(n_layers=61, kv_lora_rank=512, qk_rope_head_dim=64)


def test_mla_is_vastly_cheaper_than_the_naive_gqa_model():
    """Applying the GQA formula to an MLA model overstates KV by ~57x."""
    mla = kv_bytes_per_token_mla(**DSV3)
    naive = kv_bytes_per_token(n_layers=61, n_kv_heads=128, head_dim=128, tensor_parallel=8)
    assert naive / mla > 50
    # And the practical consequence, on 8 GPUs with 40 GiB of KV each at a 32,000-token
    # context. Spelled out rather than written 32k: at 32,768 the answer is 149, and a
    # protocol about removing ambiguity should not leave a 3% one in its headline example.
    pool = 8 * 40 * GIB
    assert round(kv_capacity_sessions(pool, mla, 32_000)) == 153
    assert round(kv_capacity_sessions(pool, naive, 32_000)) == 3


def test_mla_has_no_tensor_parallel_argument():
    """The latent is replicated per rank, so cluster KV/token does not shrink with TP.

    If a future edit adds a `tensor_parallel` parameter that divides the result, this test
    should fail — that would be reintroducing the GQA sharding assumption MLA does not obey.
    """
    with pytest.raises(TypeError):
        kv_bytes_per_token_mla(tensor_parallel=8, **DSV3)


def test_constant_state_capacity_is_flat_in_context():
    """SSM / linear-attention state is per-sequence, not per-token: no long-context cliff."""
    pool = 8 * 40 * GIB
    sessions = [
        kv_capacity_sessions(pool, kv_bytes_per_sequence=256 * 1024, avg_context_tokens=ctx)
        for ctx in (2_000, 32_000, 200_000)
    ]
    assert len(set(sessions)) == 1, "constant-state capacity must not vary with context"


def test_hybrid_stack_costs_are_additive():
    pool = 8 * 40 * GIB
    per_tok, per_seq, ctx = 70_272, 256 * 1024, 32_000
    hybrid = kv_capacity_sessions(pool, per_tok, ctx, kv_bytes_per_sequence=per_seq)
    attn_only = kv_capacity_sessions(pool, per_tok, ctx)
    ssm_only = kv_capacity_sessions(pool, kv_bytes_per_sequence=per_seq, avg_context_tokens=ctx)
    assert hybrid < attn_only and hybrid < ssm_only


def test_kv_capacity_sessions_rejects_an_empty_declaration():
    with pytest.raises(ValueError):
        kv_capacity_sessions(8 * 40 * GIB)


def test_tp_replication_penalty_for_grouped_query_models():
    """Once TP exceeds the KV head count, heads replicate and cluster KV grows with TP."""
    assert kv_heads_per_rank(2, 1) == 2
    assert kv_heads_per_rank(2, 2) == 1
    assert kv_heads_per_rank(2, 4) == 1  # cannot split further -> replicated
    geom = dict(n_layers=48, n_kv_heads=2, head_dim=128)
    at2 = kv_bytes_per_token(**geom, tensor_parallel=2)
    at4 = kv_bytes_per_token(**geom, tensor_parallel=4)
    assert at4 == pytest.approx(2 * at2), "TP=4 must cost 2x the KV/token of TP=2 here"


def test_sliding_window_reduces_kv_proportionally():
    full = kv_bytes_per_token(n_layers=48, n_kv_heads=8, head_dim=128)
    hybrid = kv_bytes_per_token(n_layers=48, n_kv_heads=8, head_dim=128, global_layer_frac=0.25)
    assert hybrid == pytest.approx(full * 0.25)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_global_layer_frac_is_validated(bad):
    with pytest.raises(ValueError):
        kv_bytes_per_token(n_layers=48, n_kv_heads=8, head_dim=128, global_layer_frac=bad)


# ---- the window only saves anything once the context has outgrown it ----------------------
#
# `global_layer_frac` is the limit the per-token cost approaches as context grows without
# bound, not the cost at a stated context. Declaring the limit everywhere under-reports KV by
# up to 1 / global_layer_frac, and under-reporting KV is the direction that over-promises
# concurrent sessions on hardware that cannot hold them.

# The hybrid 26B MoE measured on GB200: 30 layers, 5 of them global, 1,024-token window.
HYBRID = dict(n_layers=30, n_kv_heads=4, head_dim=256)
HYBRID_GLOBAL_FRAC = 1 / 6
HYBRID_WINDOW = 1024


def test_a_context_inside_the_window_leaves_every_local_layer_holding_all_of_it():
    # The case the asymptote gets most wrong. At 903 tokens under a 1,024-token window no
    # layer is capped, so the stack holds full-length KV everywhere and the fraction is 1.0.
    assert effective_layer_frac(HYBRID_GLOBAL_FRAC, HYBRID_WINDOW, 903.35) == pytest.approx(1.0)
    windowed = kv_bytes_per_token(
        **HYBRID,
        global_layer_frac=HYBRID_GLOBAL_FRAC,
        sliding_window_tokens=HYBRID_WINDOW,
        avg_context_tokens=903.35,
    )
    uniform = kv_bytes_per_token(**HYBRID)
    assert windowed == pytest.approx(uniform)
    asymptote = kv_bytes_per_token(**HYBRID, global_layer_frac=HYBRID_GLOBAL_FRAC)
    assert windowed / asymptote == pytest.approx(6.0), "the asymptote is 6x too cheap here"


def test_the_fraction_decays_toward_the_asymptote_and_never_below_it():
    # Monotone in context, bounded by the two regimes: everything full at one end, only the
    # global layers full at the other. A formula that dipped below the asymptote would report
    # KV cheaper than a model that had no local layers at all.
    previous = 1.1
    for context in (1024, 2048, 4096, 8192, 32768, 131072, 10**9):
        frac = effective_layer_frac(HYBRID_GLOBAL_FRAC, HYBRID_WINDOW, context)
        assert HYBRID_GLOBAL_FRAC <= frac <= 1.0, f"context {context} left the band at {frac}"
        assert frac <= previous, f"context {context} cost more than the shorter context did"
        previous = frac
    assert effective_layer_frac(HYBRID_GLOBAL_FRAC, HYBRID_WINDOW, 10**12) == pytest.approx(
        HYBRID_GLOBAL_FRAC, abs=1e-6
    ), "the declared fraction is the limit and must be reached in it"


def test_the_fraction_matches_the_closed_form_between_the_two_regimes():
    # Pins the model, not just its endpoints. Halving the context from 4,096 to 2,048 doubles
    # the local layers' share of it, and both cases must sit on the same curve.
    for context in (2048, 4096, 30000):
        share = min(context, HYBRID_WINDOW) / context
        expected = HYBRID_GLOBAL_FRAC + (1 - HYBRID_GLOBAL_FRAC) * share
        assert effective_layer_frac(HYBRID_GLOBAL_FRAC, HYBRID_WINDOW, context) == pytest.approx(
            expected
        )


def test_uniform_full_attention_is_untouched_by_a_window_declaration():
    # global_layer_frac of 1.0 means there are no local layers to cap, so the window is
    # irrelevant and the cost must not move.
    assert effective_layer_frac(1.0, 512, 32768) == pytest.approx(1.0)


def test_declaring_a_window_without_a_context_raises_instead_of_using_the_asymptote():
    # The silent alternative is the whole defect: the caller said the model is windowed, got
    # the limit anyway, and the inflated KV capacity reads as an ordinary number.
    for partial in ({"sliding_window_tokens": 1024}, {"avg_context_tokens": 8192}):
        with pytest.raises(ValueError, match="together"):
            kv_bytes_per_token(**HYBRID, global_layer_frac=HYBRID_GLOBAL_FRAC, **partial)


def test_omitting_both_leaves_the_declared_fraction_exactly_as_it_was():
    # Backward compatibility for every report published before the correction.
    assert kv_bytes_per_token(**HYBRID, global_layer_frac=HYBRID_GLOBAL_FRAC) == pytest.approx(
        2 * 30 * (1 / 6) * 4 * 256 * 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sliding_window_tokens": 0, "avg_context_tokens": 100},
        {"sliding_window_tokens": -1, "avg_context_tokens": 100},
        {"sliding_window_tokens": 1024, "avg_context_tokens": 0},
        {"sliding_window_tokens": 1024, "avg_context_tokens": -5},
    ],
)
def test_a_degenerate_window_or_context_is_rejected(kwargs):
    # A zero context divides by zero and returns inf, which reads downstream as a KV capacity
    # of zero tokens -- a rejected deployment, reported as though it had been computed.
    with pytest.raises(ValueError):
        effective_layer_frac(HYBRID_GLOBAL_FRAC, **kwargs)
