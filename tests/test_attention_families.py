"""The three attention families scale KV differently, and confusing them is a sizing disaster.

These tests pin the distinctions that `kv_bytes_per_token` alone cannot express.
"""

import pytest

from ascep.capacity import (
    GIB,
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
