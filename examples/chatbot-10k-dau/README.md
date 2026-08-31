# `chatbot-10k-dau` — sizing a product forecast against a real measurement

> *"We're launching a chat assistant for 10,000 daily users. How many GPUs do we buy?"*

This is the question the protocol exists to answer, run end to end. It is also the only example
here with **no measurements of its own**: a workload declaration is a product forecast, and the
hardware numbers come from a *different* example — [`moe-26b-h100-tp2`](../moe-26b-h100-tp2/) —
which is where they were actually measured.

That separation is deliberate. Layer 5 (workload) and layer 4 (measurement) compose, and keeping
them in different directories makes it obvious which half you are allowed to reuse and which
half you must re-measure on your own hardware.

| file | what it is |
|---|---|
| `workload.json` | **generated** — the layer-5 declaration, conforming to `workload.schema.json` |
| `build_workload.py` | the parameters, with every `(I)` derived by `ascep.capacity.Workload` |

## The declaration

Chapter 6's worked example (a), verbatim, as a validating artifact:

| field | value | tag |
|---|---|---|
| `daily_active_users` | 10,000 | (U) — a forecast, not telemetry |
| `sessions_per_user_per_day` | 2 | (U) |
| `avg_session_seconds` | 600 | (U) |
| `peak_to_mean` | 4.0 | (U) — single-timezone consumer traffic |
| `duty_cycle` | 0.4 | (U) |
| `input_tokens_per_request` | 1,000 | (U) |
| `output_tokens_per_request` | 400 | (U) |
| `requests_per_session` | 5 | (U) |
| `concurrent_users` | `null` | direct override unavailable pre-launch |
| `target_tok_s_per_user` | `null` | no per-stream anchor fixed; see below |

Everything else in the file is derived:

```
peak_concurrent_users = (10,000 × 2 / 86,400) × 600 × 4   = 555.56
active_sessions       = 555.56 × 0.4                      = 222.22
avg_context_tokens    = 1,000 + 400/2                     = 1,200
demand_tok_s          = 555.56 × (400 × 5) / 600          = 1,851.85
```

## The measured inputs

From `moe-26b-h100-tp2`, at TP=2 on H100 SXM:

- **KV: 574,798 tokens per GPU** (M) — the engine's own figure, per the spec's preference for
  the reported pool over the analytic one.
- **Throughput at 1,200-token context: 1,459 tok/s per GPU** (I from M). The campaign measured
  1,476 at 1,024 tokens and 1,172 at 4,096; `interpolate_throughput` reads 1,200 off that
  curve. It never extrapolates — outside the measured range it clamps to the nearest endpoint
  and you get the endpoint's value, not an invented one.

Note what C4 buys here. The same hardware does 751 tok/s per GPU at 8,192 tokens — **49% less**.
Had this workload been a document-QA assistant instead of a chat assistant, taking the headline
1,476 figure would have overstated the throughput floor by nearly 2×.

## The answer

```python
from ascep.capacity import Workload, gpus_required, interpolate_throughput

need = gpus_required(
    work,  # from workload.json
    kv_tokens_per_gpu=574_798,
    throughput_tok_s_per_gpu=interpolate_throughput(curve, 1_200),
    headroom=1.15,
    gpus_per_replica=2,  # TP=2 — capacity is bought in whole replicas
)
```

**2 GPUs — one replica of TP=2 — binding constraint `throughput`, RECOMMENDED tier.**

| floor | at 2 GPUs, after ÷1.15 headroom | vs. 556 needed |
|---|---|---|
| KV | 2,083 users | 3.7× spare |
| **Throughput** | **761 users** | **1.4× spare — this is the binding one** |

The floors are 2.7× apart, and the protocol makes you say which one you are quoting. If you buy
against the KV floor you buy nothing you need; the next useful dollar here is compute, or a
faster decode path, not memory.

Chapter 6 runs the same workload against *illustrative* per-GPU figures (180,000 KV tokens,
900 tok/s) and answers **4 GPUs**. Same workload, same formulas, half the hardware — because
the real measurement is faster than the placeholder. That gap is the entire argument for
measuring rather than estimating, and it is why the protocol treats per-GPU figures as (M)
inputs you must supply, not constants it can ship.

## What would change this answer

**The throughput input is open-loop.** `moe-26b-h100-tp2` measured with `ignore_eos` at
saturation, and says so: *"This measures engine ceiling, NOT user experience."* An engine
ceiling is an upper bound on what closed-loop traffic will see. So 761 is optimistic, the
1.4× spare margin is the thinnest thing in this table, and the 1.15 headroom divisor is **not**
a substitute for re-measuring closed-loop — headroom absorbs variance, not a systematic bias.
Treat 2 GPUs as the floor of the answer and validate before committing.

**`requests_per_session` = 5 is the assumption to attack first.** Demand is
`peak × output_tokens × requests_per_session ÷ session_seconds`, so this field scales the
throughput floor — the binding one — linearly. At 10 turns per session demand doubles to 3,704
tok/s and the answer becomes **4 GPUs**. A chat product's turn count is also notoriously
optimistic pre-launch, and it is measurable from day one of a beta.

**Context length is the second, and it moves more than the number.** Re-run the same workload
as a document assistant — 8,000-token inputs instead of 1,000 — and two things happen at once:
per-GPU throughput drops to 751 tok/s off the measured curve, and the KV floor falls to 610
users while the throughput floor rises to 784. The binding constraint **flips from `throughput`
to `kv`**, and the answer becomes 4 GPUs bought for a completely different reason. Same product,
same DAU, same model; the shopping list changes. This is the crossover C5 exists to expose.

**`duty_cycle` = 0.4 is the interesting non-answer.** Chapter 6 warns about it harder than any
other workload field, and here it changes nothing: with no per-stream rate target, duty cycle
enters only the KV floor, and the KV floor is not binding. Push it all the way to 1.0 and the
answer is still 2 GPUs — though the KV floor falls to 833 against a throughput floor of 761,
so the two are nearly touching and a small context increase would flip which one you are
buying against. **Which assumptions matter depends on which floor binds.** That is not a
caveat about this example; it is the reason a capacity number without its binding constraint
cannot be acted on.

**`peak_to_mean` = 4.0 matters in one direction only.** Raise it to 6 and you need 4 GPUs.
Drop it to 1.0 — size to the daily mean — and demand collapses to 139 concurrent users and 463
tok/s, but the answer stays at 2 GPUs, because capacity is bought in whole TP=2 replicas and
one replica is the floor. The under-sizing is real and simply has nowhere to go here; on a
larger deployment the same mistake buys a fraction of the peak-hour hardware.

## Reproducing it

```bash
python examples/chatbot-10k-dau/build_workload.py   # regenerates workload.json
pytest tests/test_application_sizing.py             # recomputes every figure above
```
