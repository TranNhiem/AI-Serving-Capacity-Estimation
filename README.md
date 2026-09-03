# ASCEP — AI Serving Capacity Estimation Protocol

**A reproducible, vendor-neutral standard for answering: *I have this hardware, I want to
serve this model, I want to build this application — how much capacity do I get, and how
much infrastructure do I need?***

[![Protocol](https://img.shields.io/badge/ASCEP-v0.5--draft-blue)](protocol/SPEC.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## The problem

Ask three teams how many GPUs a model needs and you get three answers, none reproducible.
Not because the teams are careless, but because the field reports capacity in a form that
cannot be checked:

- **"It does 2,500 tokens/s."** At what context length? Throughput falls 2–4× from chat-length
  to document-length prompts. A single number is not a property of a model.
- **"It fits 147 sessions per GPU."** At what tensor-parallel width? KV capacity per GPU is not
  a constant — it can vary by **6×** with topology, and it can go *down* when you add GPUs,
  because KV heads get replicated once TP exceeds the model's KV head count.
- **"We benchmarked it at 400 users."** Open-loop with `ignore_eos`, or closed-loop with real
  think time? Those measure different quantities and differ by multiples.
- **"P50 latency is 200 ms."** Capacity is set by the tail. P50 is decoration.

Every one of those gaps is cheap to close at measurement time and expensive to discover
after you have signed for the hardware.

## What ASCEP is

A protocol — a set of declarations, gates, formulas and a report format — plus a small
reference implementation.

- **`protocol/`** — the normative spec. [Start with SPEC.md](protocol/SPEC.md).
- **`schemas/`** — JSON Schema for the five layers: hardware, model, serving, run, workload.
- **`ascep/`** — reference implementation. `capacity.py` is the transparent formula set and is
  **stdlib-only by design**, so it runs on an air-gapped cluster login node with no `pip install`;
  `conformance.py` grades a report against C1–C12, `render.py` emits the Markdown form.
- **`templates/capacity-report.md`** — the standard report.
- **`examples/`** — worked end-to-end reports you can diff your own against.

**ASCEP is not** a leaderboard, a quality benchmark, or a procurement recommendation. It
measures capacity. A model that serves 10,000 users badly will pass ASCEP with flying colours;
pair it with a task-appropriate quality evaluation.

## The core ideas

**Capacity is the minimum of four floors, and which one binds changes with context length and prompt shape.**

| floor | formula | binds when |
|---|---|---|
| Weights | do weights + a usable KV pool fit? | always checked first; binary |
| KV | `sessions = kv_tokens ÷ avg_context` | long context |
| Prefill | `users = usable_prefill_tok_s ÷ per-user prompt demand` | prompts heavier than the run the throughput figure came from |
| Throughput | `users = usable_tok_s ÷ per-user demand` | short context |

A capacity number without its **binding constraint** is not actionable — it doesn't tell you
what to buy more of. ASCEP requires naming it.

**Four tiers, never interchanged.** Theoretical (roofline) → Measured → Sustainable
(SLO-gated) → Recommended (with headroom). They differ by integer multiples, and a reader
given one number will assume the most favourable.

**Every number is tagged.** **(M)** measured · **(I)** inferred by a named formula ·
**(T)** theoretical · **(U)** unmeasured assumption. Untagged numbers make a report
non-conforming. This one rule does more for trustworthiness than any amount of extra
benchmarking.

## Quick start

```bash
pip install -e .
```

Estimate from a workload description, no GPUs required:

```python
from ascep.capacity import Workload, capacity_at, gpus_required

# "A chat assistant for 10,000 daily users."
work = Workload(
    daily_active_users=10_000,
    sessions_per_user_per_day=2,
    avg_session_seconds=600,
    peak_to_mean=4.0,  # peak hour vs daily average
    duty_cycle=0.4,  # fraction of a session actually generating
    input_tokens_per_request=1_000,
    output_tokens_per_request=400,
    requests_per_session=5,
)

print(f"{work.peak_concurrent_users():.0f} concurrent users at peak")  # 556
print(f"{work.demand_tok_s():,.0f} tok/s aggregate demand")  # 1,852

# Per-GPU figures MUST come from a measurement at the same TP width AND the same context
# length — both of these are H100 numbers from examples/moe-26b-h100-tp2 at ~1,200 tokens.
need = gpus_required(
    work,
    kv_tokens_per_gpu=574_798,
    throughput_tok_s_per_gpu=1_459,
    headroom=1.15,
    gpus_per_replica=2,  # TP=2 — capacity is bought in whole replicas
)
print(f"{need.n_gpus} GPUs, bound by {need.binding_constraint.value}")  # 2 GPUs, throughput
```

Or go the other way — given hardware, what can it serve?

```python
cap = capacity_at(n_gpus=8, kv_tokens=574_798 * 8, throughput_tok_s=1_459 * 8, workload=work)
print(f"{cap.max_concurrent_users:.0f} users · bound by {cap.binding_constraint.value}")
# 3502 users · bound by throughput   (the KV floor is 9,580 — nearly 3× higher)
```

[`examples/chatbot-10k-dau`](examples/chatbot-10k-dau/) works this through end to end, including
what has to be true for the answer to hold and which assumption breaks it first.

Then grade and publish the result:

```bash
ascep init -o report.json           # a fillable skeleton of every field the schemas require
ascep validate report.json          # structure and vocabulary, against the schemas
ascep conformance report.json       # C1–C12, and whether the report overstates itself
ascep conformance report.json --raise   # ...and save that level into the file
ascep render report.json -o report.md

# Sizing straight from a workload declaration. Both measured figures are PER GPU and must
# come from a run at the same tensor-parallel width you will deploy at (C3) — passing the
# aggregate figures for a TP=2 replica here would silently double the answer.
ascep size --workload examples/chatbot-10k-dau/workload.json \
           --kv-tokens 574798 --throughput-tok-s 1459 --gpus-per-replica 2
# peak_concurrent_users: 555.56
# demand_tok_s: 1851.85
# gpus: 2
# binding_constraint: throughput
```

`ascep conformance` is the one to run before you show anyone a number. It reports the level it
computes — conforming, partial or non-conforming — alongside the level the report *claims*, and
says so loudly when those differ. It writes nothing unless you pass `--raise`, which saves a
computed level *stronger* than the claim into the file so the artifact carries its own grade.
It will not move a claim the other way: an overstated report is reported and left alone, since
the fix there is the author's to make.

`ascep init` deliberately produces a document that does **not** validate. Every value is `null`
or `TODO`, so the validation errors are the fill-in list, and the two places where the schemas
demand a choice — a workload's sizing basis, a measurement point's context length — are printed
by name rather than guessed at. A skeleton that validated would be one that had invented a
`gpu_count` on your behalf, and a half-filled report claiming a single-GPU deployment is exactly
the confusion the rest of this protocol exists to prevent.

Only `validate` and `bench` need the optional extras (`pip install 'ascep[run]'`, for
`jsonschema` and `httpx`). The rest run on a bare install, because the machines where capacity
questions get asked are often the ones where you cannot install anything.

`ascep bench` runs the chapter 7 procedure against any OpenAI-compatible endpoint: a declared
concurrency ladder, three repetitions a rung, SLO gates fixed before the first request, a
reproduction bundle, and a draft `report.json`.

```bash
ascep bench bench.json --dry-run   # the plan, the window count, the wall clock it implies
ascep bench bench.json
```

It reads your `hardware.json`, `model.json`, `serving.json` and `workload.json` rather than
inventing a topology, and it schema-checks all four before the first request goes out --
discovering after four hours of Slurm time that `serving.json` was malformed is discovering it
too late. What it emits is a *draft*: a load generator sees latency over HTTP and nothing else,
so the roofline, the sizing result, the scaling table and the theoretical and recommended tiers
are left null with reasons, and the report claims `non-conforming` until `ascep conformance`
says otherwise — `--raise` is what writes that verdict back into the file.

[`examples/bench-config/`](examples/bench-config/) is a complete input set — a `bench.json`
plus the four declarations it binds to — meant to be copied and edited; every key it can
contain, every refusal it can trigger, and what each exit code means are specified in
[chapter 7 §10](protocol/07-benchmark-procedure.md). If you would rather keep your own harness,
chapter 7 specifies the procedure and not the tool: `examples/*/build_report.py` shows how to
map results from it onto the schema instead. See **Status** below.

## Applying it to your stack

ASCEP is deliberately thin on assumptions. It was designed against NVIDIA H100 + vLLM, from a
campaign covering both a dense and a Mixture-of-Experts model, but nothing in the protocol is
specific to those. Be aware of the distinction: **one** worked report is published so far
(`examples/moe-26b-h100-tp2`, the MoE half), so every row below is supported by the schemas and
the formulas but not yet demonstrated by a second published example. Reports that close those
rows are the contribution this project most wants. To adapt it:

| you have | what to do |
|---|---|
| A different framework | declare `serving.framework` and report the engine's own KV-pool figure; the protocol never assumes vLLM |
| A different accelerator | declare it in `hardware.schema.json`; only the roofline needs vendor specs |
| A quantized model | declare stored precision and overhead fraction; formulas already handle sub-byte formats |
| An MoE model | declare **total** params for memory, **active** params for compute — the most common sizing error |
| An MLA model (DeepSeek-style) | declare `attention_type: mla` + `kv_lora_rank`/`qk_rope_head_dim`; the GQA formula overstates its KV by ~57× |
| A Mamba/SSM/linear-attention model | declare `kv_bytes_per_sequence`; its state is per-sequence, so it never hits a long-context KV cliff |
| Multi-node | declare inter-node fabric; report scaling efficiency across the node boundary separately |

**Attention family is a first-class declaration, not metadata.** It selects which KV formula
applies, and the schema refuses a model whose geometry doesn't match the family it claims.
Getting it wrong is an order-of-magnitude error: sizing a DeepSeek-V3-geometry model with the
grouped-query formula predicts **3** concurrent sessions on eight GPUs at a 32,000-token
context where the real answer is **153** — enough to reject a deployable model as undeployable.

## Contributing

Community reports are the point. If you run ASCEP on hardware or a model that isn't covered,
open a PR adding your report under `examples/` — a conforming report from a configuration
nobody has published is more valuable to this project than a code change.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

**v0.5, draft.** The spec and the formula set are stable enough to use and argue with. The
harness is generalized from a private benchmark campaign and now ships; expect churn in
`ascep/` before v1.0. Breaking changes to anything that would alter a conforming report's
numbers get a major version bump. The 0.4 release was one; this one is not. Everything v0.5 adds — workload
archetypes, the prefill floor, the agent-loop declarations — is opt-in and number-preserving:
omit the new fields and every formula returns exactly what it returned before, which is the
property the compatibility promise is made of and which the test suite asserts directly.

One consequence is worth stating plainly, because it is not a number change and so does not
force a major bump, but it can still move a label. C11 grades a published capacity figure
against the token mix of the benchmark rung it was read off, and a 0.4 report whose declared
workload is substantially more input-heavy than its own measurements will now grade `partial`
rather than `conforming`. Nothing about that report changed; the checker gained the ability to
notice an overstatement it was previously blind to. Both published examples grade exactly as
they did under 0.4. See [CHANGELOG.md](CHANGELOG.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
