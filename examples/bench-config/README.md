# `examples/bench-config` — a runnable `ascep bench` input set

A complete input set for `ascep bench`: the harness config `bench.json` plus the four
declaration documents it points at. The declarations are copied unchanged from the worked
example [`examples/moe-26b-h100-tp2`](../moe-26b-h100-tp2/) — a 26B MoE on 2× H100, vLLM at
tensor parallel 2.

This directory is meant to be copied and edited, not run as-is against your endpoint. Every
file in it describes a machine that is not yours, and a report produced that way is bound
under **C3** to a topology that was never measured.

## Running it

```bash
ascep bench bench.json --dry-run   # validates everything, prints the plan, writes nothing
ascep bench bench.json             # runs the ladder, writes the bundle and a draft report
```

The dry run parses and schema-checks the config and all four declarations, then prints the
endpoint, the rungs, the window count and the minimum wall clock the plan implies. As shipped
that is 22 windows and **7,920 s ≈ 2 h 12 m** of steady state, before warm-up and request
bodies. Size the batch allocation against it.

## Files

| file | what it declares |
|---|---|
| `bench.json` | the harness config: endpoint, declaration paths, workload, window, ladder, SLO gates, outputs |
| `hardware.json` | GPUs, interconnect, host (**C3**) |
| `model.json` | parameters, architecture, precision, attention geometry (**C3**) |
| `serving.json` | engine, parallel widths, GPU count, KV pool (**C3**) |
| `workload.json` | the application workload the capacity question is asked about |

## Paths resolve against two different directories

This catches people, so it is worth stating plainly:

- **`declarations.*` and `workload.corpus`** resolve relative to **the config file's
  directory**, so a config and its declarations move between machines together.
- **`output.bundle_dir` and `output.report_path`** resolve relative to **the current working
  directory**, because they are where *this* invocation puts its results, not part of the
  declaration being replayed.
- **`output.engine_logs_path`** resolves relative to the bundle's *parent* directory and must
  resolve to a file underneath it. As shipped, `bundle_dir: "runs/bundle"` and
  `engine_logs_path: "engine.log"` mean `runs/engine.log`. A log outside that tree would have
  to be named in the reproduction table with a path that resolves only on the machine that ran
  the benchmark, so `ascep bench` refuses it before the first request rather than at write
  time, when the records exist nowhere but in RAM.

## What you must change

- **`endpoint.base_url`** — the **server root**, not the API route. The adapter appends
  `/v1/chat/completions` itself; `http://host:8000/v1` is refused, because it would request
  `/v1/v1/chat/completions` and score every 404 as a server error.
- **`endpoint.model`** — the model id the endpoint answers to.
- **All four declaration documents.** They describe someone else's machine. The manifest
  hashes their bytes, so publishing a report bound to these files while running on different
  hardware is a misbinding that stays checkable after publication — which is the point of
  **C3**, and it will be checked.
- **`output.bundle_dir` and `output.report_path`** — your own results tree.
  `ascep bench` refuses to overwrite an existing bundle: the GPU hours behind one are already
  spent and its records cannot be regenerated, so a second run at the same path exits without
  measuring. Choose a new directory per run.
- **`output.engine_logs_path`** — a readable file holding the engine's own log (**C8**). It is
  the only record of the run written by the server rather than by the load generator. If the
  engine wrote none, say so in a file and point at that.
- **The `slo_gates` block.** The gates are a product decision, not a measurement, and **C7**
  requires them fixed before the first request: `declared_before_run` must be literally `true`,
  and at least one gate must be non-null. Four nulls means every window passes by definition
  and the sustainable tier becomes the measured tier wearing an SLO label, so it is refused.

## About the shipped workload

`corpus: "synthetic"` builds prompts by padding filler words to `input_tokens`, so
`input_tokens: 1500` is a **whitespace word count**, not a tokenizer count. That is deliberate:
a characters-per-token oracle moves in jumps and would make the target unreachable, whereas one
word per token lands on it by construction. What it costs is that `run.tokenizer` stays null
with a reason and the results table publishes the **server's own** token counts. Point `corpus`
at a JSONL file (prompts read from its `messages` field) to replay real traffic instead.

`ignore_eos: true` fixes output length at `output_tokens`, which is what makes throughput
comparable across rungs. Set it false to let the model decide, and expect the output-token
column to become a distribution.

## What comes out

A reproduction bundle at `bundle_dir` and a **draft** report at `report_path`. The draft
claims `non-conforming`, and stays that way until `ascep conformance` says otherwise — that
word grades the report, not the hardware. A load generator observes latency and throughput
over HTTP and nothing else, so the roofline comparison, the sizing result, the scaling table
and the theoretical and recommended tiers are left null with reasons rather than estimated.
Filling them from latency alone is how a KV-bound deployment gets sized against a throughput
number.

```bash
ascep validate  runs/report.json     # schema-check the draft
ascep conformance runs/report.json   # grade it against C1–C11
ascep render    runs/report.json -o report.md
```

Exit codes are worth wiring into the batch wrapper: `0` clean, `1` nothing was written,
`2` refused before measuring, `3` ran but did not complete as declared. `3` is deliberately not
`0` — a truncated ladder that exited 0 gets swept into the results directory beside the
complete ones, and the caveat that its concurrency figures are a lower bound survives only in
a report nobody re-reads. See [chapter 7 §10](../../protocol/07-benchmark-procedure.md).
