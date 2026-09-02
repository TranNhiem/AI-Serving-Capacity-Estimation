# A capacity number carried across a change of token mix

**Rule broken:** C11
**The one edit:** `run.results.2.input_tokens` is set to `2048`, `run.results.2.output_tokens` is set to `2048`

The throughput floor is denominated in generated tokens only, so it is blind to how much prefill the engine paid while producing them. The edit keeps rung 2 at the same 4,096-token context but re-measures it at 2,048 input and 2,048 output, a 1.00 mix, while the workload still declares 3,584:1,024, a 3.50 mix. The checker therefore raises one C11 warning at `run.results.2.prefill_tok_s`, naming the measured, sustainable and recommended tiers that read their numbers from that rung and reporting declared 3.50 against measured 1.00 as a 3.50 factor, above the 1.5 threshold.

Carrying a generated-token rate across that change of mix overstates capacity by exactly that ratio: a report promising roughly 16 concurrent users is promising a load the engine cannot prefill.

The threshold is not leniency. A rung measured at the declared mix was already paying the declared prefill cost while it generated, so `output_tok_s` embeds it and the floor is sound; prefill becomes load-bearing only when a number crosses mixes. Recording `run.results.2.prefill_tok_s` computes that floor instead of assuming it away. It stays a warning, so the case grades partial: omitting the field preserves the pre-prefill answer, and partial is the price of compatibility.

## Reproduce

```bash
ascep conformance examples/negative/c11/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c11/report.json)
```
