# SLO gates chosen after the results were in

**Rule broken:** C7
**The one edit:** `run.slo_gates.declared_before_run` is set to `false`

Every threshold value is unchanged, and every SLO rung still passes exactly as it did in the baseline. What changed is the declared_before_run flag: the gates were chosen after the latency numbers were already on screen. A gate picked that way is a description of the run, not a test of it, because you can always set the limit one notch above whatever the engine happened to produce -- a failing run becomes passing by moving the line, not by changing the result.

A reader comparing two published configs cannot tell whether "all SLO gates passed" means the system met a target or the target was fitted to the system. The grade drops to partial: the measurements themselves are still trustworthy, but the pass/fail claim they support is not.

## Reproduce

```bash
ascep conformance examples/negative/c7/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c7/report.json)
```
