# A measured tier below the sustainable tier derived from it

**Rule broken:** C6
**The one edit:** `capacity_tiers.measured.max_concurrent_users` is set to `1.6384`

Dividing measured capacity by ten puts the measured tier below the sustainable tier built on top of it, which is impossible: sustainable is measured throughput with the SLO gate applied, so it can only ever be equal to or lower than the number it was derived from. An inverted ordering means at least one of the two figures came from somewhere other than where it claims, and a reader comparing tiers can no longer tell which. The dangerous variant of this failure is the reverse direction, where a "measured" number is inflated to justify an aggressive sustainable figure and someone provisions against capacity no run ever produced. The case grades partial rather than non-conforming, which is a property of the grading ladder and not a judgement about severity: only C1 through C5 errors force non-conforming, so a C6 contradiction caps the report instead of failing it.

## Reproduce

```bash
ascep conformance examples/negative/c6/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c6/report.json)
```
