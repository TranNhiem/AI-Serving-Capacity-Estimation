# A chat context estimator applied to a tool-calling loop

**Rule broken:** C10
**The one edit:** `workload.archetypes` is set to `["code_agent"]`, `workload.agent_loop` is set to `{"turns_per_session": 24.0, "tool_calls_per_turn": 1.8, "compaction_resume_tokens": 12000.0, "session_max_context_tokens": 200000.0}`

Setting `workload.archetypes` to code_agent adds the `workload.agent_loop` object the archetype requires -- 24 turns, 1.8 tool calls per turn, 12,000 resume tokens, a 200,000-token ceiling -- so the workload reads as a tool-calling loop. The agent_loop is there so C9, which requires it, stays quiet and the case exercises C10 alone; nothing was mistyped. Both defects are pre-archetype defaults, correct for chat and wrong for a loop -- estimators nobody re-derived.

The error sits at `workload.avg_context_tokens_tag`. Under code_agent the context grows with every loop turn, so an (I) tag citing no accumulating estimator is the chat single-request mean applied to a transcript that accumulates, and every capacity figure built on the KV floor moves with it, so a mis-priced context is an error. The remedy is an estimator referencing `requests_per_session` or `context_growth_tokens_per_turn` in the workload notes, or a measured mean tagged (M).

The warning sits at `workload.kv_residency`. With `duty_cycle` 0.5 and no declared residency, the model credits each idle session with releasing its blocks; a session waiting on a tool call still holds them, so the default inflates the KV floor's user count by 1/duty_cycle, 2.00 here. It stays a warning because every report written before the field existed computes that way and has to keep working -- the warning, not the default, stops it running silent. Being outside C1 through C5, the C10 error caps the case at partial rather than failing it.

## Reproduce

```bash
ascep conformance examples/negative/c10/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c10/report.json)
```
