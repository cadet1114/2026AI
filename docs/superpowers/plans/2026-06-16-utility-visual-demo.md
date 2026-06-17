# Explainable Utility and Visual Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resource-aware, fully decomposed expected utility and expose it through a risk-aware Streamlit decision console.

**Architecture:** `allocation.py` remains the sole source of utility calculations. The pipeline carries candidate and scenario-state evidence into its JSON contract, and `visualization.py` plus `app.py` render that evidence without duplicating decision formulas.

**Tech Stack:** Python 3.11+, JSON Schema Draft 2020-12, pytest, Streamlit, pandas, Plotly.

---

### Task 1: Specify resource-aware utility behavior

**Files:**
- Modify: `tests/test_routing_allocation.py`
- Modify: `tests/test_json_contracts.py`

- [ ] **Step 1: Add a failing breakdown test**

Add assertions that a feasible candidate contains `resource_cost`, `utility_inputs`, and all six signed `utility_breakdown` terms, and that their sum equals `expected_utility`.

- [ ] **Step 2: Add a failing resource penalty test**

Build two otherwise identical units with different `resource_cost` values and assert the lower-cost unit has higher expected utility by `zeta * cost_difference`.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_routing_allocation.py tests/test_json_contracts.py -q`

Expected: FAIL because the new fields and schema properties do not exist.

### Task 2: Implement and publish the utility contract

**Files:**
- Modify: `src/emergency_commander/input_adapter.py`
- Modify: `src/emergency_commander/allocation.py`
- Modify: `src/emergency_commander/simulation.py`
- Modify: `src/emergency_commander/pipeline.py`
- Modify: `schemas/scenario.schema.json`
- Modify: `schemas/decision_output.schema.json`
- Modify: `examples/scenario_input.json`

- [ ] **Step 1: Normalize new inputs**

Add optional nonnegative `zeta` to utility weights with default `0.10`, and optional `[0,1]` `resource_cost` to units with defaults `0.55` for rescue cars and `0.25` for drones.

- [ ] **Step 2: Calculate structured contributions**

For feasible candidates, emit:

```python
breakdown = {
    "trapped_benefit": alpha * trapped_prob,
    "life_risk_benefit": beta * life_risk,
    "accessibility_benefit": gamma * accessibility,
    "arrival_time_cost": -delta * arrival_time_normalized,
    "path_risk_cost": -epsilon * path_risk,
    "resource_cost": -zeta * unit_resource_cost,
}
```

Set `expected_utility` to the sum and produce a deterministic Chinese explanation from the same values.

- [ ] **Step 3: Preserve decision evidence**

Carry breakdowns into active tasks and selected assignments. Add `utility_matrix` to plans and `scenario_state` to timeline snapshots.

- [ ] **Step 4: Update JSON Schemas**

Publish the new input and output fields with closed-object validation.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_routing_allocation.py tests/test_json_contracts.py tests/test_pipeline_replanning.py -q`

Expected: PASS.

### Task 3: Specify visual decision evidence

**Files:**
- Modify: `tests/test_visualization.py`

- [ ] **Step 1: Add failing builder tests**

Require `build_utility_frame`, `build_utility_contribution_figure`, and risk-band traces named `Ground · Low risk`, `Ground · Medium risk`, `Ground · High risk`, or `Ground · Blocked` as applicable.

- [ ] **Step 2: Add failing Streamlit assertions**

Require an `效用决策` tab and event navigation buttons.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_visualization.py -q`

Expected: FAIL because the builders and UI controls do not exist.

### Task 4: Implement the visual decision console

**Files:**
- Modify: `src/emergency_commander/visualization.py`
- Modify: `app.py`

- [ ] **Step 1: Render road risk bands**

Group ground roads by weighted risk and blocked status; render separate Plotly traces with distinct colors and widths.

- [ ] **Step 2: Add utility builders**

Return a candidate DataFrame and a Plotly waterfall whose bars use the signed `utility_breakdown` values.

- [ ] **Step 3: Add the utility decision tab**

Show assignment summaries, the full candidate matrix, a selected candidate waterfall, and its deterministic explanation.

- [ ] **Step 4: Add event navigation**

Keep slider state in `st.session_state`, add previous/next controls, and render `snapshot["scenario_state"]` on the map.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_visualization.py -q`

Expected: PASS.

### Task 5: Synchronize artifacts and verify end to end

**Files:**
- Modify: `examples/decision_output_fixed_v2.json`
- Modify: `examples/decision_output_learned_v2.json`
- Modify: `README.md`
- Modify: `agent.md`

- [ ] **Step 1: Regenerate fixed and learned example outputs**

Run both CLI modes against `examples/scenario_input.json` and write the synchronized v2 files.

- [ ] **Step 2: Run the full automated suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Start and inspect Streamlit**

Run: `.venv/bin/streamlit run app.py --server.headless true`

Verify in a browser that the initial step shows utility evidence and the road-collapse step shows the blocked road and replanned route.

- [ ] **Step 4: Review contract completeness**

Validate both regenerated outputs with `schemas/decision_output.schema.json` and confirm breakdown totals match expected utility for every feasible candidate and assignment.
