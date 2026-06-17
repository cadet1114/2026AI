# Manual Emergency Command Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a fixed-screen Streamlit command console with a deterministic complex rescue graph, manual phase controls, and visible calculation evidence for every algorithm.

**Architecture:** Keep scenario generation, route/allocation tracing, simulation orchestration, Plotly rendering, and Streamlit composition in their existing modules. Add optional trace data to algorithm APIs so existing CLI/pipeline callers remain compatible, and store only JSON-compatible records in `LiveSimulation` session state.

**Tech Stack:** Python 3.11+, Streamlit, Plotly Graph Objects, pandas, pytest.

---

### Task 1: Expand The Deterministic Rescue Graph

**Files:**
- Modify: `tests/test_random_scenario.py`
- Modify: `src/emergency_commander/random_scenario.py`

- [ ] **Step 1: Write failing size and resilience tests**

Add assertions that a generated scenario has six zones, at least twenty-seven total nodes, thirty to forty ground roads, six to nine air routes, three rescue cars, and two drones. Retain deterministic seed equality and test reachability after blocking each non-hospital road.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `pytest tests/test_random_scenario.py -q`

Expected: failure because the current generator returns three zones, nine ground roads, and three units.

- [ ] **Step 3: Implement the expanded graph**

Generate a fixed topological template with seed-jittered coordinates and risks:

```python
ZONE_IDS = ("A", "B", "C", "D", "E", "F")

district_nodes = {
    f"J{row}{column}": {
        "x": column * 4.0 + rng.uniform(-0.35, 0.35),
        "y": row * 3.2 + rng.uniform(-0.35, 0.35),
    }
    for row in range(4)
    for column in range(5)
}
```

Connect horizontal, vertical, selected diagonal, hospital, zone-spur, and air-relay edges. Assign deterministic corridor risk profiles so short high-risk and longer low-risk alternatives coexist. Add `RescueCar-3` and `Drone-2` with different speed/capacity/cost values.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `pytest tests/test_random_scenario.py -q`

Expected: all random scenario tests pass.

### Task 2: Add Explainable A* And Allocation Traces

**Files:**
- Modify: `tests/test_routing_v2.py`
- Modify: `tests/test_routing_allocation.py`
- Modify: `src/emergency_commander/routing.py`
- Modify: `src/emergency_commander/allocation.py`

- [ ] **Step 1: Write failing A* trace tests**

Call `risk_aware_astar(..., include_trace=True)` and assert:

```python
assert result["search_trace"]
assert result["search_trace"][-1]["node"] == "goal"
assert all(row["f"] == pytest.approx(row["g"] + row["h"]) for row in result["search_trace"])
```

- [ ] **Step 2: Run the A* test and confirm RED**

Run: `pytest tests/test_routing_v2.py -q`

Expected: `include_trace` is not accepted.

- [ ] **Step 3: Implement optional A* tracing**

Add `include_trace: bool = False`. For every accepted expansion append a JSON-compatible record containing `node`, rounded `g`, `h`, `f`, `frontier_size`, and accepted neighbor relaxations. Include `search_trace` only when requested.

- [ ] **Step 4: Write failing allocation trace tests**

Call `allocate_tasks(..., include_trace=True)` and assert the returned object contains assignments plus a trace whose winning total equals the assignment utility sum.

- [ ] **Step 5: Run the allocation test and confirm RED**

Run: `pytest tests/test_routing_allocation.py -q`

Expected: `include_trace` is not accepted.

- [ ] **Step 6: Implement optional route and allocation trace propagation**

Add `include_trace` to `build_utility_matrix` and `allocate_tasks`. Preserve list return values by default; when allocation tracing is requested return:

```python
{
    "assignments": assignments,
    "trace": {
        "considered": considered,
        "duplicate_zone_rejections": duplicate_zone_rejections,
        "ranked_combinations": ranked[:12],
        "winning_total": round(best_total, 6),
    },
}
```

- [ ] **Step 7: Run focused routing/allocation tests and confirm GREEN**

Run: `pytest tests/test_routing_v2.py tests/test_routing_allocation.py -q`

Expected: all tests pass.

### Task 3: Record Manual Calculation History And Transition Execution

**Files:**
- Modify: `tests/test_live_simulation.py`
- Modify: `src/emergency_commander/live_simulation.py`

- [ ] **Step 1: Write failing manual history tests**

Assert a new session remains at `validate`, each `step()` appends exactly one record with the common envelope, and serialization preserves records.

- [ ] **Step 2: Write failing execution delta tests**

After allocation, call `step(execution_minutes=1.0)` and `step(to_next_transition=True)`. Assert the clock advances by the requested minute and then by the minimum positive active transition duration.

- [ ] **Step 3: Run live simulation tests and confirm RED**

Run: `pytest tests/test_live_simulation.py -q`

Expected: missing `calculation_history`, execution arguments, and transition helper.

- [ ] **Step 4: Implement phase-specific records**

Add `calculation_history` and `_record_calculation(...)`. Record:

```python
{
    "index": len(self.calculation_history) + 1,
    "phase": phase,
    "title": title,
    "clock_minutes": round(self.clock_minutes, 6),
    "summary": summary,
    "focus": focus,
    "inputs": inputs,
    "operations": operations,
    "outputs": outputs,
}
```

Populate validation counts/diffs, Bayesian evidence/posteriors/contributions, priority weighted terms, A* traces, utility breakdowns, allocation enumeration, execution before/after states, and replan context.

- [ ] **Step 5: Implement explicit execution deltas**

Change `step` to accept keyword-only `execution_minutes` and `to_next_transition`. Compute the next transition from positive `remaining_travel` and `remaining_service` values in active unit states. Call `advance_unit_states` once with the chosen delta.

- [ ] **Step 6: Run live simulation tests and confirm GREEN**

Run: `pytest tests/test_live_simulation.py -q`

Expected: all tests pass.

### Task 4: Upgrade Plotly Map For Presentation

**Files:**
- Modify: `tests/test_visualization.py`
- Modify: `src/emergency_commander/visualization.py`

- [ ] **Step 1: Write failing pre-inference and highlight tests**

Build a map from a new session snapshot with no assessments and assert the figure renders all roads/zones/units. Build a traced snapshot and assert selected roads and focused objects create highlight traces.

- [ ] **Step 2: Run visualization tests and confirm RED**

Run: `pytest tests/test_visualization.py -q`

Expected: current map requires assessments and initialized unit states.

- [ ] **Step 3: Implement presentation map layers**

Support raw scenario fallback positions, neutral hazard-derived zone colors, risk halos, priority rank labels, blocked-road midpoint `X` markers, candidate route overlays, selected route emphasis, and calculation focus highlights. Set a compact horizontal legend and fixed responsive height.

- [ ] **Step 4: Run visualization tests and confirm GREEN**

Run: `pytest tests/test_visualization.py -q`

Expected: all visualization tests pass.

### Task 5: Build The Fixed-Screen Manual Streamlit Console

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Remove automatic advancement**

Delete `st.fragment(run_every=...)`, speed controls, automatic `session.step()` calls, and pause/resume controls. `start_random_session()` only creates and saves the session.

- [ ] **Step 2: Add manual controls and history navigation**

Add callbacks for next phase, advance one minute, advance to next transition, history previous/next, and event injection. Keep the selected history index in Streamlit session state without mutating simulation history.

- [ ] **Step 3: Build the fixed viewport composition**

Use CSS with `height: 100vh`, hidden document overflow, compact header/footer, and fixed map/inspector heights. Compose map, attached event dock, and calculation inspector in one main row. Use internal overflow only inside the inspector.

- [ ] **Step 4: Render phase-specific calculations**

Render readable formula cards, evidence contribution bars/tables, A* expansion table, utility waterfall/table, allocation rankings, execution deltas, and replanning comparison from the selected history record.

- [ ] **Step 5: Preserve final downloads**

Keep JSON and Markdown result downloads in a compact final-result inspector instead of lower-page tabs.

- [ ] **Step 6: Run syntax and full automated tests**

Run: `python -m py_compile app.py src/emergency_commander/*.py`

Run: `pytest -q`

Expected: compilation succeeds and all tests pass.

### Task 6: Browser Acceptance And Final Polish

**Files:**
- Modify as needed: `app.py`
- Modify as needed: `src/emergency_commander/visualization.py`

- [ ] **Step 1: Start Streamlit**

Run: `streamlit run app.py --server.headless true --server.port 8501`

- [ ] **Step 2: Verify the initial viewport**

At 1440 x 900, confirm there is no document-level vertical scrollbar and the map, event dock, inspector, and footer are visible together.

- [ ] **Step 3: Verify the complete manual workflow**

Generate a map, click through validation, inference, priority, routing, utility, allocation, and execution. Confirm each click creates one inspector record and no phase advances automatically.

- [ ] **Step 4: Verify event-driven replanning**

Trigger a collapse on an active route, confirm the road displays blocked styling, and manually advance through replanning until the selected route changes.

- [ ] **Step 5: Capture acceptance screenshots and run final tests**

Save screenshots under `output/playwright/manual-command-console/`, then rerun `pytest -q` and `git diff --check` before reporting completion.
