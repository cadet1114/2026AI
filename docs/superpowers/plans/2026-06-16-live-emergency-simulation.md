# Live Emergency Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-click Streamlit rescue simulation that generates a reproducible random map, automatically advances real algorithms, accepts live emergency events, replans from current state, and emits a final report.

**Architecture:** Add a deterministic scenario generator and a serializable `LiveSimulation` state machine under `src/emergency_commander/`. Reuse the existing Bayesian, risk-aware A*, utility allocation, event application, and finite-state simulation functions, while adding delivery-level rescue accounting. Keep Streamlit responsible only for session persistence, timed stepping, controls, and rendering.

**Tech Stack:** Python 3.11+, Streamlit 1.58 fragments, Plotly, pytest, Streamlit AppTest, Playwright CLI.

---

## File Map

- Create `src/emergency_commander/random_scenario.py`: deterministic, contract-valid, single-road-failure-resilient map generation.
- Create `src/emergency_commander/live_simulation.py`: serializable phase machine, event target selection, incremental execution, completion and reporting.
- Modify `src/emergency_commander/simulation.py`: track delivered rescue zones, rescued people, travel time and origin zone across hospital return.
- Modify `src/emergency_commander/pipeline.py`: allow live planning to exclude globally rescued zones without changing batch defaults.
- Modify `src/emergency_commander/visualization.py`: expose algorithm-stage and result-report data builders where useful.
- Replace `app.py`: live command workspace using Streamlit's timed fragment API and session-state dictionaries.
- Create `tests/test_random_scenario.py`: generator determinism, contract, graph redundancy and air reachability.
- Create `tests/test_live_simulation.py`: phase sequence, execution, events, pause/terminal behavior and reports.
- Modify `tests/test_simulation.py`: hospital delivery accounting regression tests.
- Modify `tests/test_visualization.py`: new Streamlit controls and one-click session behavior.
- Modify `README.md` and `agent.md`: run and acceptance instructions.

### Task 1: Deterministic Random Rescue Maps

**Files:**
- Create: `src/emergency_commander/random_scenario.py`
- Create: `tests/test_random_scenario.py`

- [ ] **Step 1: Write failing generator tests**

```python
from copy import deepcopy

from emergency_commander.contracts import validate_scenario
from emergency_commander.random_scenario import generate_random_scenario


def test_random_scenario_is_reproducible_and_contract_valid():
    first = generate_random_scenario(20260616)
    second = generate_random_scenario(20260616)
    other = generate_random_scenario(20260617)

    assert first == second
    assert first != other
    validate_scenario(first)
    assert first["events"] == []
    assert len(first["zones"]) == 3
    assert [unit["type"] for unit in first["units"]].count("rescue_car") == 2
    assert [unit["type"] for unit in first["units"]].count("drone") == 1


def test_every_single_ground_road_failure_keeps_zones_reachable():
    scenario = generate_random_scenario(7)
    for failed in scenario["roads"]:
        if failed["road_id"].startswith("R_HOSPITAL"):
            continue
        roads = deepcopy(scenario["roads"])
        next(road for road in roads if road["road_id"] == failed["road_id"])["status"] = "blocked"
        for zone in scenario["zones"]:
            assert _reachable(roads, "HQ", zone["node_id"])


def test_air_routes_reach_every_zone():
    scenario = generate_random_scenario(9)
    for zone in scenario["zones"]:
        assert _reachable(scenario["air_routes"], "HQ", zone["node_id"])
```

The test helper `_reachable` performs an undirected breadth-first search over open roads.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/test_random_scenario.py -v`

Expected: collection fails with `ModuleNotFoundError: emergency_commander.random_scenario`.

- [ ] **Step 3: Implement the deterministic generator**

Implement:

```python
def generate_random_scenario(seed: int, *, mode: str = "fixed") -> dict[str, Any]:
    rng = random.Random(int(seed))
    nodes, zones = _generate_nodes_and_zones(rng)
    scenario = {
        "scenario_id": f"random_{int(seed)}",
        "generated_at": "2026-06-16T00:00:00+08:00",
        "mode": mode,
        "run_mode": mode,
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HOSPITAL"},
        "nodes": nodes,
        "config": _default_config(),
        "zones": zones,
        "roads": _generate_ground_roads(nodes, rng),
        "air_routes": _generate_air_routes(nodes, rng),
        "units": _generate_units(rng),
        "events": [],
    }
    validate_scenario(scenario)
    return scenario
```

Implement the five named private helpers in the same module. For each zone `_generate_ground_roads` creates both `HQ -> ZONE_X` and `HQ -> RELAY_X -> ZONE_X`; no detour road is shared between zones. `_generate_air_routes` creates `HQ -> AIR_RELAY` plus one relay edge per zone. Compute positive distances with `math.hypot`, round generated values, and keep every probability in `[0, 1]`.

- [ ] **Step 4: Run generator tests and full contract tests**

Run: `.venv/bin/pytest tests/test_random_scenario.py tests/test_json_contracts.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the generator**

```bash
git add src/emergency_commander/random_scenario.py tests/test_random_scenario.py
git commit -m "feat: generate reproducible rescue maps"
```

### Task 2: Hospital Delivery Accounting

**Files:**
- Modify: `src/emergency_commander/simulation.py`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write failing delivery tests**

```python
def test_rescue_is_counted_only_after_hospital_delivery():
    scenario = stateful_scenario()
    states = initialize_unit_states(scenario)
    start_assignments(states, [rescue_assignment("A", people=3)], scenario)

    advance_unit_states(states, 3.0, scenario)
    assert states["RescueCar-1"]["delivered_targets"] == []
    assert states["RescueCar-1"]["rescued_people"] == 0

    advance_unit_states(states, 10.0, scenario)
    assert states["RescueCar-1"]["delivered_targets"] == ["A"]
    assert states["RescueCar-1"]["rescued_people"] == 3
    assert states["RescueCar-1"]["travel_minutes"] > 0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest tests/test_simulation.py::test_rescue_is_counted_only_after_hospital_delivery -v`

Expected: FAIL because `delivered_targets` does not exist.

- [ ] **Step 3: Implement delivery-level state**

Initialize these fields:

```python
"delivered_targets": [],
"rescued_people": 0,
"travel_minutes": 0.0,
```

Preserve the rescue origin and passenger count in the return task:

```python
"origin_zone": previous_task.get("target_zone"),
"estimated_people": state["onboard"],
```

Increment `travel_minutes` by actual travel consumed. On hospital arrival append `origin_zone`, add `onboard` to `rescued_people`, then clear `onboard`.

- [ ] **Step 4: Run simulation tests**

Run: `.venv/bin/pytest tests/test_simulation.py -v`

Expected: all tests pass, including existing finite-state behavior.

- [ ] **Step 5: Commit delivery accounting**

```bash
git add src/emergency_commander/simulation.py tests/test_simulation.py
git commit -m "feat: track completed rescue deliveries"
```

### Task 3: Serializable Incremental Simulation Engine

**Files:**
- Create: `src/emergency_commander/live_simulation.py`
- Modify: `src/emergency_commander/pipeline.py`
- Create: `tests/test_live_simulation.py`

- [ ] **Step 1: Write failing phase and execution tests**

```python
from emergency_commander.live_simulation import LiveSimulation, PHASES
from emergency_commander.random_scenario import generate_random_scenario


def test_live_session_starts_without_precomputing_and_advances_phases():
    session = LiveSimulation.create(generate_random_scenario(11), seed=11)
    assert session.phase == "validate"
    assert session.clock_minutes == 0
    assert session.timeline == []

    observed = []
    for _ in range(7):
        observed.append(session.phase)
        session.step()

    assert observed == ["validate", "infer", "prioritize", "route", "utility", "allocate", "execute"]
    assert session.current_plan["assignments"]
    assert [entry["phase"] for entry in session.algorithm_log[:6]] == observed[:6]


def test_execute_step_changes_clock_and_unit_position():
    session = planned_session(seed=12)
    before = {key: value["position"] for key, value in session.unit_states.items()}
    session.step()
    assert session.clock_minutes == 1.0
    assert any(state["position"] != before[key] for key, state in session.unit_states.items())
```

- [ ] **Step 2: Run phase tests and verify RED**

Run: `.venv/bin/pytest tests/test_live_simulation.py -v`

Expected: collection fails because `live_simulation` does not exist.

- [ ] **Step 3: Add global exclusion to planning**

Extend `_plan_idle_units` without changing existing callers:

```python
def _plan_idle_units(
    scenario: dict[str, Any],
    states: dict[str, dict[str, Any]],
    network: DiscreteBayesianNetwork | None,
    model_name: str,
    excluded_zones: set[str] | None = None,
) -> dict[str, Any]:
    excluded_zones = excluded_zones or set()
    eligible_assessments = [
        item for item in assessments
        if item["zone_id"] not in active_zones
        and item["zone_id"] not in excluded_zones
    ]
```

- [ ] **Step 4: Implement the phase machine**

Use a dataclass with explicit serialization:

```python
PHASES = ("validate", "infer", "prioritize", "route", "utility", "allocate", "execute", "replan", "complete")


@dataclass
class LiveSimulation:
    scenario: dict[str, Any]
    seed: int
    model_name: str = "expert_cpt"
    phase: str = "validate"
    status: str = "running"
    clock_minutes: float = 0.0
    step_count: int = 0
    unit_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    assessments: list[dict[str, Any]] = field(default_factory=list)
    utility_matrix: list[dict[str, Any]] = field(default_factory=list)
    current_plan: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    algorithm_log: list[dict[str, Any]] = field(default_factory=list)
    replan_log: list[dict[str, Any]] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)
```

`step()` dispatches one phase only. `execute` advances one minute, stores a snapshot, detects newly idle units, and either returns to `infer`, remains in `execute`, or enters a terminal status. `to_dict()` and `from_dict()` round-trip every field without recomputing.

- [ ] **Step 5: Run phase tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_live_simulation.py -v`

Expected: phase and execution tests pass.

- [ ] **Step 6: Commit the base live engine**

```bash
git add src/emergency_commander/live_simulation.py src/emergency_commander/pipeline.py tests/test_live_simulation.py
git commit -m "feat: add incremental rescue simulation"
```

### Task 4: Live Event Injection and Replanning

**Files:**
- Modify: `src/emergency_commander/live_simulation.py`
- Modify: `tests/test_live_simulation.py`

- [ ] **Step 1: Write failing event tests**

```python
@pytest.mark.parametrize("event_type", [
    "road_collapse", "fire_spread", "new_sos", "drone_update"
])
def test_supported_event_interrupts_and_replans(event_type):
    session = planned_session(seed=21)
    old_plan = deepcopy(session.current_plan)
    event = session.inject_event(event_type)

    assert event["event_type"] == event_type
    assert session.phase == "replan"
    assert len(session.event_log) == 1

    session.step()
    assert session.phase == "infer"
    run_until_phase(session, "execute")
    assert session.replan_log[-1]["old_plan"] == old_plan
    assert session.status == "running"


def test_manual_event_target_is_used_and_duplicate_event_is_not_reapplied():
    session = planned_session(seed=22)
    target = session.available_event_targets("fire_spread")[0]
    event = session.inject_event("fire_spread", target_id=target)
    fire = zone(session, target)["observations"]["fire"]
    with pytest.raises(ValueError, match="already applied"):
        session.inject_event_payload(event)
    assert zone(session, target)["observations"]["fire"] == fire
```

- [ ] **Step 2: Run event tests and verify RED**

Run: `.venv/bin/pytest tests/test_live_simulation.py -k event -v`

Expected: FAIL because event APIs are missing.

- [ ] **Step 3: Implement event targeting and injection**

Implement these concrete public methods; `inject_event` must build one payload and delegate mutation to `inject_event_payload` so callback reruns cannot apply it twice:

```python
def available_event_targets(self, event_type: str) -> list[str]:
    if event_type == "road_collapse":
        return [road["road_id"] for road in self.scenario["roads"] if road["status"] == "open"]
    completed = self.completed_zones()
    return [zone["zone_id"] for zone in self.scenario["zones"] if zone["zone_id"] not in completed]


def inject_event(self, event_type: str, target_id: str | None = None) -> dict[str, Any]:
    selected = target_id or self.select_event_target(event_type)
    event = self.build_event(event_type, selected)
    self.inject_event_payload(event)
    return event
```

Use active ground roads first for collapse; highest priority unfinished zone for fire; lowest SOS unfinished zone for new SOS; posterior closest to `0.5` for drone update. Apply the event once, invalidate only affected road missions, store the old plan, set `phase="replan"`, and keep `status="running"` unless paused.

- [ ] **Step 4: Run event tests and all live-engine tests**

Run: `.venv/bin/pytest tests/test_live_simulation.py tests/test_pipeline_replanning.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit event injection**

```bash
git add src/emergency_commander/live_simulation.py tests/test_live_simulation.py
git commit -m "feat: inject emergencies into live simulation"
```

### Task 5: Completion, Pause and Final Report

**Files:**
- Modify: `src/emergency_commander/live_simulation.py`
- Modify: `tests/test_live_simulation.py`

- [ ] **Step 1: Write failing terminal-state tests**

```python
def test_paused_and_terminal_sessions_do_not_advance():
    session = planned_session(seed=31)
    session.pause()
    frozen = session.to_dict()
    session.step()
    assert session.to_dict() == frozen


def test_final_report_counts_delivered_rescues_not_drone_recon():
    session = run_to_terminal(seed=32)
    report = session.build_result()
    delivered = {
        zone
        for state in session.unit_states.values()
        for zone in state.get("delivered_targets", [])
    }
    assert set(report["completed_zones"]) == delivered
    assert report["rescued_people"] == sum(
        state.get("rescued_people", 0) for state in session.unit_states.values()
    )
    assert report["end_reason"] in {"all_rescues_complete", "no_feasible_tasks", "timeout"}
    assert report["algorithm_log"]
    assert report["timeline"]
```

- [ ] **Step 2: Run terminal tests and verify RED**

Run: `.venv/bin/pytest tests/test_live_simulation.py -k "paused or final_report" -v`

Expected: FAIL because pause/report behavior is missing.

- [ ] **Step 3: Implement lifecycle and reporting**

Add `pause()`, `resume()`, `_terminal_reason()`, and `build_result()`. Stop at all deliveries complete, no feasible inactive work, `max_minutes=120`, or `max_steps=500`. Report seed, clock, completed and incomplete zones, rescued people, per-unit metrics, initial/final plans, event/replan logs, assessments, utility matrix, timeline and algorithm log.

- [ ] **Step 4: Run all engine tests**

Run: `.venv/bin/pytest tests/test_random_scenario.py tests/test_simulation.py tests/test_live_simulation.py tests/test_pipeline_replanning.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit lifecycle and reporting**

```bash
git add src/emergency_commander/live_simulation.py tests/test_live_simulation.py
git commit -m "feat: finalize live rescue reports"
```

### Task 6: Streamlit Live Command Workspace

**Files:**
- Replace: `app.py`
- Modify: `tests/test_visualization.py`

- [ ] **Step 1: Write failing AppTest expectations**

```python
def test_streamlit_live_demo_exposes_one_click_and_event_controls():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    labels = {button.label for button in app.button}
    assert "随机生成并启动" in labels
    assert {"道路坍塌", "火势蔓延", "新增求救", "无人机情报"} <= labels
    assert any("等待生成场景" in item.value for item in app.markdown)


def test_one_click_creates_a_running_live_session():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(button for button in app.button if button.label == "随机生成并启动").click().run()
    assert app.session_state["live_simulation"]["status"] == "running"
    assert app.session_state["live_simulation"]["seed"]
    assert not app.exception
```

- [ ] **Step 2: Run AppTest and verify RED**

Run: `.venv/bin/pytest tests/test_visualization.py -k streamlit -v`

Expected: FAIL because current page still exposes timeline navigation instead of live controls.

- [ ] **Step 3: Implement one-click session controls**

Replace precomputed `run_pipeline` state with:

```python
def start_random_session() -> None:
    seed = secrets.randbelow(900_000_000) + 100_000_000
    scenario = generate_random_scenario(seed, mode=current_mode())
    session = LiveSimulation.create(scenario, seed=seed)
    st.session_state["live_simulation"] = session.to_dict()


@st.fragment(run_every=refresh_interval)
def live_workspace() -> None:
    session = LiveSimulation.from_dict(st.session_state["live_simulation"])
    if session.status == "running":
        session.step(network=load_learned_network() if session.model_name == "learned_cpt" else None)
        st.session_state["live_simulation"] = session.to_dict()
    render_workspace(session)
```

`LiveSimulation.step` accepts an optional network argument but the session dictionary stores only `model_name`; this keeps `st.session_state` serializable.

Keep the existing industrial paper/ink/orange/cyan art direction, but use a 62/38 map and algorithm layout. Add start, pause/resume, regenerate, speed, seed, clock, status, four event buttons, advanced target selection, stage rail, logs, tasks, event history and final-report downloads.

- [ ] **Step 4: Make event controls mutate the current session exactly once**

Each event callback reconstructs the session, calls `inject_event`, stores `to_dict()`, and records a flash message. Disable event controls when no target exists or the session is terminal.

- [ ] **Step 5: Run Streamlit tests**

Run: `.venv/bin/pytest tests/test_visualization.py -v`

Expected: all visualization and AppTest tests pass without exceptions.

- [ ] **Step 6: Commit the live page**

```bash
git add app.py tests/test_visualization.py
git commit -m "feat: add live emergency command workspace"
```

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `agent.md`

- [ ] **Step 1: Document the acceptance flow and algorithms**

Add exact startup command, one-click flow, automatic phases, four event controls, report downloads, and algorithms used at each phase. Preserve existing proposal-comparison content in `agent.md` and append a “实时网页验收” section.

- [ ] **Step 2: Run formatting and full automated tests**

Run:

```bash
.venv/bin/python -m compileall -q src app.py
.venv/bin/pytest -q
git diff --check
```

Expected: compile exits 0, all tests pass, and diff check emits no errors.

- [ ] **Step 3: Start Streamlit for browser acceptance**

Run: `.venv/bin/streamlit run app.py --server.headless true --server.port 8501`

Expected: health endpoint responds and the page loads at `http://127.0.0.1:8501`.

- [ ] **Step 4: Execute the browser acceptance path**

Using Playwright CLI:

1. Capture the initial waiting page.
2. Click “随机生成并启动” once.
3. Wait across several fragment refreshes and verify stage/clock or unit positions advance.
4. Click “道路坍塌” while running.
5. Verify event count increases, replan appears in the algorithm rail/log, and execution resumes.
6. Increase speed and wait until terminal, or use repeated timed waits within the 120-minute simulation cap.
7. Verify the final report and JSON/Markdown download buttons.
8. Save screenshots under `output/playwright/live-simulation-*.png`.

- [ ] **Step 5: Audit every acceptance requirement**

Record evidence for: one click creates a random map; automatic real phase progression; side-by-side algorithm demonstration; event injection at any running time; current-state replanning and continuation; terminal result output. Do not mark complete if any item is supported only by source inspection rather than runtime evidence.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md agent.md output/playwright/live-simulation-*.png
git commit -m "docs: add live simulation acceptance guide"
```
