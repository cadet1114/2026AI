# P0/P1 Bayesian Rescue Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a verified full-Bayesian, stateful, coordinate-routed rescue experiment and Streamlit demo on Apple M3 with 16GB memory.

**Architecture:** Replace linear probability fusion with a self-contained discrete Bayesian network and CPT learner. Extend the shared JSON contract with coordinates, air routes, and unit state, then drive both CLI and Streamlit from the same pipeline. Keep public-data anchoring, simulation, evaluation, and UI as separate modules with explicit provenance.

**Tech Stack:** Python 3.13, NumPy, pandas, scikit-learn, requests, Streamlit, Plotly, pytest.

---

### Task 1: Full Bayesian Network

**Files:** Create `src/emergency_commander/bayesian_network.py`, `src/emergency_commander/expert_cpts.py`; modify `src/emergency_commander/inference.py`; test `tests/test_bayesian_network.py`.

- [ ] Write failing tests for CPT validation, exact posterior inference, missing evidence, and learned CPT estimation.
- [ ] Run `python3 -m pytest tests/test_bayesian_network.py -q` and confirm failures are caused by missing implementation.
- [ ] Implement topological validation, enumeration inference, Dirichlet CPT learning, and contribution traces.
- [ ] Replace linear inference with Bayesian posteriors while retaining life-risk and priority formulas.
- [ ] Run Bayesian and legacy inference tests until green.

### Task 2: Credible Hybrid Experiment

**Files:** Create `src/emergency_commander/public_data.py`, `src/emergency_commander/experiment.py`, `config/experiment.yaml`; modify `src/emergency_commander/cli.py`; test `tests/test_experiment.py`.

- [ ] Write failing tests for deterministic public mapping, provenance, noisy ancestral sampling, fold isolation, and metric schema.
- [ ] Download a bounded USGS CSV through the official API and store source metadata/checksum.
- [ ] Generate 50,000 hybrid rows with noise, missingness, imbalance controls, and provenance.
- [ ] Run five-fold CPT learning and report Brier, accuracy, F1, ROC-AUC, calibration, and missing-evidence results.
- [ ] Write learned CPT, metrics JSON, CSV summaries, and Markdown report artifacts.

### Task 3: True A* And Drone Routing

**Files:** Modify `src/emergency_commander/input_adapter.py`, `src/emergency_commander/routing.py`, `src/emergency_commander/allocation.py`, schemas and scenario; test `tests/test_routing_v2.py`.

- [ ] Write failing tests proving nonzero admissible heuristic use and strict separation of roads from air routes.
- [ ] Add node coordinates and route-layer validation.
- [ ] Implement coordinate-aware A* telemetry and independent drone graph traversal.
- [ ] Update allocation to choose the appropriate route layer per unit.
- [ ] Verify collapse, fire-limit, and route-layer regressions.

### Task 4: Stateful Dynamic Dispatch

**Files:** Create `src/emergency_commander/simulation.py`; modify `pipeline.py`, `replanning.py`, `allocation.py`, schemas and scenario; test `tests/test_simulation.py`.

- [ ] Write failing tests for elapsed-time movement, task completion, capacity, hospital return, and targeted replanning.
- [ ] Implement unit-state initialization and deterministic clock advancement.
- [ ] Preserve valid active missions and reassign only idle or invalidated units.
- [ ] Emit state snapshots and event timeline data in decision output.
- [ ] Verify road collapse and drone evidence scenarios end to end.

### Task 5: Streamlit Command Center

**Files:** Create `app.py`, `src/emergency_commander/visualization.py`, `.streamlit/config.toml`; modify dependencies and README; test `tests/test_visualization.py`.

- [ ] Write failing tests for map traces, metric frames, timeline snapshots, and app import.
- [ ] Build the industrial emergency-operations layout with Plotly map and comparison charts.
- [ ] Add fixed/learned switch, timeline selector, scenario upload, and report download.
- [ ] Run Streamlit headlessly and verify health endpoint.
- [ ] Inspect desktop and mobile layouts in the in-app browser and fix visible defects.

### Task 6: Completion And Handoff

**Files:** Modify `README.md`, `HANDOFF.md`, `计划.md`; create `.github/workflows/test.yml`.

- [ ] Run full pytest, compile, JSON parsing, experiment reproducibility, and Streamlit health checks.
- [ ] Audit each P0/P1 requirement against current artifacts and browser evidence.
- [ ] Record exact commands, runtime, peak dataset size, metrics, and known limitations.
- [ ] Commit the feature branch and push it to GitHub.

