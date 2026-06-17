# Manual Emergency Command Console Design

## Goal

Build a fixed-height Streamlit command console that presents a larger, more complex random rescue map and lets the presenter manually trigger every algorithm phase. Each click must expose the calculation evidence behind the result, so the audience can see why Bayesian inference, risk-aware A*, expected utility, global allocation, and event-driven replanning outperform naive nearest-target decisions.

## Confirmed Product Decisions

- The application uses one desktop screen and does not require page scrolling at a 1440 x 900 viewport.
- The presenter controls progress. There is no automatic timer or background phase advancement.
- A normal execution click advances one simulated minute. A second execution control advances exactly to the next unit state transition.
- Random event controls sit immediately beside the map.
- The generated map is visible as soon as it is created, before validation or inference runs.
- Plotly Graph Objects remains the map technology because the scenario uses schematic coordinates, must work offline, and needs precise route and hazard styling.
- The visual direction is an industrial emergency operations board: dark graphite shell, warm map surface, safety orange for active decisions, cyan for intelligence, and restrained green/red status accents.

## Single-Screen Layout

The app occupies `100vh` and hides body overflow. The central workspace uses fixed rows so all primary controls remain visible:

1. A compact 54 px command header contains the title, random seed, model, current phase, simulated time, and progress.
2. The main row fills the remaining height. The map occupies about 60 percent of the width, a narrow event dock is attached to its right edge, and the calculation inspector occupies about 40 percent.
3. A compact footer shows execution granularity, unit state cards, and route/utility/event counters.

The map and inspector may scroll internally only when their content exceeds their fixed area. The document itself must not scroll at the target desktop viewport.

## Manual Interaction Model

### Session Creation

`Generate Complex Map` creates a deterministic scenario from a displayed seed and initializes a `LiveSimulation`. It must not call `step()`. The raw road graph, zones, infrastructure, and units are drawn immediately with neutral pre-inference zone styling.

The command header exposes:

- `Next Algorithm Step`: executes exactly one current phase.
- `Advance 1 Minute`: available during the execute phase.
- `Advance To Next Transition`: available during the execute phase and advances by the smallest positive remaining travel or service duration among active units.
- `New Complex Map`: replaces the current session with a new seed.
- `Previous Detail` and `Next Detail`: move through calculation history without rewinding simulation state.

### Phase Sequence

The manual sequence remains:

1. Validate and normalize input.
2. Infer Bayesian posteriors.
3. Calculate and rank priorities.
4. Search candidate routes with risk-aware A*.
5. Calculate expected utility.
6. Enumerate and select the global assignment.
7. Execute unit state transitions.
8. Apply an event and replan from current state when requested.
9. Freeze and summarize the final result.

Each click appends one immutable calculation record and selects that new record in the inspector.

## Complex Scenario Generator

The random scenario expands from three zones and nine ground links to a presentation-scale graph:

- Six disaster zones distributed across a wider coordinate field.
- At least eighteen intermediate road nodes arranged as districts, junctions, bridges, and relay points.
- Between thirty and forty ground roads, with redundant loops and at least two viable paths from headquarters to every zone before failures.
- Six to nine air corridors connecting an air relay network to all zones.
- Three rescue vehicles and two drones with visibly different speed, capacity, cost, and risk constraints.

The generator must remain deterministic for a seed and contract-valid. It deliberately creates decision conflicts:

- A short route with elevated fire or damage risk.
- A longer, safer bypass.
- Congested central links that make Euclidean nearest-target selection suboptimal.
- Different zone urgency, trapped probability evidence, and accessibility.
- Several roads shared by otherwise attractive routes, so a collapse demonstrates meaningful replanning.

Every single non-hospital ground-road failure must still leave every zone reachable. This guarantees that a collapse changes the optimum instead of trivially ending the demo.

## Map Rendering

The Plotly map uses the full available panel and preserves equal axis scale.

- Roads are grouped by low, medium, and high weighted risk. Width and opacity reinforce risk without overwhelming labels.
- Blocked roads use a black dashed line and a visible `X` marker at the midpoint.
- Air corridors are cyan dotted lines.
- Zones have a translucent risk halo, a priority rank badge, and a concise label. Before inference, they use neutral styling derived from raw hazard observations.
- Candidate routes are thin translucent overlays. Selected routes are bright, wider, and unit-colored.
- Units show type, direction, target, and travel progress. Vehicles and drones use distinct symbols.
- The object involved in the selected calculation record receives a pulse-like highlight or thicker outline.
- Hover cards expose road risk components, zone evidence/posteriors, and route `ETA`, path risk, and total A* cost.

The map legend is compact and horizontal. Infrastructure and unit labels must remain readable at the target viewport.

## Calculation Records

`LiveSimulation` gains a serializable `calculation_history` list. Each record contains a common envelope:

```text
index, phase, title, clock_minutes, summary, focus, inputs, operations, outputs
```

`focus` identifies map objects to highlight. The remaining fields are JSON-compatible dictionaries and lists rendered by phase-specific inspector components.

### Validation

Show schema/contract checks, input counts, normalized defaults, initialized unit states, and the before/after normalization changes.

### Bayesian Inference

For every zone, show discretized evidence, both query targets, posterior distributions, and evidence contribution deltas from `DiscreteBayesianNetwork.explain()`. The interface must describe exact enumeration honestly; it must not claim to display every hidden CPT multiplication unless the inference engine records it.

### Priority Calculation

Show each weighted term for life risk and priority:

```text
life_risk = fire_weight * fire
          + trapped_weight * trapped_probability
          + urgency_weight * urgency

priority = trapped_weight * trapped_probability
         + life_weight * life_risk
         + urgency_weight * urgency
         + accessibility_weight * passability_probability
```

Display all term values, totals, and final rank.

### Risk-Aware A*

Extend `risk_aware_astar` with an optional serializable search trace. For each expanded node record `g`, `h`, `f`, frontier size, and accepted neighbor relaxations. Candidate results show path nodes, road IDs, ETA, weighted path risk, total cost, heuristic, and expanded-node count.

The inspector compares at least the selected candidate with the direct or fastest-time alternative when available, explaining why a slightly longer route can have lower risk-adjusted cost.

### Expected Utility

Show the six signed contributions already used by the system:

- trapped benefit
- life-risk benefit
- accessibility or mission-fit benefit
- arrival-time cost
- path-risk cost
- resource cost

Display the formula, waterfall contribution values, total utility, and infeasibility reason for rejected candidates.

### Global Allocation

Extend allocation with an optional trace that records the number of combinations considered, duplicate-zone combinations rejected, feasible totals, successive best totals, and the winning assignment. The UI shows a compact ranked table rather than every combination when the search space is large.

### Execution

Record the chosen delta, each unit's state before and after, remaining duration changes, arrivals, rescues, returns, deliveries, and cumulative clock.

`Advance To Next Transition` computes the minimum positive transition duration among active units and calls state advancement once with that delta. It never loops over hidden one-minute steps.

### Event And Replanning

Events are enabled after initial allocation begins. Triggering an event applies it immediately, captures the old plan, moves the session to `replan`, and highlights the affected road or zone. Subsequent manual clicks repeat inference, routing, utility, and allocation using current unit positions. The inspector compares old and new selected routes and identifies the changed constraints.

## Event Dock

The dock is visually attached to the map and contains four compact controls:

- Road collapse
- Fire spread
- New SOS
- Drone intelligence update

Automatic target selection remains the default and shows the chosen target before confirmation. An expandable advanced section permits manual targeting without increasing the page height. Events are disabled before initial allocation and after completion.

## Component Boundaries

- `random_scenario.py`: deterministic complex graph generation only.
- `routing.py`: route search plus optional trace production.
- `allocation.py`: utility/allocation plus optional enumeration trace.
- `live_simulation.py`: phase orchestration, calculation records, and transition-sized execution.
- `visualization.py`: Plotly figures and map styling only.
- `app.py`: Streamlit state, controls, fixed-screen composition, and inspector rendering.

The trace APIs are optional so CLI and existing pipeline consumers keep their current outputs unless they explicitly request detail.

## Error Handling

- A generation or phase error sets the session to `error`, retains the last valid map, and renders a concise inspector error with the failed phase.
- If no route exists for one candidate, the candidate remains visible as rejected and does not stop the phase.
- If no active transition exists, the transition button is disabled and the normal phase control remains available.
- Streamlit session payloads remain JSON-serializable; no Plotly objects or network instances are stored in session state.

## Testing And Acceptance

Automated tests must verify:

- Same seed produces the same expanded scenario; another seed differs.
- Scenario size meets the node, zone, road, air-route, and unit bounds.
- Every zone is reachable and remains reachable after any single eligible road failure.
- New sessions remain at `validate` until the presenter clicks.
- Each phase adds exactly one correctly shaped calculation record.
- A* trace values satisfy `f = g + h` and end on the returned route.
- Allocation trace identifies the same winning assignment returned by the function.
- One-minute and next-transition execution use the requested deltas.
- Events are gated correctly and replanning starts from current state.
- Serialization preserves calculation history and selected execution behavior.

Browser acceptance at 1440 x 900 must verify:

- No document-level vertical scrollbar.
- The complex map, event dock, calculation inspector, and footer are visible together.
- The map is visible immediately after generation.
- Every algorithm phase advances only after one explicit click.
- The inspector displays phase-specific arithmetic or trace data.
- A road collapse visibly blocks a road and produces a different route when the blocked road affected the active plan.
- The final result remains readable and downloadable without leaving the command console.

## Out Of Scope

- Real geographic tiles, latitude/longitude geocoding, or Internet-dependent map services.
- Continuous animation or automatic timers.
- Rewinding simulation state.
- Replacing exhaustive allocation with an optimizer intended for large production fleets.
- Changing the Bayesian model structure or retraining learned CPTs.
