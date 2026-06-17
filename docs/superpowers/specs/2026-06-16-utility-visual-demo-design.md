# Explainable Utility and Visual Demo Design

## Goal

Complete the proposal's expected-utility decision layer and make the Streamlit demo visibly explain why each unit-zone assignment wins, what it costs, and how event-driven replanning changes the map.

## Decision Model

Each feasible unit-zone candidate uses:

```text
EU = alpha * trapped_probability
   + beta * life_risk
   + gamma * mission_accessibility
   - delta * normalized_arrival_time
   - epsilon * path_risk
   - zeta * resource_cost
```

`resource_cost` is a normalized per-dispatch unit property representing personnel, fuel, or battery consumption. Defaults are `0.55` for rescue cars and `0.25` for drones, and scenarios may override them. Every signed contribution is returned in `utility_breakdown`; the total must equal `expected_utility` within rounding tolerance.

## Pipeline Contract

- Candidate records expose feasibility, rejection reason, route, input factors, signed contributions, resource cost, total utility, and a deterministic Chinese explanation.
- Selected assignments preserve the same breakdown and explanation while units remain active.
- Public plans expose `utility_matrix` so the UI does not recompute decision logic.
- Timeline snapshots include `scenario_state`, allowing the map to render road and zone changes at the selected event step.
- Existing fixed and learned CPT modes continue to use the same decision contract.

## Demo Design

Keep the current offline command-center visual language. Add an `效用决策` tab containing:

1. selected assignment cards with total utility and explanation;
2. a unit-zone candidate matrix including infeasible options;
3. a contribution waterfall for the selected candidate;
4. explicit resource cost, ETA, path risk, and feasibility reason.

The map renders ground roads by risk band and blocked state, keeps air corridors visually separate, and uses the timeline snapshot's scenario state. The sidebar adds previous/next event controls while retaining the timeline slider.

## Verification

- Unit tests prove the resource penalty changes utility and every breakdown sums to the total.
- Contract tests prove the new fields validate in both fixed and learned outputs.
- Visualization tests prove risk layers, utility frames, and contribution figures exist.
- Streamlit AppTest proves the utility tab and event controls load without exceptions.
- Browser verification checks initial and collapsed-road steps visually.

