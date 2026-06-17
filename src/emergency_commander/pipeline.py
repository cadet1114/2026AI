from __future__ import annotations

from copy import deepcopy
from math import hypot
from typing import Any

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.contracts import validate_decision_output
from emergency_commander.inference import assess_zones
from emergency_commander.input_adapter import normalize_scenario
from emergency_commander.replanning import apply_event
from emergency_commander.routing import NoRouteError, risk_aware_astar
from emergency_commander.simulation import (
    advance_unit_states,
    initialize_unit_states,
    start_assignments,
)


POSITION_ANCHOR_PREFIX = "__unit_"


def _position_anchor_id(unit_id: str) -> str:
    return f"{POSITION_ANCHOR_PREFIX}{unit_id}_position"


def _strip_position_anchor(scenario: dict[str, Any], unit_id: str) -> None:
    anchor_id = _position_anchor_id(unit_id)
    scenario.get("nodes", {}).pop(anchor_id, None)
    connector_prefix = f"{anchor_id}_connector_"
    for collection in ("roads", "air_routes"):
        scenario[collection] = [
            road
            for road in scenario.get(collection, [])
            if not road["road_id"].startswith(connector_prefix)
        ]


def _same_position(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> bool:
    if not left or not right:
        return False
    return (
        abs(float(left["x"]) - float(right["x"])) <= 1e-6
        and abs(float(left["y"]) - float(right["y"])) <= 1e-6
    )


def _route_graph_for_unit(
    route_graph: list[dict[str, Any]], unit_id: str
) -> list[dict[str, Any]]:
    return [
        road
        for road in route_graph
        if road.get("labels", {}).get("unit_anchor") in {None, unit_id}
    ]


def _hide_temporary_route_edges(route: dict[str, Any]) -> dict[str, Any]:
    road_ids = route.get("road_ids", [])
    if any(road_id.startswith(POSITION_ANCHOR_PREFIX) for road_id in road_ids):
        route["_edge_road_ids"] = list(road_ids)
        route["road_ids"] = [
            road_id
            for road_id in road_ids
            if not road_id.startswith(POSITION_ANCHOR_PREFIX)
        ]
    return route


def _distance_to_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
) -> float:
    segment_x = float(end["x"]) - float(start["x"])
    segment_y = float(end["y"]) - float(start["y"])
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared <= 1e-12:
        return hypot(
            float(point["x"]) - float(start["x"]),
            float(point["y"]) - float(start["y"]),
        )
    projection = (
        (float(point["x"]) - float(start["x"])) * segment_x
        + (float(point["y"]) - float(start["y"])) * segment_y
    ) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    nearest_x = float(start["x"]) + projection * segment_x
    nearest_y = float(start["y"]) + projection * segment_y
    return hypot(float(point["x"]) - nearest_x, float(point["y"]) - nearest_y)


def _road_by_id(scenario: dict[str, Any], road_id: str | None) -> dict[str, Any] | None:
    if road_id is None:
        return None
    for road in scenario.get("roads", []):
        if road["road_id"] == road_id:
            return road
    return None


def _current_route_segment(
    scenario: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, str, dict[str, Any] | None] | None:
    task = state.get("current_task") or {}
    route = task.get("route") or {}
    path = route.get("path") or []
    if len(path) < 2 or not isinstance(state.get("position"), dict):
        return None

    nodes = scenario.get("nodes", {})
    position = state["position"]
    candidates = []
    edge_road_ids = route.get("_edge_road_ids", route.get("road_ids", []))
    for index, (start_id, end_id) in enumerate(zip(path, path[1:])):
        if start_id not in nodes or end_id not in nodes:
            continue
        road_id = None
        if index < len(edge_road_ids):
            road_id = edge_road_ids[index]
        candidates.append(
            (
                _distance_to_segment(position, nodes[start_id], nodes[end_id]),
                index,
                start_id,
                end_id,
                _road_by_id(scenario, road_id),
            )
        )
    if not candidates:
        return None
    _, _, start_id, end_id, road = min(candidates, key=lambda item: (item[0], item[1]))
    return start_id, end_id, road


def _connector_travel_time(
    *,
    position: dict[str, float],
    endpoint: dict[str, float],
    segment_start: dict[str, float] | None,
    segment_end: dict[str, float] | None,
    base_road: dict[str, Any] | None,
) -> float:
    distance = hypot(
        float(position["x"]) - endpoint["x"],
        float(position["y"]) - endpoint["y"],
    )
    if not base_road or not segment_start or not segment_end:
        return distance
    segment_distance = hypot(
        float(segment_start["x"]) - float(segment_end["x"]),
        float(segment_start["y"]) - float(segment_end["y"]),
    )
    if segment_distance <= 1e-6:
        return distance
    return float(base_road["travel_time_base"]) * distance / segment_distance


def _ensure_position_anchor(
    scenario: dict[str, Any], state: dict[str, Any], unit: dict[str, Any]
) -> str:
    """Attach a temporary graph node at the unit's current visual position."""
    nodes = scenario.get("nodes", {})
    position = state.get("position")
    if not nodes or not isinstance(position, dict):
        state.pop("_temporary_route_edges", None)
        return state["current_node"]
    for node_id, coordinates in nodes.items():
        if not node_id.startswith(POSITION_ANCHOR_PREFIX) and _same_position(
            position, coordinates
        ):
            state["current_node"] = node_id
            state.pop("_temporary_route_edges", None)
            return node_id

    unit_id = state["unit_id"]
    anchor_id = _position_anchor_id(unit_id)
    _strip_position_anchor(scenario, unit_id)
    nodes[anchor_id] = {
        "x": round(float(position["x"]), 6),
        "y": round(float(position["y"]), 6),
    }

    if unit["type"] == "drone":
        state["current_node"] = anchor_id
        state.pop("_temporary_route_edges", None)
        return anchor_id

    segment = _current_route_segment(scenario, state)
    base_road = None
    segment_start_id = None
    segment_end_id = None
    if segment:
        segment_start_id, segment_end_id, base_road = segment
        if base_road and base_road.get("status", "open") == "blocked":
            ranked_endpoint_ids = [segment_start_id]
        else:
            ranked_endpoint_ids = [segment_start_id, segment_end_id]
    elif state.get("current_node") in nodes and not str(state["current_node"]).startswith(
        POSITION_ANCHOR_PREFIX
    ):
        ranked_endpoint_ids = [state["current_node"]]
    else:
        ranked_endpoint_ids = []

    temporary_edges = []
    for node_id in ranked_endpoint_ids:
        if node_id not in nodes:
            continue
        distance = hypot(
            float(position["x"]) - nodes[node_id]["x"],
            float(position["y"]) - nodes[node_id]["y"],
        )
        travel_time = _connector_travel_time(
            position=position,
            endpoint=nodes[node_id],
            segment_start=nodes.get(segment_start_id) if segment_start_id else None,
            segment_end=nodes.get(segment_end_id) if segment_end_id else None,
            base_road=base_road,
        )
        if distance <= 1e-6:
            continue
        labels = {"synthetic": True, "unit_anchor": unit_id}
        if base_road:
            labels["base_road_id"] = base_road["road_id"]
        temporary_edges.append(
            {
                "road_id": f"{anchor_id}_connector_{node_id}",
                "from": anchor_id,
                "to": node_id,
                "distance": round(distance, 6),
                "travel_time_base": round(travel_time, 6),
                "status": "open",
                "bidirectional": True,
                "risk": deepcopy(base_road["risk"])
                if base_road
                else {
                    "fire": 0.0,
                    "damage": 0.0,
                    "congestion": 0.0,
                    "secondary_disaster": 0.0,
                },
                "labels": labels,
            }
        )
    state["current_node"] = anchor_id
    state["_temporary_route_edges"] = temporary_edges
    return anchor_id


def _interrupt_unit_for_replanning(
    state: dict[str, Any], scenario: dict[str, Any], unit: dict[str, Any]
) -> bool:
    task = state.get("current_task")
    if (
        state["status"] not in {"en_route", "rescuing"}
        or not task
        or task.get("target_zone") is None
        or int(state.get("onboard", 0)) > 0
    ):
        return False
    _ensure_position_anchor(scenario, state, unit)
    state["status"] = "idle"
    state["current_task"] = None
    state["remaining_travel"] = 0.0
    state["remaining_service"] = 0.0
    return True


def _public_plan(
    assessments: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    utility_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    assignments = []
    routes = []
    for state in states.values():
        task = state.get("current_task")
        if not task:
            continue
        route = task["route"]
        if task.get("target_zone") is not None:
            assignments.append(
                {
                    "unit_id": state["unit_id"],
                    "target_zone": task["target_zone"],
                    "mission_type": task["mission_type"],
                    "expected_utility": task.get("expected_utility"),
                    "estimated_people": task.get("estimated_people", 0),
                    "reason": task.get("reason", "active_mission"),
                    "feasibility_reason": task.get("feasibility_reason", "feasible"),
                    "explanation": task.get("explanation", task.get("reason", "active_mission")),
                    "resource_cost": task.get("resource_cost", state.get("resource_cost", 0.0)),
                    "utility_inputs": deepcopy(task.get("utility_inputs")),
                    "utility_breakdown": deepcopy(task.get("utility_breakdown")),
                    "unit_status": state["status"],
                }
            )
        routes.append(
            {
                "unit_id": state["unit_id"],
                "target_zone": task.get("target_zone"),
                "mission_type": task["mission_type"],
                "path": route["path"],
                "road_ids": route["road_ids"],
                "eta": route["eta"],
                "remaining_eta": state["remaining_travel"],
                "path_risk": route["path_risk"],
                "route_layer": route.get("route_layer", "ground"),
                "heuristic": route.get("heuristic", "zero_fallback"),
                "expanded_nodes": route.get("expanded_nodes", 0),
            }
        )
    return {
        "zone_assessment": assessments,
        "assignments": sorted(assignments, key=lambda item: item["unit_id"]),
        "routes": sorted(routes, key=lambda item: item["unit_id"]),
        "utility_matrix": deepcopy(utility_matrix),
    }


def _plan_idle_units(
    scenario: dict[str, Any],
    states: dict[str, dict[str, Any]],
    network: DiscreteBayesianNetwork | None,
    model_name: str,
) -> dict[str, Any]:
    assessments = assess_zones(scenario, network, model_name=model_name)
    active_zones = {
        state["current_task"].get("target_zone")
        or state["current_task"].get("origin_zone")
        for state in states.values()
        if state.get("current_task")
        and (
            state["current_task"].get("target_zone")
            or state["current_task"].get("origin_zone")
        )
    }
    planning = deepcopy(scenario)
    planning["units"] = []
    for unit in scenario["units"]:
        state = states[unit["unit_id"]]
        if state["status"] != "idle":
            continue
        available = deepcopy(unit)
        available["start_node"] = state["current_node"]
        if state.get("_temporary_route_edges"):
            available["_temporary_route_edges"] = deepcopy(
                state["_temporary_route_edges"]
            )
        planning["units"].append(available)
    eligible_assessments = [
        assessment for assessment in assessments if assessment["zone_id"] not in active_zones
    ]
    matrix = (
        build_utility_matrix(planning, eligible_assessments)
        if planning["units"] and eligible_assessments
        else []
    )
    matrix = [
        candidate
        for candidate in matrix
        if candidate["target_zone"]
        not in states[candidate["unit_id"]].get("completed_targets", [])
    ]
    new_assignments = allocate_tasks(planning, matrix) if matrix else []
    assessment_by_zone = {item["zone_id"]: item for item in assessments}
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    for assignment in new_assignments:
        unit = units[assignment["unit_id"]]
        trapped = assessment_by_zone[assignment["target_zone"]]["trapped_prob"]
        capacity = int(unit.get("capacity", 4 if unit["type"] == "rescue_car" else 0))
        assignment["estimated_people"] = (
            max(1, round(trapped * capacity))
            if assignment["mission_type"] == "rescue" and trapped >= 0.5
            else 0
        )
    start_assignments(states, new_assignments, scenario)
    return _public_plan(assessments, states, matrix)


def _reroute_returning_unit(
    state: dict[str, Any],
    scenario: dict[str, Any],
) -> None:
    task = state.get("current_task") or {}
    target_node = task.get("target_node") or scenario.get("hospital", {}).get("node_id")
    if not target_node:
        state["status"] = "stranded"
        return
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    unit = units[state["unit_id"]]
    start_node = _ensure_position_anchor(scenario, state, unit)
    constraints = unit.get("constraints", {})
    try:
        route = risk_aware_astar(
            _route_graph_for_unit(
                [*scenario["roads"], *state.get("_temporary_route_edges", [])],
                state["unit_id"],
            ),
            nodes=scenario.get("nodes"),
            start=start_node,
            goal=target_node,
            speed=float(unit["speed"]),
            risk_weights=scenario["config"]["weights"]["astar_risk"],
            unit_type=unit["type"],
            max_fire_risk=constraints.get("max_fire_risk"),
            route_layer="ground",
        )
        route["risk_policy"] = "standard"
        _hide_temporary_route_edges(route)
    except NoRouteError:
        try:
            route = risk_aware_astar(
                _route_graph_for_unit(
                    [*scenario["roads"], *state.get("_temporary_route_edges", [])],
                    state["unit_id"],
                ),
                nodes=scenario.get("nodes"),
                start=start_node,
                goal=target_node,
                speed=float(unit["speed"]),
                risk_weights=scenario["config"]["weights"]["astar_risk"],
                unit_type=unit["type"],
                max_fire_risk=None,
                route_layer="ground",
            )
            route["risk_policy"] = "relaxed_fire_limit"
            route["relaxed_constraints"] = ["max_fire_risk"]
            _hide_temporary_route_edges(route)
        except NoRouteError:
            state["status"] = "stranded"
            return
    task["route"] = route
    task["initial_eta"] = float(route["eta"])
    state["remaining_travel"] = float(route["eta"])
    state["current_task"] = task


def _invalidate_affected_missions(
    states: dict[str, dict[str, Any]], event: dict[str, Any], scenario: dict[str, Any]
) -> None:
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    if event["event_type"] != "road_collapse":
        for state in states.values():
            _interrupt_unit_for_replanning(state, scenario, units[state["unit_id"]])
        return
    collapsed_route = event["target_id"]
    for state in states.values():
        task = state.get("current_task")
        if (
            state["status"] in {"en_route", "returning"}
            and task
            and task["route"].get("route_layer", "ground") == "ground"
            and collapsed_route in task["route"]["road_ids"]
        ):
            if state["status"] == "returning":
                _reroute_returning_unit(state, scenario)
            else:
                _interrupt_unit_for_replanning(state, scenario, units[state["unit_id"]])


def run_pipeline(
    raw_scenario: dict[str, Any],
    *,
    process_events: bool = True,
    network: DiscreteBayesianNetwork | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run Bayesian inference, stateful dispatch, routing, and event replanning."""
    scenario = normalize_scenario(raw_scenario)
    model_name = model_name or (
        "learned_cpt" if scenario["run_mode"] == "learned" else "expert_cpt"
    )
    states = initialize_unit_states(scenario)
    simulation_clock = 0.0
    current_plan = _plan_idle_units(scenario, states, network, model_name)
    timeline = [
        {
            "step": 0,
            "clock_minutes": simulation_clock,
            "event": None,
            "plan": deepcopy(current_plan),
            "unit_states": deepcopy(states),
            "scenario_state": deepcopy(scenario),
        }
    ]
    replan_log: list[dict[str, Any]] = []

    if process_events:
        for step, event in enumerate(
            sorted(scenario["events"], key=lambda item: item.get("trigger_step", 0)),
            start=1,
        ):
            old_plan = deepcopy(current_plan)
            elapsed = float(event.get("elapsed_minutes", 0.0))
            advance_unit_states(states, elapsed, scenario)
            simulation_clock += elapsed
            scenario = normalize_scenario(apply_event(scenario, event))
            _invalidate_affected_missions(states, event, scenario)
            current_plan = _plan_idle_units(scenario, states, network, model_name)
            snapshot = {
                "step": step,
                "clock_minutes": round(simulation_clock, 6),
                "event": deepcopy(event),
                "plan": deepcopy(current_plan),
                "unit_states": deepcopy(states),
                "scenario_state": deepcopy(scenario),
            }
            timeline.append(snapshot)
            replan_log.append(
                {
                    "trigger_event": deepcopy(event),
                    "clock_minutes": round(simulation_clock, 6),
                    "old_plan": old_plan,
                    "new_plan": deepcopy(current_plan),
                    "reason": event.get("description", event["event_type"]),
                }
            )

    output = {
        "scenario_id": scenario["scenario_id"],
        "run_mode": scenario["run_mode"],
        "bayesian_model": model_name,
        "weights_used": deepcopy(scenario["config"]["weights"]),
        **current_plan,
        "unit_states": states,
        "simulation_clock": round(simulation_clock, 6),
        "timeline": timeline,
        "replan_log": replan_log,
    }
    validate_decision_output(output)
    return output
