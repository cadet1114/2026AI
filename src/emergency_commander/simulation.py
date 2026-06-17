from __future__ import annotations

from copy import deepcopy
from math import hypot
from typing import Any

from emergency_commander.routing import NoRouteError, risk_aware_astar


def _route_graph_for_unit(
    route_graph: list[dict[str, Any]], unit_id: str
) -> list[dict[str, Any]]:
    return [
        road
        for road in route_graph
        if road.get("labels", {}).get("unit_anchor") in {None, unit_id}
    ]


def initialize_unit_states(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = scenario.get("nodes", {})
    states = {}
    for unit in scenario["units"]:
        start = unit["start_node"]
        states[unit["unit_id"]] = {
            "unit_id": unit["unit_id"],
            "type": unit["type"],
            "status": "idle",
            "current_node": start,
            "position": deepcopy(nodes.get(start, {"x": 0.0, "y": 0.0})),
            "capacity": int(unit.get("capacity", 4 if unit["type"] == "rescue_car" else 0)),
            "resource_cost": float(
                unit.get("resource_cost", 0.55 if unit["type"] == "rescue_car" else 0.25)
            ),
            "onboard": 0,
            "current_task": None,
            "remaining_travel": 0.0,
            "remaining_service": 0.0,
            "completed_missions": 0,
            "completed_targets": [],
            "delivered_targets": [],
            "rescued_people": 0,
            "travel_minutes": 0.0,
        }
    return states


def _zone_node(scenario: dict[str, Any], zone_id: str) -> str:
    for zone in scenario["zones"]:
        if zone["zone_id"] == zone_id:
            return zone["node_id"]
    raise ValueError(f"unknown target zone '{zone_id}'")


def start_assignments(
    states: dict[str, dict[str, Any]],
    assignments: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> None:
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    for assignment in assignments:
        state = states[assignment["unit_id"]]
        if state["status"] != "idle":
            continue
        route = deepcopy(assignment["route"])
        target_node = _zone_node(scenario, assignment["target_zone"])
        state["status"] = "en_route"
        state["remaining_travel"] = float(route["eta"])
        state["remaining_service"] = float(
            units[assignment["unit_id"]].get("service_time", 1.0)
        )
        state["current_task"] = {
            "mission_type": assignment["mission_type"],
            "target_zone": assignment["target_zone"],
            "target_node": target_node,
            "estimated_people": int(assignment.get("estimated_people", 0)),
            "expected_utility": assignment.get("expected_utility"),
            "reason": assignment.get("reason", "feasible"),
            "feasibility_reason": assignment.get("feasibility_reason", "feasible"),
            "explanation": assignment.get("explanation", assignment.get("reason", "feasible")),
            "resource_cost": assignment.get("resource_cost", state["resource_cost"]),
            "utility_inputs": deepcopy(assignment.get("utility_inputs")),
            "utility_breakdown": deepcopy(assignment.get("utility_breakdown")),
            "route": route,
            "initial_eta": float(route["eta"]),
        }
        state.pop("_temporary_route_edges", None)


def _route_position(
    route: dict[str, Any], progress: float, nodes: dict[str, dict[str, float]]
) -> dict[str, float]:
    path = route["path"]
    if len(path) < 2 or any(node not in nodes for node in path):
        return deepcopy(nodes.get(path[0], {"x": 0.0, "y": 0.0}))
    lengths = [
        hypot(
            nodes[start]["x"] - nodes[end]["x"],
            nodes[start]["y"] - nodes[end]["y"],
        )
        for start, end in zip(path, path[1:])
    ]
    total = sum(lengths)
    if total <= 0:
        return deepcopy(nodes[path[-1]])
    remaining = max(0.0, min(1.0, progress)) * total
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            ratio = 0.0 if length <= 0 else remaining / length
            start = nodes[path[index]]
            end = nodes[path[index + 1]]
            return {
                "x": round(start["x"] + (end["x"] - start["x"]) * ratio, 6),
                "y": round(start["y"] + (end["y"] - start["y"]) * ratio, 6),
            }
        remaining -= length
    return deepcopy(nodes[path[-1]])


def _start_return_to_hospital(
    state: dict[str, Any], scenario: dict[str, Any], unit: dict[str, Any]
) -> None:
    previous_task = state.get("current_task") or {}
    hospital = scenario.get("hospital", {}).get("node_id")
    if not hospital:
        state["status"] = "idle"
        state["current_task"] = None
        state["completed_missions"] += 1
        return
    try:
        route = risk_aware_astar(
            _route_graph_for_unit(scenario["roads"], state["unit_id"]),
            nodes=scenario.get("nodes"),
            start=state["current_node"],
            goal=hospital,
            speed=float(unit["speed"]),
            risk_weights=scenario["config"]["weights"]["astar_risk"],
            unit_type=unit["type"],
            max_fire_risk=unit.get("constraints", {}).get("max_fire_risk"),
            route_layer="ground",
        )
    except NoRouteError:
        state["status"] = "stranded"
        return
    state["status"] = "returning"
    state["remaining_travel"] = float(route["eta"])
    state["current_task"] = {
        "mission_type": "medical_transport",
        "target_zone": None,
        "origin_zone": previous_task.get("target_zone"),
        "target_node": hospital,
        "estimated_people": state["onboard"],
        "route": route,
        "initial_eta": float(route["eta"]),
    }


def _complete_service_task(
    state: dict[str, Any], scenario: dict[str, Any], unit: dict[str, Any]
) -> None:
    task = state["current_task"]
    if task["mission_type"] == "reconnaissance":
        if task.get("target_zone") is not None:
            state["completed_targets"].append(task["target_zone"])
        state["status"] = "idle"
        state["current_task"] = None
        state["completed_missions"] += 1
        return

    if task.get("target_zone") is not None:
        state["completed_targets"].append(task["target_zone"])
    state["onboard"] = min(state["capacity"], task["estimated_people"])
    if state["onboard"]:
        _start_return_to_hospital(state, scenario, unit)
    else:
        state["status"] = "idle"
        state["current_task"] = None
        state["completed_missions"] += 1


def advance_unit_states(
    states: dict[str, dict[str, Any]], elapsed_minutes: float, scenario: dict[str, Any]
) -> None:
    if elapsed_minutes < 0:
        raise ValueError("elapsed_minutes must be nonnegative")
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    nodes = scenario.get("nodes", {})
    for unit_id, state in states.items():
        remaining_time = float(elapsed_minutes)
        unit = units[unit_id]
        while remaining_time > 1e-9 and state["status"] in {
            "en_route",
            "rescuing",
            "returning",
        }:
            if state["status"] in {"en_route", "returning"}:
                task = state["current_task"]
                travel = float(state["remaining_travel"])
                consumed = min(remaining_time, travel)
                state["remaining_travel"] = round(travel - consumed, 8)
                state["travel_minutes"] = round(
                    float(state.get("travel_minutes", 0.0)) + consumed, 8
                )
                remaining_time -= consumed
                initial_eta = max(float(task["initial_eta"]), 1e-9)
                progress = 1.0 - state["remaining_travel"] / initial_eta
                state["position"] = _route_position(task["route"], progress, nodes)
                if state["remaining_travel"] > 1e-9:
                    break
                state["current_node"] = task["target_node"]
                state["position"] = deepcopy(nodes.get(task["target_node"], state["position"]))
                if state["status"] == "returning":
                    origin_zone = task.get("origin_zone")
                    if origin_zone and origin_zone not in state["delivered_targets"]:
                        state["delivered_targets"].append(origin_zone)
                    state["rescued_people"] += state["onboard"]
                    state["onboard"] = 0
                    state["status"] = "idle"
                    state["current_task"] = None
                    state["completed_missions"] += 1
                else:
                    state["status"] = "rescuing"
                    if state["remaining_service"] <= 1e-9:
                        _complete_service_task(state, scenario, unit)
            elif state["status"] == "rescuing":
                service = float(state["remaining_service"])
                consumed = min(remaining_time, service)
                state["remaining_service"] = round(service - consumed, 8)
                remaining_time -= consumed
                if state["remaining_service"] > 1e-9:
                    break
                _complete_service_task(state, scenario, unit)
