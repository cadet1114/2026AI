from __future__ import annotations

from itertools import product
from math import hypot
from typing import Any

from emergency_commander.routing import NoRouteError, risk_aware_astar


def _zone_lookup(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {zone["zone_id"]: zone for zone in scenario["zones"]}


def _resource_cost(unit: dict[str, Any]) -> float:
    default = 0.25 if unit["type"] == "drone" else 0.55
    return max(0.0, min(1.0, float(unit.get("resource_cost", default))))


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
    if any(road_id.startswith("__unit_") for road_id in road_ids):
        route["_edge_road_ids"] = list(road_ids)
        route["road_ids"] = [
            road_id for road_id in road_ids if not road_id.startswith("__unit_")
        ]
    return route


def _rejection_explanation(unit_id: str, zone_id: str, reason: str) -> str:
    labels = {
        "passability_below_vehicle_minimum": "道路可通行概率低于车辆安全阈值",
        "fire_risk_above_vehicle_maximum": "区域火灾风险超过车辆承受上限",
        "no_air_route": "空中航线不可达",
        "no_ground_route": "地面道路不可达",
    }
    return f"{unit_id} 未分配至 {zone_id}区：{labels.get(reason, reason)}。"


def _utility_explanation(
    unit_id: str,
    zone_id: str,
    breakdown: dict[str, float],
    expected_utility: float,
) -> str:
    benefits = (
        breakdown["trapped_benefit"]
        + breakdown["life_risk_benefit"]
        + breakdown["accessibility_benefit"]
    )
    costs = -(
        breakdown["arrival_time_cost"]
        + breakdown["path_risk_cost"]
        + breakdown["resource_cost"]
    )
    return (
        f"派遣 {unit_id} 前往 {zone_id}区：救援收益 {benefits:.3f}，"
        f"到达、路径与资源成本合计 {costs:.3f}，总期望效用 {expected_utility:.3f}。"
    )


def _route_with_risk_fallback(
    route_graph: list[dict[str, Any]],
    *,
    nodes: dict[str, dict[str, float]] | None,
    start: str,
    goal: str,
    speed: float,
    risk_weights: dict[str, float],
    unit_type: str,
    max_fire_risk: float | None,
    route_layer: str,
    include_trace: bool,
) -> tuple[dict[str, Any], str]:
    try:
        route = risk_aware_astar(
            route_graph,
            nodes=nodes,
            start=start,
            goal=goal,
            speed=speed,
            risk_weights=risk_weights,
            unit_type=unit_type,
            max_fire_risk=max_fire_risk,
            route_layer=route_layer,
            include_trace=include_trace,
        )
        route["risk_policy"] = "standard"
        return route, "feasible"
    except NoRouteError:
        if (
            unit_type == "drone"
            or max_fire_risk is None
            or not any(road.get("status") == "blocked" for road in route_graph)
        ):
            raise
    route = risk_aware_astar(
        route_graph,
        nodes=nodes,
        start=start,
        goal=goal,
        speed=speed,
        risk_weights=risk_weights,
        unit_type=unit_type,
        max_fire_risk=None,
        route_layer=route_layer,
        include_trace=include_trace,
    )
    route["risk_policy"] = "relaxed_fire_limit"
    route["relaxed_constraints"] = ["max_fire_risk"]
    return route, "feasible_with_risk_override"


def _direct_air_route(
    nodes: dict[str, dict[str, float]] | None,
    *,
    start: str,
    goal: str,
    speed: float,
    include_trace: bool,
) -> dict[str, Any]:
    if speed <= 0:
        raise ValueError("speed must be positive")
    if start == goal:
        result = {
            "path": [start],
            "road_ids": [],
            "eta": 0.0,
            "path_risk": 0.0,
            "total_cost": 0.0,
            "route_layer": "air",
            "heuristic": "direct_air",
            "heuristic_start": 0.0,
            "expanded_nodes": 0,
        }
        if include_trace:
            result["search_trace"] = []
        return result
    if not nodes or start not in nodes or goal not in nodes:
        raise NoRouteError(
            f"missing coordinates for direct air route from {start} to {goal}"
        )

    distance = hypot(
        float(nodes[start]["x"]) - float(nodes[goal]["x"]),
        float(nodes[start]["y"]) - float(nodes[goal]["y"]),
    )
    eta = distance / speed
    result = {
        "path": [start, goal],
        "road_ids": [],
        "eta": round(eta, 6),
        "path_risk": 0.0,
        "total_cost": round(eta, 6),
        "route_layer": "air",
        "heuristic": "direct_air",
        "heuristic_start": round(eta, 6),
        "expanded_nodes": 1,
    }
    if include_trace:
        result["search_trace"] = [
            {
                "node": start,
                "g": 0.0,
                "h": round(eta, 6),
                "f": round(eta, 6),
                "frontier_size": 1,
                "relaxations": [
                    {
                        "neighbor": goal,
                        "road_id": "direct_air",
                        "g": round(eta, 6),
                        "h": 0.0,
                        "f": round(eta, 6),
                        "edge_cost": round(eta, 6),
                    }
                ],
            },
            {
                "node": goal,
                "g": round(eta, 6),
                "h": 0.0,
                "f": round(eta, 6),
                "frontier_size": 0,
                "relaxations": [],
            },
        ]
    return result


def build_utility_matrix(
    scenario: dict[str, Any],
    assessments: list[dict[str, Any]],
    *,
    include_trace: bool = False,
) -> list[dict[str, Any]]:
    """Build explainable unit-zone candidates, including rejected options."""
    utility_weights = scenario["config"]["weights"]["utility"]
    risk_weights = scenario["config"]["weights"]["astar_risk"]
    zones = _zone_lookup(scenario)
    candidates: list[dict[str, Any]] = []

    for unit in scenario["units"]:
        is_drone = unit["type"] == "drone"
        unit_resource_cost = _resource_cost(unit)
        for assessment in assessments:
            zone = zones[assessment["zone_id"]]
            constraints = unit.get("constraints", {})
            feasible = True
            reason = "feasible"
            if not is_drone and assessment["passability_prob"] < constraints.get("min_passability", 0.45):
                feasible = False
                reason = "passability_below_vehicle_minimum"
            elif not is_drone and zone["observations"]["fire"] > constraints.get("max_fire_risk", 1.0):
                feasible = False
                reason = "fire_risk_above_vehicle_maximum"

            route = None
            if feasible:
                try:
                    if is_drone:
                        route = _direct_air_route(
                            scenario.get("nodes"),
                            start=unit["start_node"],
                            goal=assessment["node_id"],
                            speed=float(unit["speed"]),
                            include_trace=include_trace,
                        )
                        reason = "direct_air_route"
                    else:
                        route_graph = _route_graph_for_unit(
                            [
                                *scenario["roads"],
                                *unit.get("_temporary_route_edges", []),
                            ],
                            unit["unit_id"],
                        )
                        route, reason = _route_with_risk_fallback(
                            route_graph,
                            nodes=scenario.get("nodes"),
                            start=unit["start_node"],
                            goal=assessment["node_id"],
                            speed=float(unit["speed"]),
                            risk_weights=risk_weights,
                            unit_type=unit["type"],
                            max_fire_risk=constraints.get("max_fire_risk"),
                            route_layer="ground",
                            include_trace=include_trace,
                        )
                        _hide_temporary_route_edges(route)
                except NoRouteError:
                    feasible = False
                    reason = "no_air_route" if is_drone else "no_ground_route"

            expected_utility = None
            utility_inputs = None
            utility_breakdown = None
            explanation = _rejection_explanation(
                unit["unit_id"], assessment["zone_id"], reason
            )
            if feasible and route is not None:
                accessibility_value = (
                    1.0 - assessment["passability_prob"] if is_drone else assessment["passability_prob"]
                )
                arrival_time_normalized = min(route["eta"] / 20.0, 1.0)
                utility_inputs = {
                    "trapped_prob": round(assessment["trapped_prob"], 6),
                    "life_risk": round(assessment["life_risk"], 6),
                    "accessibility": round(accessibility_value, 6),
                    "arrival_time_normalized": round(arrival_time_normalized, 6),
                    "path_risk": round(route["path_risk"], 6),
                    "resource_cost": round(unit_resource_cost, 6),
                }
                raw_breakdown = {
                    "trapped_benefit": utility_weights["alpha"] * assessment["trapped_prob"],
                    "life_risk_benefit": utility_weights["beta"] * assessment["life_risk"],
                    "accessibility_benefit": utility_weights["gamma"] * accessibility_value,
                    "arrival_time_cost": -utility_weights["delta"] * arrival_time_normalized,
                    "path_risk_cost": -utility_weights["epsilon"] * route["path_risk"],
                    "resource_cost": -utility_weights.get("zeta", 0.10) * unit_resource_cost,
                }
                utility_breakdown = {
                    name: round(value, 6) for name, value in raw_breakdown.items()
                }
                expected_utility = round(sum(utility_breakdown.values()), 6)
                explanation = _utility_explanation(
                    unit["unit_id"], assessment["zone_id"], utility_breakdown, expected_utility
                )

            candidates.append(
                {
                    "unit_id": unit["unit_id"],
                    "unit_type": unit["type"],
                    "target_zone": assessment["zone_id"],
                    "mission_type": "reconnaissance" if is_drone else "rescue",
                    "feasible": feasible,
                    "reason": reason,
                    "explanation": explanation,
                    "resource_cost": round(unit_resource_cost, 6),
                    "utility_inputs": utility_inputs,
                    "utility_breakdown": utility_breakdown,
                    "expected_utility": expected_utility,
                    "route": route,
                }
            )

    return candidates


def allocate_tasks(
    scenario: dict[str, Any],
    utility_matrix: list[dict[str, Any]],
    *,
    include_trace: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Maximize total utility by enumerating the small demo assignment space."""
    by_unit: dict[str, list[dict[str, Any] | None]] = {}
    for unit in scenario["units"]:
        unit_candidates = [
            item
            for item in utility_matrix
            if item["unit_id"] == unit["unit_id"] and item["feasible"]
        ]
        by_unit[unit["unit_id"]] = [None, *unit_candidates]

    unit_ids = [unit["unit_id"] for unit in scenario["units"]]
    best_total = float("-inf")
    best_combination: tuple[dict[str, Any] | None, ...] | None = None
    considered = 0
    duplicate_zone_rejections = 0
    ranked_combinations: list[dict[str, Any]] = []
    for combination in product(*(by_unit[unit_id] for unit_id in unit_ids)):
        considered += 1
        zones = [item["target_zone"] for item in combination if item is not None]
        if len(zones) != len(set(zones)):
            duplicate_zone_rejections += 1
            continue
        total = sum(item["expected_utility"] for item in combination if item is not None)
        ranked_combinations.append(
            {
                "assignments": [
                    {
                        "unit_id": item["unit_id"],
                        "target_zone": item["target_zone"],
                    }
                    for item in combination
                    if item is not None
                ],
                "total": round(total, 6),
            }
        )
        if total > best_total:
            best_total = total
            best_combination = combination

    assignments: list[dict[str, Any]] = []
    if best_combination is None:
        if not include_trace:
            return assignments
        return {
            "assignments": assignments,
            "trace": {
                "considered": considered,
                "duplicate_zone_rejections": duplicate_zone_rejections,
                "ranked_combinations": [],
                "winning_total": None,
            },
        }
    for item in best_combination:
        if item is None:
            continue
        assignments.append(
            {
                "unit_id": item["unit_id"],
                "target_zone": item["target_zone"],
                "mission_type": item["mission_type"],
                "expected_utility": item["expected_utility"],
                "reason": item["explanation"],
                "feasibility_reason": item["reason"],
                "explanation": item["explanation"],
                "resource_cost": item["resource_cost"],
                "utility_inputs": item["utility_inputs"],
                "utility_breakdown": item["utility_breakdown"],
                "route": item["route"],
            }
        )
    if not include_trace:
        return assignments
    ranked_combinations.sort(key=lambda item: item["total"], reverse=True)
    return {
        "assignments": assignments,
        "trace": {
            "considered": considered,
            "duplicate_zone_rejections": duplicate_zone_rejections,
            "ranked_combinations": ranked_combinations[:12],
            "winning_total": round(best_total, 6),
        },
    }
