from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class AEngineUnavailable(RuntimeError):
    """Raised when teammate A's algorithm package cannot be loaded or executed."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
A_ENGINE_SRC_CANDIDATES = (
    PROJECT_ROOT / "vendor" / "a_engine" / "src",
    PROJECT_ROOT / "external" / "zicheng_ai_emergency_commander" / "src",
)


def run_a_engine_on_grid(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run teammate A's inference, utility allocation, and risk-aware A* on B's grid.

    The Streamlit app keeps B's 24x24 tile map as the source of truth. This adapter
    turns every traversable grid cell into a graph node and every 4-neighbor move
    into a directed edge, then calls A's algorithm modules.
    """
    modules = _load_a_engine_modules()
    a_scenario = _grid_to_a_scenario(scenario)

    try:
        assessments = modules["assess_zones"](a_scenario)
        utility_matrix = modules["build_utility_matrix"](
            a_scenario,
            assessments,
            include_trace=False,
        )
        assignments_raw = modules["allocate_tasks"](
            a_scenario,
            utility_matrix,
            include_trace=False,
        )
    except Exception as exc:  # noqa: BLE001 - keep UI alive if external engine changes.
        raise AEngineUnavailable(str(exc)) from exc

    if isinstance(assignments_raw, dict):
        assignments_list = assignments_raw.get("assignments", [])
    else:
        assignments_list = assignments_raw

    assignments, routes, route_details = _convert_a_output_to_grid(
        a_scenario,
        scenario,
        assessments,
        assignments_list,
    )
    zone_scores = _zone_scores_from_assessments(assessments, scenario)

    return {
        "zone_scores": zone_scores,
        "assignments": assignments,
        "routes": routes,
        "route_details": route_details,
        "engine_summary": {
            "engine": "A 同学算法适配",
            "grid_size": f"{scenario['map']['width']}x{scenario['map']['height']}",
            "graph_nodes": len(a_scenario["nodes"]),
            "ground_edges": len(a_scenario["roads"]),
            "zones": len(a_scenario["zones"]),
            "utility_candidates": len(utility_matrix),
            "assigned_units": len(assignments),
            "note": "网格区块保持 B 端格式；概率、效用分配和风险 A* 来自 A 同学模块。",
        },
    }


def _load_a_engine_modules() -> dict[str, Any]:
    engine_src = next((path for path in A_ENGINE_SRC_CANDIDATES if path.exists()), None)
    if engine_src is None:
        raise AEngineUnavailable(
            "未找到 A 同学算法源码目录：vendor/a_engine/src 或 external/zicheng_ai_emergency_commander/src"
        )
    src_text = str(engine_src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    try:
        from emergency_commander.allocation import allocate_tasks, build_utility_matrix
        from emergency_commander.inference import assess_zones
        from emergency_commander.routing import NoRouteError, risk_aware_astar
    except Exception as exc:  # noqa: BLE001
        raise AEngineUnavailable(f"A 同学算法模块导入失败：{exc}") from exc

    return {
        "assess_zones": assess_zones,
        "build_utility_matrix": build_utility_matrix,
        "allocate_tasks": allocate_tasks,
        "risk_aware_astar": risk_aware_astar,
        "NoRouteError": NoRouteError,
    }


def _grid_to_a_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    map_data = scenario["map"]
    width = int(map_data["width"])
    height = int(map_data["height"])
    base = _as_cell(map_data["base"])
    hospital = _as_cell(map_data["hospital"])
    targets = {zone: _as_cell(point) for zone, point in map_data["targets"].items()}

    nodes = {
        _node_id((x, y)): {"x": float(x), "y": float(y)}
        for y in range(height)
        for x in range(width)
    }
    nodes["HQ"] = {"x": float(base[0]), "y": float(base[1])}
    nodes["HOSPITAL"] = {"x": float(hospital[0]), "y": float(hospital[1])}

    for zone, target in targets.items():
        nodes[f"ZONE_{zone}"] = {"x": float(target[0]), "y": float(target[1])}

    return {
        "scenario_id": "b_grid_adapter",
        "run_mode": "fixed",
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HOSPITAL"},
        "nodes": nodes,
        "config": {
            "weights": _a_engine_weights(),
            "thresholds": {
                "car_min_passability": 0.15,
                "drone_recon_priority_risk": 0.70,
            },
        },
        "zones": _convert_zones(scenario),
        "roads": _grid_edges(scenario, nodes),
        "air_routes": [],
        "units": _convert_units(scenario),
        "events": [],
    }


def _a_engine_weights() -> dict[str, dict[str, float]]:
    return {
        "trapped": {
            "sos": 0.35,
            "collapse": 0.30,
            "human_activity": 0.20,
            "smoke": 0.15,
        },
        "passability": {
            "road_damage": 0.42,
            "fire_risk": 0.34,
            "congestion": 0.18,
            "drone_confidence": 0.06,
        },
        "life_risk": {
            "fire": 0.40,
            "trapped_prob": 0.35,
            "time_urgency": 0.25,
        },
        "priority": {
            "trapped_prob": 0.40,
            "life_risk": 0.30,
            "time_urgency": 0.20,
            "accessibility": 0.10,
        },
        "utility": {
            "alpha": 0.46,
            "beta": 0.34,
            "gamma": 0.24,
            "delta": 0.10,
            "epsilon": 0.16,
            "zeta": 0.04,
        },
        "astar_risk": {
            "fire": 0.42,
            "damage": 0.24,
            "congestion": 0.46,
            "secondary": 0.36,
        },
    }


def _convert_zones(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    zones = []
    targets = scenario["map"]["targets"]
    for zone_id, evidence in scenario["zones"].items():
        target = _as_cell(targets[zone_id])
        urgency = float(evidence.get("urgency", evidence.get("smoke", 0.4)))
        observations = {
            "sos_signal": _clip(evidence.get("sos_signal", 0.0)),
            "building_collapse": _clip(evidence.get("building_collapse", 0.0)),
            "smoke": _clip(evidence.get("smoke", 0.0)),
            "fire": _clip(evidence.get("fire", 0.0)),
            "road_damage": _clip(evidence.get("road_damage", 0.0)),
            "human_activity": _clip(evidence.get("human_activity", 0.0)),
            "congestion": _clip(evidence.get("congestion", 0.0)),
            "time_urgency": _clip(urgency),
            "drone_confidence": 0.72,
        }
        observations["hazard_intensity"] = max(
            observations["building_collapse"],
            observations["fire"],
            observations["road_damage"],
        )
        zones.append(
            {
                "zone_id": zone_id,
                "node_id": _node_id(target),
                "observations": observations,
                "labels": {"grid_target": list(target)},
            }
        )
    return zones


def _convert_units(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    units = []
    for unit_id, detail in scenario["units"].items():
        is_drone = detail["type"] == "drone"
        start = _as_cell(detail["start"])
        units.append(
            {
                "unit_id": unit_id,
                "type": "drone" if is_drone else "rescue_car",
                "start_node": _node_id(start),
                "speed": float(detail.get("speed", 1.0)),
                "can_transport": not is_drone,
                "capacity": 0 if is_drone else int(detail.get("capacity", 4)),
                "service_time": 0.5 if is_drone else 1.0,
                "resource_cost": 0.25 if is_drone else 0.55,
                "constraints": {
                    "max_fire_risk": 0.95 if not is_drone else 1.0,
                    "min_passability": 0.15 if not is_drone else 0.0,
                },
            }
        )
    return units


def _grid_edges(scenario: dict[str, Any], nodes: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    map_data = scenario["map"]
    width = int(map_data["width"])
    height = int(map_data["height"])
    base = _as_cell(map_data["base"])
    hospital = _as_cell(map_data["hospital"])
    targets = {_as_cell(point) for point in map_data.get("targets", {}).values()}
    protected = {base, hospital, *targets}
    blocked = _cell_set(map_data.get("blocked", []))
    buildings = _cell_set(map_data.get("buildings", []))
    water = _cell_set(map_data.get("water", []))
    impassable = (blocked | buildings | water) - protected

    edges = []
    for y in range(height):
        for x in range(width):
            start = (x, y)
            if start in impassable:
                continue
            for end in _neighbors(start, width, height):
                if end in impassable:
                    continue
                edges.append(_edge_for_move(scenario, start, end, nodes))
    return edges


def _edge_for_move(
    scenario: dict[str, Any],
    start: tuple[int, int],
    end: tuple[int, int],
    nodes: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "road_id": f"grid_{start[0]}_{start[1]}__{end[0]}_{end[1]}",
        "from": _node_id(start),
        "to": _node_id(end),
        "distance": 1.0,
        "travel_time_base": _base_travel_time(scenario, end),
        "status": "open",
        "bidirectional": False,
        "risk": _risk_for_cell(scenario, end),
        "labels": {
            "grid_from": list(start),
            "grid_to": list(end),
            "terrain": _terrain_for_cell(scenario, end),
        },
    }


def _base_travel_time(scenario: dict[str, Any], cell: tuple[int, int]) -> float:
    map_data = scenario["map"]
    roads = _cell_set(map_data.get("roads", []))
    park = _cell_set(map_data.get("park", []))
    congestion = _cell_set(map_data.get("congestion", []))
    collapse = _cell_set(map_data.get("collapse_cells", []))
    fire = _cell_set(map_data.get("fire", []))

    if cell in roads:
        base = 1.0
    elif cell in park:
        base = 1.5
    else:
        base = 1.8
    if cell in congestion:
        base += 0.7
    if cell in collapse:
        base += 0.9
    if cell in fire:
        base += 1.2
    return round(base, 3)


def _risk_for_cell(scenario: dict[str, Any], cell: tuple[int, int]) -> dict[str, float]:
    map_data = scenario["map"]
    fire = _cell_set(map_data.get("fire", []))
    congestion = _cell_set(map_data.get("congestion", []))
    collapse = _cell_set(map_data.get("collapse_cells", []))
    blocked = _cell_set(map_data.get("blocked", []))
    roads = _cell_set(map_data.get("roads", []))

    return {
        "fire": 1.0 if cell in fire else 0.08,
        "damage": 0.95 if cell in blocked else (0.72 if cell in collapse else 0.18),
        "congestion": 0.95 if cell in congestion else (0.12 if cell in roads else 0.28),
        "secondary_disaster": 0.88 if cell in collapse else (0.35 if cell in fire else 0.12),
    }


def _terrain_for_cell(scenario: dict[str, Any], cell: tuple[int, int]) -> str:
    map_data = scenario["map"]
    if cell in _cell_set(map_data.get("roads", [])):
        return "road"
    if cell in _cell_set(map_data.get("park", [])):
        return "park"
    return "ground"


def _zone_scores_from_assessments(
    assessments: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> dict[str, dict[str, float]]:
    score_map = {}
    for assessment in assessments:
        zone_id = assessment["zone_id"]
        evidence = scenario["zones"].get(zone_id, {})
        score_map[zone_id] = {
            "trapped_probability": round(float(assessment["trapped_prob"]), 3),
            "road_accessibility": round(float(assessment["passability_prob"]), 3),
            "life_risk": round(float(assessment["life_risk"]), 3),
            "urgency": round(float(evidence.get("urgency", evidence.get("smoke", 0.4))), 3),
            "priority": round(float(assessment["priority_score"]) * 100.0, 1),
        }
    return score_map


def _convert_a_output_to_grid(
    a_scenario: dict[str, Any],
    grid_scenario: dict[str, Any],
    assessments: list[dict[str, Any]],
    assignments_list: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[list[int]]], dict[str, dict[str, Any]]]:
    assignments: dict[str, str] = {}
    routes: dict[str, list[list[int]]] = {}
    route_details: dict[str, dict[str, Any]] = {}

    for item in sorted(assignments_list, key=lambda row: row.get("unit_id", "")):
        unit_id = item.get("unit_id")
        target_zone = item.get("target_zone")
        route = item.get("route") or {}
        if not unit_id or not target_zone or not route:
            continue
        assignments[unit_id] = target_zone
        routes[unit_id] = _path_nodes_to_grid_points(a_scenario, route.get("path", []))
        route_details[unit_id] = _route_detail_from_assignment(item, route)

    _fill_missing_unit_routes(
        a_scenario,
        grid_scenario,
        assessments,
        assignments,
        routes,
        route_details,
    )
    return assignments, routes, route_details


def _route_detail_from_assignment(
    assignment: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any]:
    return {
        "engine": "A 同学算法适配",
        "eta": round(float(route.get("eta", 0.0)), 2),
        "path_risk": round(float(route.get("path_risk", 0.0)), 3),
        "total_cost": round(float(route.get("total_cost", 0.0)), 2),
        "route_layer": route.get("route_layer", "ground"),
        "expanded_nodes": route.get("expanded_nodes", 0),
        "expected_utility": assignment.get("expected_utility"),
        "feasibility_reason": assignment.get("feasibility_reason", assignment.get("reason", "")),
        "reason": assignment.get("reason", ""),
        "utility_breakdown": assignment.get("utility_breakdown"),
    }


def _fill_missing_unit_routes(
    a_scenario: dict[str, Any],
    grid_scenario: dict[str, Any],
    assessments: list[dict[str, Any]],
    assignments: dict[str, str],
    routes: dict[str, list[list[int]]],
    route_details: dict[str, dict[str, Any]],
) -> None:
    missing_units = [
        unit_id
        for unit_id in grid_scenario["units"]
        if unit_id not in assignments
    ]
    if not missing_units:
        return
    ranked_zones = [row["zone_id"] for row in assessments]
    used_zones = set(assignments.values())
    for unit_id in missing_units:
        if grid_scenario["units"][unit_id]["type"] != "drone":
            continue
        target_zone = next(
            (zone for zone in ranked_zones if zone not in used_zones),
            ranked_zones[0] if ranked_zones else "",
        )
        if not target_zone:
            continue
        target_cell = _as_cell(grid_scenario["map"]["targets"][target_zone])
        start_cell = _as_cell(grid_scenario["units"][unit_id]["start"])
        assignments[unit_id] = target_zone
        routes[unit_id] = [list(start_cell), list(target_cell)]
        route_details[unit_id] = {
            "engine": "A 同学算法适配",
            "eta": 0.0,
            "path_risk": 0.0,
            "total_cost": 0.0,
            "route_layer": "fallback",
            "expanded_nodes": 0,
            "reason": "A 引擎未给该单位分配任务，适配层仅补齐展示路线。",
        }
        used_zones.add(target_zone)


def _path_nodes_to_grid_points(
    a_scenario: dict[str, Any],
    path_nodes: list[str],
) -> list[list[int]]:
    points: list[list[int]] = []
    for node in path_nodes:
        point = _point_from_node(a_scenario, node)
        if point and (not points or points[-1] != point):
            points.append(point)
    return points


def _point_from_node(a_scenario: dict[str, Any], node_id: str) -> list[int] | None:
    if node_id.startswith("n_"):
        parts = node_id.split("_")
        if len(parts) == 3:
            return [int(parts[1]), int(parts[2])]
    node = a_scenario["nodes"].get(node_id)
    if not node:
        return None
    return [int(round(float(node["x"]))), int(round(float(node["y"])))]


def _neighbors(cell: tuple[int, int], width: int, height: int) -> list[tuple[int, int]]:
    x, y = cell
    return [
        (nx, ny)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= nx < width and 0 <= ny < height
    ]


def _cell_set(cells: Any) -> set[tuple[int, int]]:
    return {_as_cell(cell) for cell in cells or []}


def _as_cell(value: list[int] | tuple[int, int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _node_id(cell: tuple[int, int]) -> str:
    return f"n_{cell[0]}_{cell[1]}"


def _clip(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
