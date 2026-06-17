from __future__ import annotations

import copy
import heapq
import json
import random
from pathlib import Path
from typing import Any


Coordinate = tuple[int, int]


ROUTE_COLORS = {
    "RescueCar-1": "#d62728",
    "RescueCar-2": "#1f77b4",
    "Drone-1": "#2ca02c",
}

ALLOWED_ZONE_FIELDS = {
    "sos_signal",
    "building_collapse",
    "smoke",
    "fire",
    "road_damage",
    "human_activity",
    "urgency",
    "congestion",
}

CELL_UPDATE_TARGETS = {
    "add_blocked_cells": ("blocked", "add"),
    "remove_blocked_cells": ("blocked", "remove"),
    "add_fire_cells": ("fire", "add"),
    "remove_fire_cells": ("fire", "remove"),
    "add_congestion_cells": ("congestion", "add"),
    "remove_congestion_cells": ("congestion", "remove"),
    "add_collapse_cells": ("collapse_cells", "add"),
    "remove_collapse_cells": ("collapse_cells", "remove"),
}


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load the preset emergency scenario used by the Streamlit demo."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def clone_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(scenario)


def generate_random_scenario(seed: int | str | None = None) -> dict[str, Any]:
    """Generate a validated 24x24 disaster scenario for non-hardcoded demos."""
    rng_seed: int | str
    if seed is None:
        rng_seed = random.SystemRandom().randint(1, 2_147_483_647)
    else:
        rng_seed = seed

    for attempt in range(1, 101):
        rng = random.Random(f"{rng_seed}:{attempt}")
        scenario = _build_random_scenario(rng)
        if _random_scenario_has_valid_routes(scenario):
            return scenario

    fallback_path = Path(__file__).resolve().parents[1] / "data" / "scenario.json"
    return load_scenario(fallback_path)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_zone_scores(scenario: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute the demo's explainable risk and priority scores."""
    scores: dict[str, dict[str, float]] = {}

    for zone_name, evidence in scenario["zones"].items():
        trapped_probability = clamp(
            0.35 * evidence.get("sos_signal", 0.0)
            + 0.30 * evidence.get("building_collapse", 0.0)
            + 0.20 * evidence.get("human_activity", 0.0)
            + 0.15 * evidence.get("smoke", 0.0)
        )
        road_accessibility = clamp(
            1
            - (
                0.45 * evidence.get("road_damage", 0.0)
                + 0.35 * evidence.get("fire", 0.0)
                + 0.20 * evidence.get("congestion", 0.0)
            )
        )
        urgency = clamp(evidence.get("urgency", evidence.get("smoke", 0.4)))
        life_risk = clamp(
            0.40 * evidence.get("fire", 0.0)
            + 0.35 * trapped_probability
            + 0.25 * urgency
        )
        priority = clamp(
            0.40 * trapped_probability
            + 0.30 * life_risk
            + 0.20 * urgency
            + 0.10 * road_accessibility
        )

        scores[zone_name] = {
            "trapped_probability": round(trapped_probability, 3),
            "road_accessibility": round(road_accessibility, 3),
            "life_risk": round(life_risk, 3),
            "urgency": round(urgency, 3),
            "priority": round(priority * 100, 1),
        }

    return scores


def assign_tasks(
    scenario: dict[str, Any], zone_scores: dict[str, dict[str, float]]
) -> dict[str, str]:
    """Assign two rescue cars and one drone with stable demo-oriented rules."""
    assignments: dict[str, str] = {}
    remaining_zones = set(zone_scores)

    drone_target = max(
        remaining_zones,
        key=lambda zone: (
            scenario["zones"][zone].get("fire", 0.0)
            + (1 - zone_scores[zone]["road_accessibility"])
            + zone_scores[zone]["life_risk"]
        ),
    )
    assignments["Drone-1"] = drone_target
    remaining_zones.remove(drone_target)

    car_targets = sorted(
        remaining_zones,
        key=lambda zone: (
            zone_scores[zone]["priority"],
            zone_scores[zone]["road_accessibility"],
        ),
        reverse=True,
    )
    for unit, fallback_index in (("RescueCar-1", 0), ("RescueCar-2", 1)):
        if car_targets:
            assignments[unit] = car_targets.pop(0)
        else:
            ranked = sorted(
                zone_scores,
                key=lambda zone: zone_scores[zone]["priority"],
                reverse=True,
            )
            assignments[unit] = ranked[min(fallback_index, len(ranked) - 1)]

    return {
        "RescueCar-1": assignments["RescueCar-1"],
        "RescueCar-2": assignments["RescueCar-2"],
        "Drone-1": assignments["Drone-1"],
    }


def plan_routes(
    scenario: dict[str, Any], assignments: dict[str, str]
) -> dict[str, list[list[int]]]:
    routes: dict[str, list[list[int]]] = {}
    targets = scenario["map"]["targets"]

    for unit_name, zone_name in assignments.items():
        unit = scenario["units"][unit_name]
        start = _as_coordinate(unit["start"])
        goal = _as_coordinate(targets[zone_name])
        unit_type = unit["type"]
        path = _a_star(scenario, start, goal, unit_type)
        routes[unit_name] = [[x, y] for x, y in path]

    return routes


def calculate_route_cost(
    scenario: dict[str, Any],
    unit_name: str,
    route: list[list[int]],
) -> float:
    """Calculate the same terrain-aware movement cost used by A*."""
    if not route:
        return 0.0

    unit_type = scenario["units"][unit_name]["type"]
    blocked = {_as_coordinate(cell) for cell in scenario["map"].get("blocked", [])}
    buildings = {_as_coordinate(cell) for cell in scenario["map"].get("buildings", [])}
    water = {_as_coordinate(cell) for cell in scenario["map"].get("water", [])}
    roads = {_as_coordinate(cell) for cell in scenario["map"].get("roads", [])}
    blocked.update(buildings)
    blocked.update(water)
    fire = {_as_coordinate(cell) for cell in scenario["map"].get("fire", [])}
    congestion = {_as_coordinate(cell) for cell in scenario["map"].get("congestion", [])}
    collapse = {_as_coordinate(cell) for cell in scenario["map"].get("collapse_cells", [])}

    total = 0.0
    for raw_cell in route[1:]:
        cell = _as_coordinate(raw_cell)
        if unit_type == "car" and cell in blocked:
            return float("inf")
        total += _cell_cost(
            cell,
            fire,
            congestion,
            blocked,
            collapse,
            roads,
            unit_type,
        )

    return round(total, 2)


def apply_road_collapse(scenario: dict[str, Any]) -> dict[str, Any]:
    updated = clone_scenario(scenario)
    blocked = {_as_coordinate(cell) for cell in updated["map"].get("blocked", [])}
    blocked.update(_as_coordinate(cell) for cell in updated["map"].get("collapse_cells", []))
    updated["map"]["blocked"] = [[x, y] for x, y in sorted(blocked)]
    updated["map"]["collapsed"] = True
    return updated


def apply_update_json_to_scenario(
    scenario: dict[str, Any], update_json: dict[str, Any]
) -> dict[str, Any]:
    """Apply structured disaster updates without mutating the original scenario."""
    updated = clone_scenario(scenario)
    map_data = updated["map"]
    width = int(map_data.get("width", 10))
    height = int(map_data.get("height", 10))

    for update in update_json.get("updates", []):
        if not isinstance(update, dict):
            continue
        update_type = update.get("type")

        if update_type == "target_update":
            target = update.get("target")
            fields = update.get("fields", {})
            if target not in updated.get("zones", {}) or not isinstance(fields, dict):
                continue
            for key, value in fields.items():
                if key in ALLOWED_ZONE_FIELDS and isinstance(value, (int, float)):
                    updated["zones"][target][key] = round(clamp(float(value)), 3)
            continue

        if update_type in CELL_UPDATE_TARGETS:
            field, action = CELL_UPDATE_TARGETS[update_type]
            cells = _valid_cells(update.get("cells", []), width, height)
            if action == "add":
                _add_cells(map_data, field, cells)
            else:
                _remove_cells(map_data, field, cells)

    return updated


def summarize_update_json(update_json: dict[str, Any]) -> str:
    """Create a short human-readable summary for the latest update."""
    parts: list[str] = []
    for update in update_json.get("updates", []):
        if not isinstance(update, dict):
            continue
        update_type = update.get("type")
        if update_type == "target_update":
            target = update.get("target")
            fields = update.get("fields", {})
            if target and isinstance(fields, dict):
                changed = "，".join(
                    f"{key}={value:.2f}" if isinstance(value, (int, float)) else f"{key}={value}"
                    for key, value in fields.items()
                )
                parts.append(f"{target}区指标更新：{changed}")
        elif update_type in CELL_UPDATE_TARGETS:
            cells = update.get("cells", [])
            if cells:
                field, action = CELL_UPDATE_TARGETS[update_type]
                action_text = "新增" if action == "add" else "移除"
                parts.append(f"{action_text}{field}格 {len(cells)} 个：{cells}")
    return "；".join(parts) if parts else "已接收更新，但没有可应用的有效字段。"


def build_demo_state(
    scenario: dict[str, Any], include_routes: bool = True
) -> dict[str, Any]:
    zone_scores = compute_zone_scores(scenario)
    assignments = assign_tasks(scenario, zone_scores)
    routes = plan_routes(scenario, assignments) if include_routes else {}
    return {
        "zone_scores": zone_scores,
        "assignments": assignments,
        "routes": routes,
        "report_text": generate_report(scenario, zone_scores, assignments, routes),
    }


def _build_random_scenario(rng: random.Random) -> dict[str, Any]:
    width = 24
    height = 24
    base = (rng.randint(1, 3), rng.randint(1, 3))
    hospital = (rng.randint(20, 22), rng.randint(20, 22))

    reserved = {base, hospital}
    targets = {
        "A": _random_point(rng, range(15, 21), range(5, 10), reserved),
        "B": _random_point(rng, range(6, 12), range(15, 21), reserved),
        "C": _random_point(rng, range(17, 22), range(12, 17), reserved),
    }
    reserved.update(targets.values())

    roads: set[Coordinate] = set()
    low_hub = (rng.randint(5, 7), rng.randint(2, 4))
    mid_hub = (rng.randint(6, 9), rng.randint(10, 13))
    high_hub = (rng.randint(14, 17), rng.randint(13, 17))
    upper_hub = (rng.randint(17, 21), rng.randint(18, 21))

    for start, end in (
        (base, low_hub),
        (low_hub, mid_hub),
        (mid_hub, high_hub),
        (high_hub, upper_hub),
        (upper_hub, hospital),
        (low_hub, targets["A"]),
        (mid_hub, targets["B"]),
        (high_hub, targets["C"]),
    ):
        _add_manhattan_road(roads, start, end, rng)

    for _ in range(rng.randint(2, 4)):
        y = rng.randint(5, 20)
        x1 = rng.randint(1, 8)
        x2 = rng.randint(15, 22)
        _add_axis_road(roads, (x1, y), (x2, y))

    for _ in range(rng.randint(1, 3)):
        x = rng.randint(5, 20)
        y1 = rng.randint(2, 8)
        y2 = rng.randint(15, 22)
        _add_axis_road(roads, (x, y1), (x, y2))

    r2_start = _nearby_start_cell(base, width, height)
    roads.add(r2_start)
    protected = reserved | {r2_start}

    water = _random_water_cells(rng, width, height, protected | roads)
    park = _random_park_cells(rng, width, height, protected | roads | water)
    buildings = _random_building_cells(
        rng,
        width,
        height,
        protected | roads | water,
        rng.randint(42, 66),
    )

    fire = _random_hazard_cells_near_targets(
        rng,
        targets,
        rng.randint(5, 11),
        width,
        height,
        protected | water | buildings,
    )
    congestion = _sample_layer_cells(
        rng,
        [cell for cell in roads if cell not in protected and cell not in fire],
        rng.randint(7, 14),
    )
    collapse_candidates = _collapse_candidates(width, height, roads, buildings)
    collapse_cells = _sample_layer_cells(
        rng,
        [
            cell
            for cell in collapse_candidates
            if cell not in protected
            and cell not in water
            and cell not in buildings
            and cell not in fire
            and cell not in congestion
        ],
        rng.randint(4, 9),
    )
    blocked = _sample_layer_cells(
        rng,
        [
            cell
            for cell in roads
            if cell not in protected
            and cell not in fire
            and cell not in congestion
            and cell not in collapse_cells
        ],
        rng.randint(2, 5),
    )

    return {
        "zones": _random_zone_evidence(rng),
        "units": {
            "RescueCar-1": {
                "type": "car",
                "start": list(base),
                "speed": 1.15,
            },
            "RescueCar-2": {
                "type": "car",
                "start": list(r2_start),
                "speed": 1.0,
            },
            "Drone-1": {
                "type": "drone",
                "start": list(base),
                "speed": 1.8,
            },
        },
        "map": {
            "width": width,
            "height": height,
            "base": list(base),
            "hospital": list(hospital),
            "targets": {zone: list(point) for zone, point in targets.items()},
            "roads": _cells_to_json(roads),
            "buildings": _cells_to_json(buildings),
            "water": _cells_to_json(water),
            "park": _cells_to_json(park),
            "blocked": _cells_to_json(blocked),
            "fire": _cells_to_json(fire),
            "congestion": _cells_to_json(congestion),
            "collapse_cells": _cells_to_json(collapse_cells),
        },
    }


def _random_scenario_has_valid_routes(scenario: dict[str, Any]) -> bool:
    zone_scores = compute_zone_scores(scenario)
    assignments = assign_tasks(scenario, zone_scores)
    routes = plan_routes(scenario, assignments)

    valid_routes = 0
    valid_car_routes = 0
    targets = scenario["map"]["targets"]
    for unit_name, zone_name in assignments.items():
        route = routes.get(unit_name, [])
        if len(route) <= 1:
            continue
        if _as_coordinate(route[-1]) != _as_coordinate(targets[zone_name]):
            continue
        valid_routes += 1
        if scenario["units"][unit_name]["type"] == "car":
            valid_car_routes += 1

    return valid_routes >= 2 and valid_car_routes >= 1


def _random_zone_evidence(rng: random.Random) -> dict[str, dict[str, float]]:
    return {
        "A": {
            "sos_signal": _rand_score(rng, 0.76, 0.98),
            "building_collapse": _rand_score(rng, 0.62, 0.95),
            "smoke": _rand_score(rng, 0.45, 0.82),
            "fire": _rand_score(rng, 0.35, 0.78),
            "road_damage": _rand_score(rng, 0.35, 0.75),
            "human_activity": _rand_score(rng, 0.45, 0.82),
            "urgency": _rand_score(rng, 0.68, 0.96),
            "congestion": _rand_score(rng, 0.18, 0.55),
        },
        "B": {
            "sos_signal": _rand_score(rng, 0.20, 0.68),
            "building_collapse": _rand_score(rng, 0.15, 0.58),
            "smoke": _rand_score(rng, 0.12, 0.62),
            "fire": _rand_score(rng, 0.08, 0.55),
            "road_damage": _rand_score(rng, 0.10, 0.48),
            "human_activity": _rand_score(rng, 0.20, 0.64),
            "urgency": _rand_score(rng, 0.25, 0.70),
            "congestion": _rand_score(rng, 0.05, 0.42),
        },
        "C": {
            "sos_signal": _rand_score(rng, 0.45, 0.90),
            "building_collapse": _rand_score(rng, 0.30, 0.76),
            "smoke": _rand_score(rng, 0.66, 0.98),
            "fire": _rand_score(rng, 0.66, 0.98),
            "road_damage": _rand_score(rng, 0.40, 0.86),
            "human_activity": _rand_score(rng, 0.34, 0.76),
            "urgency": _rand_score(rng, 0.66, 0.96),
            "congestion": _rand_score(rng, 0.16, 0.58),
        },
    }


def _rand_score(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 3)


def _random_point(
    rng: random.Random,
    x_range: range,
    y_range: range,
    reserved: set[Coordinate],
) -> Coordinate:
    for _ in range(100):
        point = (rng.choice(list(x_range)), rng.choice(list(y_range)))
        if point not in reserved:
            return point
    return (x_range.start, y_range.start)


def _nearby_start_cell(base: Coordinate, width: int, height: int) -> Coordinate:
    x, y = base
    for point in ((x, y + 1), (x + 1, y), (x + 1, y + 1), (x - 1, y)):
        if 0 <= point[0] < width and 0 <= point[1] < height:
            return point
    return base


def _add_manhattan_road(
    roads: set[Coordinate],
    start: Coordinate,
    end: Coordinate,
    rng: random.Random,
) -> None:
    if rng.random() < 0.5:
        corner = (end[0], start[1])
    else:
        corner = (start[0], end[1])
    _add_axis_road(roads, start, corner)
    _add_axis_road(roads, corner, end)


def _add_axis_road(roads: set[Coordinate], start: Coordinate, end: Coordinate) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            roads.add((x1, y))
        return
    if y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            roads.add((x, y1))
        return
    _add_axis_road(roads, start, (x2, y1))
    _add_axis_road(roads, (x2, y1), end)


def _random_water_cells(
    rng: random.Random,
    width: int,
    height: int,
    avoid: set[Coordinate],
) -> set[Coordinate]:
    cells: set[Coordinate] = set()
    if rng.random() < 0.72:
        x = rng.choice([0, 1, 21, 22])
        w = rng.randint(1, 2)
        y = rng.randint(10, 16)
        h = rng.randint(5, 9)
        cells.update(_rect_cells(x, y, w, h, width, height, avoid))
    if rng.random() < 0.45:
        x = rng.randint(1, 18)
        y = rng.randint(6, 18)
        cells.update(_rect_cells(x, y, rng.randint(2, 4), rng.randint(2, 4), width, height, avoid))
    return cells


def _random_park_cells(
    rng: random.Random,
    width: int,
    height: int,
    avoid: set[Coordinate],
) -> set[Coordinate]:
    cells: set[Coordinate] = set()
    for _ in range(rng.randint(1, 2)):
        x = rng.randint(2, 17)
        y = rng.randint(12, 20)
        cells.update(_rect_cells(x, y, rng.randint(3, 5), rng.randint(2, 4), width, height, avoid))
    return cells


def _random_building_cells(
    rng: random.Random,
    width: int,
    height: int,
    avoid: set[Coordinate],
    target_count: int,
) -> set[Coordinate]:
    cells: set[Coordinate] = set()
    attempts = 0
    while len(cells) < target_count and attempts < 220:
        attempts += 1
        x = rng.randint(3, width - 6)
        y = rng.randint(4, height - 6)
        w = rng.randint(2, 5)
        h = rng.randint(2, 4)
        rect = _rect_cells(x, y, w, h, width, height, avoid | cells)
        if len(rect) >= 4:
            cells.update(rect)
    return set(sorted(cells)[:target_count])


def _random_hazard_cells_near_targets(
    rng: random.Random,
    targets: dict[str, Coordinate],
    count: int,
    width: int,
    height: int,
    avoid: set[Coordinate],
) -> set[Coordinate]:
    cells: set[Coordinate] = set()
    target_order = ["C", "A", "B"]
    while len(cells) < count:
        target = targets[target_order[len(cells) % len(target_order)]]
        radius = rng.randint(1, 3)
        candidates = [
            (target[0] + dx, target[1] + dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if abs(dx) + abs(dy) <= radius + 1
        ]
        rng.shuffle(candidates)
        added = False
        for cell in candidates:
            if (
                0 <= cell[0] < width
                and 0 <= cell[1] < height
                and cell not in avoid
                and cell not in cells
            ):
                cells.add(cell)
                added = True
                break
        if not added:
            break
    return cells


def _collapse_candidates(
    width: int,
    height: int,
    roads: set[Coordinate],
    buildings: set[Coordinate],
) -> list[Coordinate]:
    candidates = set(roads)
    for x, y in buildings:
        for cell in _neighbors((x, y), width, height):
            candidates.add(cell)
    return sorted(candidates)


def _sample_layer_cells(
    rng: random.Random,
    candidates: list[Coordinate],
    count: int,
) -> set[Coordinate]:
    unique_candidates = sorted(set(candidates))
    if not unique_candidates:
        return set()
    sample_count = min(count, len(unique_candidates))
    return set(rng.sample(unique_candidates, sample_count))


def _rect_cells(
    x: int,
    y: int,
    w: int,
    h: int,
    width: int,
    height: int,
    avoid: set[Coordinate],
) -> set[Coordinate]:
    return {
        (cx, cy)
        for cx in range(x, min(x + w, width))
        for cy in range(y, min(y + h, height))
        if (cx, cy) not in avoid
    }


def _cells_to_json(cells: set[Coordinate]) -> list[list[int]]:
    return [[x, y] for x, y in sorted(cells)]


def generate_report(
    scenario: dict[str, Any],
    zone_scores: dict[str, dict[str, float]],
    assignments: dict[str, str],
    routes: dict[str, list[list[int]]],
    route_details: dict[str, dict[str, Any]] | None = None,
) -> str:
    if not zone_scores:
        return "请先加载灾情并执行智能推理。"

    ranked = sorted(
        zone_scores.items(), key=lambda item: item[1]["priority"], reverse=True
    )
    leader, leader_scores = ranked[0]
    lines = [
        "救援决策简报",
        "",
        "1. 当前判断",
        f"- 最高优先级：{leader}区（{leader_scores['priority']:.1f}）。",
        f"- 关键依据：被困概率 {leader_scores['trapped_probability']:.2f}，"
        f"生命风险 {leader_scores['life_risk']:.2f}，"
        f"道路可通行概率 {leader_scores['road_accessibility']:.2f}。",
        "",
        "2. 任务与路线",
    ]

    for unit_name, target in assignments.items():
        route = routes.get(unit_name, [])
        route_length = max(len(route) - 1, 0)
        route_cost = _route_cost_from_details_or_grid(
            scenario,
            unit_name,
            route,
            route_details,
        )
        if unit_name == "Drone-1":
            lines.append(
                f"- {unit_name}：无人机侦查；长度 {route_length} 格，代价 {route_cost:.1f}。"
            )
        else:
            lines.append(
                f"- {unit_name}：前往 {target}区救援；长度 {route_length} 格，代价 {route_cost:.1f}。"
            )

    lines.extend(["", "3. 重规划说明"])
    if scenario["map"].get("collapsed"):
        lines.append("- 塌方风险格已转为断路，系统已重新执行风险感知 A*。")
        lines.append("- 当前路线会优先避开断路、火灾和高拥堵区域。")
    else:
        lines.append("- 系统按当前地图完成概率推理、任务分配和路线规划。")
        lines.append("- 后续灾情变化或道路塌方会触发前后对比与动态重规划。")

    return "\n".join(lines)


def _route_cost_from_details_or_grid(
    scenario: dict[str, Any],
    unit_name: str,
    route: list[list[int]],
    route_details: dict[str, dict[str, Any]] | None,
) -> float:
    detail = (route_details or {}).get(unit_name, {})
    if isinstance(detail.get("total_cost"), (int, float)):
        return round(float(detail["total_cost"]), 2)
    return calculate_route_cost(scenario, unit_name, route)


def _a_star(
    scenario: dict[str, Any],
    start: Coordinate,
    goal: Coordinate,
    unit_type: str,
) -> list[Coordinate]:
    width = scenario["map"]["width"]
    height = scenario["map"]["height"]
    blocked = {_as_coordinate(cell) for cell in scenario["map"].get("blocked", [])}
    buildings = {_as_coordinate(cell) for cell in scenario["map"].get("buildings", [])}
    water = {_as_coordinate(cell) for cell in scenario["map"].get("water", [])}
    roads = {_as_coordinate(cell) for cell in scenario["map"].get("roads", [])}
    blocked.update(buildings)
    blocked.update(water)
    fire = {_as_coordinate(cell) for cell in scenario["map"].get("fire", [])}
    congestion = {_as_coordinate(cell) for cell in scenario["map"].get("congestion", [])}
    collapse = {_as_coordinate(cell) for cell in scenario["map"].get("collapse_cells", [])}

    frontier: list[tuple[float, Coordinate]] = [(0.0, start)]
    came_from: dict[Coordinate, Coordinate | None] = {start: None}
    cost_so_far: dict[Coordinate, float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt in _neighbors(current, width, height):
            if unit_type == "car" and nxt in blocked:
                continue
            move_cost = _cell_cost(
                nxt,
                fire,
                congestion,
                blocked,
                collapse,
                roads,
                unit_type,
            )
            new_cost = cost_so_far[current] + move_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return [start]

    path = []
    current: Coordinate | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def _cell_cost(
    cell: Coordinate,
    fire: set[Coordinate],
    congestion: set[Coordinate],
    blocked: set[Coordinate],
    collapse: set[Coordinate],
    roads: set[Coordinate],
    unit_type: str,
) -> float:
    cost = 1.0
    if roads and unit_type == "car" and cell not in roads:
        cost += 0.8
    if cell in congestion:
        cost += 3.5 if unit_type == "car" else 0.8
    if cell in fire:
        cost += 5.0 if unit_type == "car" else 1.5
    if cell in collapse and cell not in blocked:
        cost += 4.0 if unit_type == "car" else 0.8
    if unit_type == "drone" and cell in blocked:
        cost += 0.3
    return cost


def _neighbors(cell: Coordinate, width: int, height: int) -> list[Coordinate]:
    x, y = cell
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [
        (nx, ny)
        for nx, ny in candidates
        if 0 <= nx < width and 0 <= ny < height
    ]


def _manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _as_coordinate(value: list[int] | tuple[int, int]) -> Coordinate:
    return int(value[0]), int(value[1])


def _valid_cells(value: Any, width: int, height: int) -> list[Coordinate]:
    if not isinstance(value, list):
        return []

    cells: list[Coordinate] = []
    seen: set[Coordinate] = set()
    for item in value:
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(part, (int, float)) for part in item)
        ):
            x, y = int(item[0]), int(item[1])
            if 0 <= x < width and 0 <= y < height and (x, y) not in seen:
                seen.add((x, y))
                cells.append((x, y))
    return cells


def _add_cells(map_data: dict[str, Any], field: str, cells: list[Coordinate]) -> None:
    existing = {_as_coordinate(cell) for cell in map_data.get(field, [])}
    existing.update(cells)
    map_data[field] = [[x, y] for x, y in sorted(existing)]


def _remove_cells(map_data: dict[str, Any], field: str, cells: list[Coordinate]) -> None:
    remove_set = set(cells)
    existing = {_as_coordinate(cell) for cell in map_data.get(field, [])}
    remaining = existing - remove_set
    map_data[field] = [[x, y] for x, y in sorted(remaining)]
