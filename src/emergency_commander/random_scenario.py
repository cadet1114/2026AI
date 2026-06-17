from __future__ import annotations

from collections import deque
import random
from math import hypot
from typing import Any

from emergency_commander.contracts import validate_scenario


ZONE_ID_POOL = ("A", "B", "C", "D", "E", "F", "G")
ZONE_COUNT_RANGE = (4, 7)
GRID_ROWS = 3
GRID_COLUMNS = 6
MAP_X_RANGE = (-2.5, 22.5)
MAP_Y_RANGE = (-3.5, 12.0)
MIN_ZONE_DISTANCE = 5.0
ROAD_PRUNE_RANGE = (4, 8)
MIN_BYPASS_ROADS_AFTER_PRUNE = 2


def _round_probability(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _distance(nodes: dict[str, dict[str, float]], start: str, end: str) -> float:
    value = hypot(
        nodes[start]["x"] - nodes[end]["x"],
        nodes[start]["y"] - nodes[end]["y"],
    )
    return round(max(value, 0.5), 3)


def _default_config() -> dict[str, Any]:
    return {
        "weights": {
            "trapped": {
                "sos": 0.35,
                "collapse": 0.30,
                "human_activity": 0.20,
                "smoke": 0.15,
            },
            "passability": {
                "road_damage": 0.45,
                "fire_risk": 0.35,
                "congestion": 0.20,
                "drone_confidence": 0.0,
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
                "alpha": 0.30,
                "beta": 0.25,
                "gamma": 0.20,
                "delta": 0.15,
                "epsilon": 0.10,
                "zeta": 0.10,
            },
            "astar_risk": {
                "fire": 0.35,
                "damage": 0.25,
                "congestion": 0.20,
                "secondary": 0.20,
            },
        },
        "thresholds": {
            "car_min_passability": 0.45,
            "drone_recon_priority_risk": 0.70,
        },
    }


def _generate_nodes_and_zones(
    rng: random.Random,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, float]] = {
        "HQ": {"x": -4.0, "y": 4.0},
        "HOSPITAL": {"x": -8.0, "y": 4.0},
        "AIR_RELAY": {"x": 10.0, "y": 15.0},
    }
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            nodes[f"J{row}{column}"] = {
                "x": round(column * 4.0 + rng.uniform(-0.32, 0.32), 3),
                "y": round(row * 4.0 + rng.uniform(-0.32, 0.32), 3),
            }

    zones: list[dict[str, Any]] = []
    zone_count = rng.randint(*ZONE_COUNT_RANGE)
    placed: list[tuple[float, float]] = []
    for zone_id in ZONE_ID_POOL[:zone_count]:
        node_id = f"ZONE_{zone_id}"
        for _ in range(500):
            x = round(rng.uniform(*MAP_X_RANGE), 3)
            y = round(rng.uniform(*MAP_Y_RANGE), 3)
            if all(hypot(x - px, y - py) >= MIN_ZONE_DISTANCE for px, py in placed):
                break
        else:
            raise RuntimeError("could not place non-overlapping random disaster zones")
        placed.append((x, y))
        nodes[node_id] = {"x": x, "y": y}
        hazard = rng.uniform(0.42, 0.94)
        observations = {
            "hazard_intensity": _round_probability(hazard),
            "sos_signal": _round_probability(rng.uniform(0.48, 0.98)),
            "building_collapse": _round_probability(rng.uniform(0.35, 0.90)),
            "smoke": _round_probability(rng.uniform(0.25, 0.88)),
            "fire": _round_probability(rng.uniform(0.15, 0.62)),
            "road_damage": _round_probability(rng.uniform(0.08, 0.48)),
            "human_activity": _round_probability(rng.uniform(0.35, 0.92)),
            "congestion": _round_probability(rng.uniform(0.05, 0.42)),
            "time_urgency": _round_probability(rng.uniform(0.48, 0.97)),
            "drone_confidence": 0.0,
        }
        zones.append(
            {
                "zone_id": zone_id,
                "node_id": node_id,
                "observations": observations,
            }
        )
    return nodes, zones


def _risk(rng: random.Random, *, air: bool = False) -> dict[str, float]:
    return {
        "fire": _round_probability(rng.uniform(0.02, 0.24 if air else 0.38)),
        "damage": _round_probability(rng.uniform(0.0, 0.04 if air else 0.32)),
        "congestion": _round_probability(rng.uniform(0.0, 0.03 if air else 0.28)),
        "secondary_disaster": _round_probability(rng.uniform(0.01, 0.18)),
    }


def _road(
    road_id: str,
    start: str,
    end: str,
    nodes: dict[str, dict[str, float]],
    rng: random.Random,
    *,
    air: bool = False,
    risk_profile: str = "mixed",
    status: str = "open",
) -> dict[str, Any]:
    distance = _distance(nodes, start, end)
    pace = rng.uniform(0.72, 1.15) if not air else rng.uniform(0.28, 0.46)
    risk = _risk(rng, air=air)
    if not air and risk_profile == "safe":
        risk = {
            "fire": _round_probability(rng.uniform(0.02, 0.13)),
            "damage": _round_probability(rng.uniform(0.01, 0.12)),
            "congestion": _round_probability(rng.uniform(0.01, 0.14)),
            "secondary_disaster": _round_probability(rng.uniform(0.01, 0.10)),
        }
    elif not air and risk_profile == "hazardous":
        risk = {
            "fire": _round_probability(rng.uniform(0.48, 0.68)),
            "damage": _round_probability(rng.uniform(0.28, 0.52)),
            "congestion": _round_probability(rng.uniform(0.45, 0.72)),
            "secondary_disaster": _round_probability(rng.uniform(0.22, 0.48)),
        }
    elif not air and risk_profile == "blocked":
        risk = {
            "fire": _round_probability(rng.uniform(0.20, 0.55)),
            "damage": _round_probability(rng.uniform(0.76, 0.96)),
            "congestion": _round_probability(rng.uniform(0.55, 0.90)),
            "secondary_disaster": _round_probability(rng.uniform(0.42, 0.74)),
        }
    return {
        "road_id": road_id,
        "from": start,
        "to": end,
        "distance": distance,
        "travel_time_base": round(max(0.4, distance * pace), 3),
        "status": status,
        "bidirectional": True,
        "risk": risk,
    }


def _open_graph_reaches_zones(
    roads: list[dict[str, Any]], zone_node_ids: set[str]
) -> bool:
    graph: dict[str, set[str]] = {}
    for road in roads:
        if road["status"] != "open":
            continue
        start = road["from"]
        end = road["to"]
        graph.setdefault(start, set()).add(end)
        if road.get("bidirectional", True):
            graph.setdefault(end, set()).add(start)

    queue: deque[str] = deque(["HQ"])
    visited = {"HQ"}
    while queue:
        node_id = queue.popleft()
        for neighbor in graph.get(node_id, set()) - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return zone_node_ids.issubset(visited)


def _single_failure_tolerant(
    roads: list[dict[str, Any]], zone_node_ids: set[str]
) -> bool:
    """Keep generated maps demonstrable even after one additional road failure."""
    open_roads = [road for road in roads if road["status"] == "open"]
    for candidate in open_roads:
        remaining = [road for road in roads if road is not candidate]
        if not _open_graph_reaches_zones(remaining, zone_node_ids):
            return False
    return True


def _can_prune_road(road: dict[str, Any]) -> bool:
    if road["status"] != "open":
        return False
    if road["from"] in {"HQ", "HOSPITAL"} or road["to"] in {"HQ", "HOSPITAL"}:
        return False
    if road["from"].startswith("ZONE_") or road["to"].startswith("ZONE_"):
        return False
    return road["from"].startswith("J") and road["to"].startswith("J")


def _prune_random_roads(
    roads: list[dict[str, Any]],
    nodes: dict[str, dict[str, float]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Remove non-critical road segments so random maps do not look grid-stamped."""
    zone_node_ids = {node_id for node_id in nodes if node_id.startswith("ZONE_")}
    target_prunes = rng.randint(*ROAD_PRUNE_RANGE)
    candidates = [road for road in roads if _can_prune_road(road)]
    rng.shuffle(candidates)
    pruned = 0

    for road in candidates:
        if pruned >= target_prunes:
            break
        if "BYPASS" in road["road_id"]:
            bypass_count = sum(
                candidate["status"] == "open" and "BYPASS" in candidate["road_id"]
                for candidate in roads
            )
            if bypass_count <= MIN_BYPASS_ROADS_AFTER_PRUNE:
                continue

        trial = [candidate for candidate in roads if candidate is not road]
        if not _open_graph_reaches_zones(trial, zone_node_ids):
            continue
        if not _single_failure_tolerant(trial, zone_node_ids):
            continue
        roads = trial
        pruned += 1

    return roads


def _generate_ground_roads(
    nodes: dict[str, dict[str, float]], rng: random.Random
) -> list[dict[str, Any]]:
    roads: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_road(
        start: str,
        end: str,
        *,
        road_id: str | None = None,
        risk_profile: str = "mixed",
        status: str = "open",
    ) -> None:
        key = tuple(sorted((start, end)))
        if key in seen and status != "blocked":
            return
        if status != "blocked":
            seen.add(key)
        roads.append(
            _road(
                road_id or f"R_{start}_{end}",
                start,
                end,
                nodes,
                rng,
                risk_profile=risk_profile,
                status=status,
            )
        )

    add_road("HOSPITAL", "HQ", road_id="R_HOSPITAL_HQ", risk_profile="safe")
    for entry in rng.sample(("J00", "J10", "J20"), k=2):
        add_road("HQ", entry, road_id=f"R_HQ_{entry}", risk_profile="safe")

    # Two horizontal trunk roads keep the map readable without filling every cell
    # with square blocks.
    for row in (0, 2):
        for column in range(GRID_COLUMNS - 1):
            add_road(
                f"J{row}{column}",
                f"J{row}{column + 1}",
                risk_profile="safe" if column not in {2, 3} else "mixed",
            )

    # Middle junctions connect the trunks but do not form a full rectangular grid.
    for column in range(GRID_COLUMNS):
        add_road(
            f"J0{column}",
            f"J1{column}",
            risk_profile="hazardous" if column in {2, 3} else "mixed",
        )
        add_road(
            f"J1{column}",
            f"J2{column}",
            risk_profile="hazardous" if column in {2, 3} else "mixed",
        )

    middle_segments = [(f"J1{column}", f"J1{column + 1}") for column in range(GRID_COLUMNS - 1)]
    for start, end in rng.sample(middle_segments, k=rng.randint(1, 3)):
        add_road(start, end, risk_profile="hazardous")

    diagonal_candidates = [
        ("J00", "J11"),
        ("J01", "J10"),
        ("J01", "J12"),
        ("J04", "J15"),
        ("J05", "J14"),
        ("J10", "J21"),
        ("J11", "J20"),
        ("J12", "J21"),
        ("J13", "J24"),
        ("J14", "J25"),
        ("J15", "J24"),
    ]
    for start, end in rng.sample(diagonal_candidates, k=rng.randint(3, 5)):
        add_road(
            start,
            end,
            road_id=f"R_{start}_{end}_BYPASS",
            risk_profile="safe",
        )

    blocked_candidates = [
        ("J11", "J12"),
        ("J12", "J13"),
        ("J13", "J14"),
        ("J02", "J12"),
        ("J13", "J23"),
    ]
    for start, end in rng.sample(blocked_candidates, k=rng.randint(1, 2)):
        add_road(
            start,
            end,
            road_id=f"R_{start}_{end}_BLOCKED",
            risk_profile="blocked",
            status="blocked",
        )

    junction_ids = [
        f"J{row}{column}"
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    ]
    zone_ids = sorted(
        node_id.removeprefix("ZONE_")
        for node_id in nodes
        if node_id.startswith("ZONE_")
    )
    for zone_id in zone_ids:
        zone_node = f"ZONE_{zone_id}"
        nearest = sorted(
            junction_ids,
            key=lambda junction_id: _distance(nodes, zone_node, junction_id),
        )
        for index, junction in enumerate(nearest[:2], start=1):
            add_road(
                junction,
                zone_node,
                road_id=f"R_{junction}_ZONE_{zone_id}_{index}",
                risk_profile="mixed",
            )
    return _prune_random_roads(roads, nodes, rng)


def _generate_air_routes(
    nodes: dict[str, dict[str, float]], rng: random.Random
) -> list[dict[str, Any]]:
    routes = [_road("AIR_HQ_RELAY", "HQ", "AIR_RELAY", nodes, rng, air=True)]
    zone_ids = sorted(
        node_id.removeprefix("ZONE_")
        for node_id in nodes
        if node_id.startswith("ZONE_")
    )
    for zone_id in zone_ids:
        routes.append(
            _road(
                f"AIR_RELAY_{zone_id}",
                "AIR_RELAY",
                f"ZONE_{zone_id}",
                nodes,
                rng,
                air=True,
            )
        )
    direct_count = rng.randint(1, min(3, len(zone_ids)))
    for zone_id in rng.sample(zone_ids, k=direct_count):
        routes.append(
            _road(
                f"AIR_HQ_ZONE_{zone_id}_DIRECT",
                "HQ",
                f"ZONE_{zone_id}",
                nodes,
                rng,
                air=True,
            )
        )
    return routes


def _generate_units(rng: random.Random) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "RescueCar-1",
            "type": "rescue_car",
            "start_node": "HQ",
            "speed": round(rng.uniform(1.0, 1.3), 3),
            "can_transport": True,
            "capacity": 4,
            "service_time": 1.0,
            "resource_cost": 0.50,
            "constraints": {"max_fire_risk": 0.72, "min_passability": 0.45},
        },
        {
            "unit_id": "RescueCar-2",
            "type": "rescue_car",
            "start_node": "HQ",
            "speed": round(rng.uniform(0.95, 1.2), 3),
            "can_transport": True,
            "capacity": 6,
            "service_time": 1.2,
            "resource_cost": 0.60,
            "constraints": {"max_fire_risk": 0.72, "min_passability": 0.45},
        },
        {
            "unit_id": "RescueCar-3",
            "type": "rescue_car",
            "start_node": "HQ",
            "speed": round(rng.uniform(1.12, 1.42), 3),
            "can_transport": True,
            "capacity": 3,
            "service_time": 0.8,
            "resource_cost": 0.72,
            "constraints": {"max_fire_risk": 0.64, "min_passability": 0.50},
        },
        {
            "unit_id": "Drone-1",
            "type": "drone",
            "start_node": "HQ",
            "speed": round(rng.uniform(1.8, 2.3), 3),
            "can_transport": False,
            "capacity": 0,
            "service_time": 0.5,
            "resource_cost": 0.25,
            "constraints": {},
        },
        {
            "unit_id": "Drone-2",
            "type": "drone",
            "start_node": "HQ",
            "speed": round(rng.uniform(2.1, 2.6), 3),
            "can_transport": False,
            "capacity": 0,
            "service_time": 0.35,
            "resource_cost": 0.34,
            "constraints": {},
        },
    ]


def generate_random_scenario(seed: int, *, mode: str = "fixed") -> dict[str, Any]:
    """Generate a reproducible, contract-valid rescue scenario."""
    if mode not in {"fixed", "learned"}:
        raise ValueError("mode must be 'fixed' or 'learned'")
    seed = int(seed)
    rng = random.Random(seed)
    nodes, zones = _generate_nodes_and_zones(rng)
    scenario = {
        "scenario_id": f"random_{seed}",
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
