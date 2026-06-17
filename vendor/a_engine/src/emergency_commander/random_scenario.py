from __future__ import annotations

import random
from math import hypot
from typing import Any

from emergency_commander.contracts import validate_scenario


ZONE_IDS = ("A", "B", "C", "D", "E", "F")
GRID_ROWS = 3
GRID_COLUMNS = 6


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
    anchors = (
        (2.0, -3.0),
        (18.0, -3.0),
        (-2.5, 6.0),
        (22.5, 6.0),
        (2.0, 11.0),
        (18.0, 11.0),
    )
    for zone_id, (base_x, base_y) in zip(ZONE_IDS, anchors):
        node_id = f"ZONE_{zone_id}"
        x = round(base_x + rng.uniform(-0.45, 0.45), 3)
        y = round(base_y + rng.uniform(-0.45, 0.45), 3)
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
    return {
        "road_id": road_id,
        "from": start,
        "to": end,
        "distance": distance,
        "travel_time_base": round(max(0.4, distance * pace), 3),
        "status": "open",
        "bidirectional": True,
        "risk": risk,
    }


def _generate_ground_roads(
    nodes: dict[str, dict[str, float]], rng: random.Random
) -> list[dict[str, Any]]:
    roads = [
        _road("R_HOSPITAL_HQ", "HOSPITAL", "HQ", nodes, rng, risk_profile="safe"),
        _road("R_HQ_J00", "HQ", "J00", nodes, rng, risk_profile="safe"),
        _road("R_HQ_J10", "HQ", "J10", nodes, rng, risk_profile="safe"),
    ]

    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS - 1):
            profile = "hazardous" if row == 1 and 1 <= column <= 3 else "safe"
            roads.append(
                _road(
                    f"R_J{row}{column}_J{row}{column + 1}",
                    f"J{row}{column}",
                    f"J{row}{column + 1}",
                    nodes,
                    rng,
                    risk_profile=profile,
                )
            )

    omitted_verticals: set[tuple[int, int]] = set()
    for row in range(GRID_ROWS - 1):
        omitted_verticals.update(
            (row, column)
            for column in rng.sample(range(1, GRID_COLUMNS - 1), k=2)
        )
    for row in range(GRID_ROWS - 1):
        for column in range(GRID_COLUMNS):
            if (row, column) in omitted_verticals:
                continue
            profile = "hazardous" if column in {2, 3} else "safe"
            roads.append(
                _road(
                    f"R_J{row}{column}_J{row + 1}{column}",
                    f"J{row}{column}",
                    f"J{row + 1}{column}",
                    nodes,
                    rng,
                    risk_profile=profile,
                )
            )

    diagonal_candidates = [
        ("J00", "J11"),
        ("J01", "J10"),
        ("J04", "J15"),
        ("J05", "J14"),
        ("J10", "J21"),
        ("J11", "J20"),
        ("J14", "J25"),
        ("J15", "J24"),
    ]
    for start, end in rng.sample(diagonal_candidates, k=2):
        roads.append(
            _road(
                f"R_{start}_{end}_BYPASS",
                start,
                end,
                nodes,
                rng,
                risk_profile="safe",
            )
        )

    zone_connections = {
        "A": ("J00", "J01"),
        "B": ("J04", "J05"),
        "C": ("J10", "J20"),
        "D": ("J15", "J25"),
        "E": ("J20", "J21"),
        "F": ("J24", "J25"),
    }
    for zone_id, junctions in zone_connections.items():
        for index, junction in enumerate(junctions, start=1):
            roads.append(
                _road(
                    f"R_{junction}_ZONE_{zone_id}_{index}",
                    junction,
                    f"ZONE_{zone_id}",
                    nodes,
                    rng,
                    risk_profile="mixed",
                )
            )
    return roads


def _generate_air_routes(
    nodes: dict[str, dict[str, float]], rng: random.Random
) -> list[dict[str, Any]]:
    routes = [_road("AIR_HQ_RELAY", "HQ", "AIR_RELAY", nodes, rng, air=True)]
    for zone_id in ZONE_IDS:
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
    for zone_id in rng.sample(ZONE_IDS, k=2):
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
