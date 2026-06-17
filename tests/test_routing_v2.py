import pytest

from emergency_commander.allocation import build_utility_matrix
from emergency_commander.inference import assess_zones
from emergency_commander.routing import risk_aware_astar
from tests.test_routing_allocation import allocation_scenario, road


RISK_WEIGHTS = {"fire": 0.35, "damage": 0.25, "congestion": 0.2, "secondary": 0.2}


def test_coordinate_astar_uses_nonzero_admissible_heuristic():
    nodes = {
        "HQ": {"x": 0.0, "y": 0.0},
        "A": {"x": 1.0, "y": 0.0},
        "B": {"x": 0.0, "y": 1.0},
        "GOAL": {"x": 2.0, "y": 0.0},
    }
    routes = [
        road("hq-a", "HQ", "A", 1.0),
        road("a-goal", "A", "GOAL", 1.0),
        road("hq-b", "HQ", "B", 2.0),
        road("b-goal", "B", "GOAL", 2.0),
    ]

    result = risk_aware_astar(
        routes,
        nodes=nodes,
        start="HQ",
        goal="GOAL",
        speed=1.0,
        risk_weights=RISK_WEIGHTS,
    )

    assert result["path"] == ["HQ", "A", "GOAL"]
    assert result["heuristic"] == "euclidean_time_lower_bound"
    assert result["heuristic_start"] > 0.0
    assert result["expanded_nodes"] <= 3


def test_coordinate_astar_exposes_expansion_trace_when_requested():
    nodes = {
        "HQ": {"x": 0.0, "y": 0.0},
        "A": {"x": 1.0, "y": 0.0},
        "GOAL": {"x": 2.0, "y": 0.0},
    }
    routes = [
        road("hq-a", "HQ", "A", 1.0),
        road("a-goal", "A", "GOAL", 1.0),
    ]

    result = risk_aware_astar(
        routes,
        nodes=nodes,
        start="HQ",
        goal="GOAL",
        speed=1.0,
        risk_weights=RISK_WEIGHTS,
        include_trace=True,
    )

    assert result["search_trace"][-1]["node"] == "GOAL"
    assert len(result["search_trace"]) == result["expanded_nodes"]
    assert all(
        row["f"] == pytest.approx(row["g"] + row["h"])
        for row in result["search_trace"]
    )
    assert result["search_trace"][0]["relaxations"][0]["neighbor"] == "A"


def test_drone_utility_flies_directly_and_never_uses_ground_roads():
    scenario = allocation_scenario()
    scenario["nodes"] = {
        "HQ": {"x": 0.0, "y": 0.0},
        "ZONE_A": {"x": 5.0, "y": 0.0},
        "ZONE_B": {"x": 0.0, "y": 4.0},
        "ZONE_C": {"x": 4.0, "y": 4.0},
        "AIR_RELAY": {"x": 2.0, "y": 2.0},
    }
    scenario["air_routes"] = [
        road("air-hq-relay", "HQ", "AIR_RELAY", 2.0),
        road("air-relay-a", "AIR_RELAY", "ZONE_A", 2.0),
        road("air-relay-b", "AIR_RELAY", "ZONE_B", 2.0),
        road("air-relay-c", "AIR_RELAY", "ZONE_C", 2.0),
    ]

    matrix = build_utility_matrix(scenario, assess_zones(scenario))
    drone_c = next(
        candidate
        for candidate in matrix
        if candidate["unit_id"] == "Drone-1" and candidate["target_zone"] == "C"
    )

    assert drone_c["route"]["route_layer"] == "air"
    assert drone_c["reason"] == "direct_air_route"
    assert drone_c["route"]["path"] == ["HQ", "ZONE_C"]
    assert drone_c["route"]["road_ids"] == []
    assert all(not route_id.startswith("to_") for route_id in drone_c["route"]["road_ids"])


def test_drone_remains_feasible_without_air_graph():
    scenario = allocation_scenario()
    scenario["nodes"] = {
        "HQ": {"x": 0.0, "y": 0.0},
        "ZONE_A": {"x": 5.0, "y": 0.0},
        "ZONE_B": {"x": 0.0, "y": 4.0},
        "ZONE_C": {"x": 4.0, "y": 4.0},
    }
    scenario["air_routes"] = []

    matrix = build_utility_matrix(scenario, assess_zones(scenario))
    drone_candidates = [item for item in matrix if item["unit_id"] == "Drone-1"]

    assert drone_candidates
    assert all(
        item["feasible"] and item["reason"] == "direct_air_route"
        for item in drone_candidates
    )
    assert all(item["route"]["road_ids"] == [] for item in drone_candidates)
