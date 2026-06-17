from copy import deepcopy

import pytest

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.inference import assess_zones
from emergency_commander.routing import NoRouteError, risk_aware_astar


RISK_WEIGHTS = {
    "fire": 0.35,
    "damage": 0.25,
    "congestion": 0.20,
    "secondary": 0.20,
}


def road(road_id, start, end, travel_time, *, fire=0.0, damage=0.0, status="open"):
    return {
        "road_id": road_id,
        "from": start,
        "to": end,
        "distance": travel_time,
        "travel_time_base": travel_time,
        "status": status,
        "bidirectional": True,
        "risk": {
            "fire": fire,
            "damage": damage,
            "congestion": 0.0,
            "secondary_disaster": 0.0,
        },
    }


def test_risk_aware_astar_chooses_safer_detour():
    roads = [
        road("direct", "HQ", "ZONE_A", 2.0, fire=1.0),
        road("detour_1", "HQ", "X", 1.2),
        road("detour_2", "X", "ZONE_A", 1.2),
    ]

    route = risk_aware_astar(
        roads,
        start="HQ",
        goal="ZONE_A",
        speed=1.0,
        risk_weights=RISK_WEIGHTS,
    )

    assert route["path"] == ["HQ", "X", "ZONE_A"]
    assert route["eta"] == pytest.approx(2.4)
    assert route["path_risk"] == pytest.approx(0.0)


def test_risk_aware_astar_rejects_blocked_only_route():
    roads = [road("collapsed", "HQ", "ZONE_A", 2.0, status="blocked")]

    with pytest.raises(NoRouteError, match="HQ.*ZONE_A"):
        risk_aware_astar(
            roads,
            start="HQ",
            goal="ZONE_A",
            speed=1.0,
            risk_weights=RISK_WEIGHTS,
        )


def test_rescue_route_rejects_road_above_unit_fire_limit():
    roads = [road("burning", "HQ", "ZONE_A", 2.0, fire=0.9)]

    with pytest.raises(NoRouteError, match="HQ.*ZONE_A"):
        risk_aware_astar(
            roads,
            start="HQ",
            goal="ZONE_A",
            speed=1.0,
            risk_weights=RISK_WEIGHTS,
            max_fire_risk=0.7,
        )


def test_utility_matrix_uses_high_fire_route_when_safe_route_is_blocked():
    scenario = allocation_scenario()
    scenario["zones"] = [
        {
            "zone_id": "A",
            "node_id": "ZONE_A",
            "observations": {
                "sos_signal": 0.90,
                "building_collapse": 0.80,
                "smoke": 0.20,
                "fire": 0.20,
                "road_damage": 0.10,
                "human_activity": 0.60,
                "congestion": 0.10,
                "time_urgency": 0.80,
                "drone_confidence": 0.0,
            },
        }
    ]
    scenario["nodes"] = {
        "HQ": {"x": 0.0, "y": 0.0},
        "SAFE": {"x": 1.0, "y": 1.0},
        "ZONE_A": {"x": 2.0, "y": 0.0},
    }
    scenario["roads"] = [
        road("safe_detour_1", "HQ", "SAFE", 1.0, fire=0.1, status="blocked"),
        road("safe_detour_2", "SAFE", "ZONE_A", 1.0, fire=0.1),
        road("burning_direct", "HQ", "ZONE_A", 1.4, fire=0.95),
    ]
    scenario["air_routes"] = []
    scenario["units"] = [
        {
            "unit_id": "RescueCar-1",
            "type": "rescue_car",
            "start_node": "HQ",
            "speed": 1.0,
            "can_transport": True,
            "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
        }
    ]
    assessments = [
        {
            "zone_id": "A",
            "node_id": "ZONE_A",
            "trapped_prob": 0.8,
            "passability_prob": 0.9,
            "life_risk": 0.6,
            "priority_score": 0.8,
        }
    ]

    matrix = build_utility_matrix(scenario, assessments)

    candidate = matrix[0]
    assert candidate["feasible"]
    assert candidate["route"]["road_ids"] == ["burning_direct"]
    assert candidate["route"]["risk_policy"] == "relaxed_fire_limit"
    assert candidate["reason"] == "feasible_with_risk_override"


def allocation_scenario():
    observations = {
        "A": (0.95, 0.85, 0.55, 0.50, 0.20, 0.80),
        "B": (0.30, 0.20, 0.20, 0.10, 0.10, 0.35),
        "C": (0.80, 0.70, 0.90, 0.85, 0.30, 0.90),
    }
    zones = []
    for zone_id, (sos, collapse, fire, damage, congestion, urgency) in observations.items():
        zones.append(
            {
                "zone_id": zone_id,
                "node_id": f"ZONE_{zone_id}",
                "observations": {
                    "sos_signal": sos,
                    "building_collapse": collapse,
                    "smoke": fire,
                    "fire": fire,
                    "road_damage": damage,
                    "human_activity": sos * 0.7,
                    "congestion": congestion,
                    "time_urgency": urgency,
                    "drone_confidence": 0.0,
                },
            }
        )

    return {
        "nodes": {
            "HQ": {"x": 0.0, "y": 0.0},
            "ZONE_A": {"x": 5.0, "y": 0.0},
            "ZONE_B": {"x": 0.0, "y": 4.0},
            "ZONE_C": {"x": 4.0, "y": 4.0},
        },
        "config": {
            "weights": {
                "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
                "passability": {"road_damage": 0.45, "fire_risk": 0.35, "congestion": 0.20, "drone_confidence": 0.0},
                "life_risk": {"fire": 0.40, "trapped_prob": 0.35, "time_urgency": 0.25},
                "priority": {"trapped_prob": 0.40, "life_risk": 0.30, "time_urgency": 0.20, "accessibility": 0.10},
                "utility": {"alpha": 0.30, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "epsilon": 0.10},
                "astar_risk": RISK_WEIGHTS,
            },
            "thresholds": {"car_min_passability": 0.45, "drone_recon_priority_risk": 0.70},
        },
        "zones": zones,
        "roads": [
            road("to_a", "HQ", "ZONE_A", 5.0, fire=0.2),
            road("to_b", "HQ", "ZONE_B", 4.0),
            road("to_c", "HQ", "ZONE_C", 3.0, fire=0.8, damage=0.6),
        ],
        "air_routes": [
            road("air_to_a", "HQ", "ZONE_A", 2.5),
            road("air_to_b", "HQ", "ZONE_B", 2.0),
            road("air_to_c", "HQ", "ZONE_C", 2.2),
        ],
        "units": [
            {
                "unit_id": "RescueCar-1",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 1.2,
                "can_transport": True,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
            {
                "unit_id": "RescueCar-2",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 1.0,
                "can_transport": True,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
            {
                "unit_id": "Drone-1",
                "type": "drone",
                "start_node": "HQ",
                "speed": 2.0,
                "can_transport": False,
                "constraints": {},
            },
        ],
    }


def test_allocator_sends_drone_to_high_risk_low_accessibility_zone():
    scenario = allocation_scenario()
    assessments = assess_zones(scenario)

    matrix = build_utility_matrix(scenario, assessments)
    assignments = allocate_tasks(scenario, matrix)

    by_unit = {item["unit_id"]: item for item in assignments}
    assert by_unit["Drone-1"]["target_zone"] == "C"
    assert by_unit["Drone-1"]["mission_type"] == "reconnaissance"
    assert {by_unit["RescueCar-1"]["target_zone"], by_unit["RescueCar-2"]["target_zone"]} == {"A", "B"}
    assert all(item["target_zone"] != "C" for item in assignments if item["mission_type"] == "rescue")


def test_drone_can_fly_directly_without_predefined_air_route():
    scenario = allocation_scenario()
    scenario["air_routes"] = []
    scenario["units"] = [
        {
            "unit_id": "Drone-1",
            "type": "drone",
            "start_node": "HQ",
            "speed": 2.0,
            "can_transport": False,
            "constraints": {},
        }
    ]
    assessments = assess_zones(scenario)

    matrix = build_utility_matrix(scenario, assessments)

    candidate = next(
        item
        for item in matrix
        if item["unit_id"] == "Drone-1" and item["target_zone"] == "C"
    )
    assert candidate["feasible"]
    assert candidate["reason"] == "direct_air_route"
    assert candidate["route"]["route_layer"] == "air"
    assert candidate["route"]["path"] == ["HQ", "ZONE_C"]
    assert candidate["route"]["road_ids"] == []


def test_allocator_exposes_ranked_enumeration_trace_when_requested():
    scenario = allocation_scenario()
    matrix = build_utility_matrix(scenario, assess_zones(scenario))

    result = allocate_tasks(scenario, matrix, include_trace=True)

    assignments = result["assignments"]
    trace = result["trace"]
    assert trace["considered"] > 0
    assert trace["duplicate_zone_rejections"] > 0
    assert trace["ranked_combinations"]
    assert trace["winning_total"] == pytest.approx(
        sum(item["expected_utility"] for item in assignments)
    )
    assert trace["ranked_combinations"][0]["total"] == trace["winning_total"]


def test_utility_candidate_exposes_signed_breakdown_and_explanation():
    scenario = allocation_scenario()
    scenario["config"]["weights"]["utility"]["zeta"] = 0.10
    scenario["units"][0]["resource_cost"] = 0.55
    assessments = assess_zones(scenario)

    matrix = build_utility_matrix(scenario, assessments)
    candidate = next(
        item
        for item in matrix
        if item["unit_id"] == "RescueCar-1" and item["target_zone"] == "A"
    )

    assert candidate["resource_cost"] == pytest.approx(0.55)
    assert set(candidate["utility_inputs"]) == {
        "trapped_prob",
        "life_risk",
        "accessibility",
        "arrival_time_normalized",
        "path_risk",
        "resource_cost",
    }
    assert set(candidate["utility_breakdown"]) == {
        "trapped_benefit",
        "life_risk_benefit",
        "accessibility_benefit",
        "arrival_time_cost",
        "path_risk_cost",
        "resource_cost",
    }
    assert sum(candidate["utility_breakdown"].values()) == pytest.approx(
        candidate["expected_utility"], abs=1e-6
    )
    assert "RescueCar-1" in candidate["explanation"]
    assert "A区" in candidate["explanation"]
    assert "资源" in candidate["explanation"]


def test_resource_cost_penalty_lowers_expected_utility_by_zeta_weight():
    scenario = allocation_scenario()
    scenario["config"]["weights"]["utility"]["zeta"] = 0.20
    low_cost = deepcopy(scenario["units"][0])
    low_cost["unit_id"] = "LowCostCar"
    low_cost["resource_cost"] = 0.20
    high_cost = deepcopy(low_cost)
    high_cost["unit_id"] = "HighCostCar"
    high_cost["resource_cost"] = 0.80
    scenario["units"] = [low_cost, high_cost]
    assessments = assess_zones(scenario)

    matrix = build_utility_matrix(scenario, assessments)
    candidates = {
        item["unit_id"]: item
        for item in matrix
        if item["target_zone"] == "A" and item["feasible"]
    }

    assert (
        candidates["LowCostCar"]["expected_utility"]
        - candidates["HighCostCar"]["expected_utility"]
    ) == pytest.approx(0.20 * (0.80 - 0.20), abs=1e-6)
