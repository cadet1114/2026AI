from emergency_commander.simulation import (
    advance_unit_states,
    initialize_unit_states,
    start_assignments,
)
from tests.test_routing_allocation import allocation_scenario


def stateful_scenario():
    scenario = allocation_scenario()
    scenario["hospital"] = {"node_id": "HOSPITAL"}
    scenario["nodes"]["HOSPITAL"] = {"x": -2.0, "y": 0.0}
    scenario["roads"].append(
        {
            "road_id": "hospital-hq",
            "from": "HOSPITAL",
            "to": "HQ",
            "distance": 2.0,
            "travel_time_base": 2.0,
            "status": "open",
            "bidirectional": True,
            "risk": {
                "fire": 0.0,
                "damage": 0.0,
                "congestion": 0.0,
                "secondary_disaster": 0.0,
            },
        }
    )
    for unit in scenario["units"]:
        if unit["type"] == "rescue_car":
            unit["capacity"] = 4
            unit["service_time"] = 1.0
        else:
            unit["capacity"] = 0
            unit["service_time"] = 0.5
    return scenario


def test_initialize_unit_states_tracks_capacity_location_and_idle_status():
    scenario = stateful_scenario()

    states = initialize_unit_states(scenario)

    car = states["RescueCar-1"]
    assert car["status"] == "idle"
    assert car["current_node"] == "HQ"
    assert car["position"] == {"x": 0.0, "y": 0.0}
    assert car["capacity"] == 4
    assert car["onboard"] == 0


def test_drone_completes_recon_and_becomes_idle_at_target():
    scenario = stateful_scenario()
    states = initialize_unit_states(scenario)
    assignment = {
        "unit_id": "Drone-1",
        "target_zone": "C",
        "mission_type": "reconnaissance",
        "expected_utility": 0.8,
        "estimated_people": 0,
        "route": {
            "path": ["HQ", "ZONE_C"],
            "road_ids": ["air_to_c"],
            "eta": 2.0,
            "path_risk": 0.1,
            "route_layer": "air",
        },
    }

    start_assignments(states, [assignment], scenario)
    advance_unit_states(states, 3.0, scenario)

    drone = states["Drone-1"]
    assert drone["status"] == "idle"
    assert drone["current_node"] == "ZONE_C"
    assert drone["current_task"] is None
    assert drone["completed_targets"] == ["C"]


def test_rescue_car_picks_up_to_capacity_and_returns_to_hospital():
    scenario = stateful_scenario()
    states = initialize_unit_states(scenario)
    assignment = {
        "unit_id": "RescueCar-1",
        "target_zone": "A",
        "mission_type": "rescue",
        "expected_utility": 0.9,
        "estimated_people": 7,
        "route": {
            "path": ["HQ", "ZONE_A"],
            "road_ids": ["to_a"],
            "eta": 4.0,
            "path_risk": 0.1,
            "route_layer": "ground",
        },
    }

    start_assignments(states, [assignment], scenario)
    advance_unit_states(states, 5.1, scenario)

    car = states["RescueCar-1"]
    assert car["status"] == "returning"
    assert car["onboard"] == 4
    assert car["current_task"]["target_node"] == "HOSPITAL"
    assert car["current_task"]["route"]["path"] == ["ZONE_A", "HQ", "HOSPITAL"]
    assert car["delivered_targets"] == []
    assert car["rescued_people"] == 0

    advance_unit_states(states, 10.0, scenario)

    assert car["status"] == "idle"
    assert car["current_node"] == "HOSPITAL"
    assert car["onboard"] == 0
    assert car["completed_missions"] == 1
    assert car["delivered_targets"] == ["A"]
    assert car["rescued_people"] == 4
    assert car["travel_minutes"] > 0


def test_partial_travel_updates_visual_position_without_falsifying_safe_node():
    scenario = stateful_scenario()
    states = initialize_unit_states(scenario)
    assignment = {
        "unit_id": "RescueCar-2",
        "target_zone": "B",
        "mission_type": "rescue",
        "expected_utility": 0.5,
        "estimated_people": 0,
        "route": {
            "path": ["HQ", "ZONE_B"],
            "road_ids": ["to_b"],
            "eta": 4.0,
            "path_risk": 0.0,
            "route_layer": "ground",
        },
    }

    start_assignments(states, [assignment], scenario)
    advance_unit_states(states, 2.0, scenario)

    car = states["RescueCar-2"]
    assert car["status"] == "en_route"
    assert car["current_node"] == "HQ"
    assert car["position"] == {"x": 0.0, "y": 2.0}
    assert car["remaining_travel"] == 2.0
