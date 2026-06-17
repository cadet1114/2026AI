import copy

import pytest

from emergency_commander.inference import assess_zones
from emergency_commander.input_adapter import ScenarioValidationError, normalize_scenario


def make_scenario():
    return {
        "scenario_id": "test_case",
        "generated_at": "2026-06-15T20:00:00+08:00",
        "mode": "fixed",
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HOSPITAL"},
        "config": {
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
        },
        "zones": [
            {
                "zone_id": "A",
                "node_id": "ZONE_A",
                "observations": {
                    "sos_signal": 0.90,
                    "building_collapse": 0.85,
                    "smoke": 0.75,
                    "fire": 0.55,
                    "road_damage": 0.60,
                    "human_activity": 0.60,
                    "congestion": 0.30,
                    "time_urgency": 0.80,
                    "drone_confidence": 0.00,
                },
            },
            {
                "zone_id": "B",
                "node_id": "ZONE_B",
                "observations": {
                    "sos_signal": 0.20,
                    "building_collapse": 0.25,
                    "smoke": 0.30,
                    "fire": 0.20,
                    "road_damage": 0.10,
                    "human_activity": 0.20,
                    "congestion": 0.10,
                    "time_urgency": 0.30,
                    "drone_confidence": 0.00,
                },
            },
        ],
        "roads": [],
        "units": [],
        "events": [],
    }


def test_normalize_scenario_adds_optional_defaults_without_mutating_input():
    raw = make_scenario()
    raw["zones"][0].pop("labels", None)
    original = copy.deepcopy(raw)

    normalized = normalize_scenario(raw)

    assert normalized["run_mode"] == "fixed"
    assert normalized["zones"][0]["labels"] == {}
    assert raw == original


def test_normalize_scenario_rejects_observation_outside_unit_interval():
    raw = make_scenario()
    raw["zones"][0]["observations"]["fire"] = 1.2

    with pytest.raises(ScenarioValidationError, match="fire"):
        normalize_scenario(raw)


def test_fixed_inference_ranks_high_sos_zone_first():
    scenario = normalize_scenario(make_scenario())

    assessments = assess_zones(scenario)

    by_id = {item["zone_id"]: item for item in assessments}
    assert by_id["A"]["trapped_prob"] > by_id["B"]["trapped_prob"]
    assert 0.0 < by_id["A"]["passability_prob"] < 1.0
    assert by_id["A"]["priority_score"] > by_id["B"]["priority_score"]
    assert all(0.0 <= item["priority_score"] <= 1.0 for item in assessments)
    assert by_id["A"]["inference_model"] == "expert_cpt"
    assert by_id["A"]["bayesian_evidence"]["sos_signal"] == "high"
    assert by_id["A"]["trapped_explanation"]["contributions"]


def test_normalize_scenario_rejects_invalid_road_risk():
    raw = make_scenario()
    raw["roads"] = [
        {
            "road_id": "R1",
            "from": "HQ",
            "to": "ZONE_A",
            "distance": 2.0,
            "travel_time_base": 2.0,
            "status": "open",
            "risk": {"fire": 1.2, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
        }
    ]

    with pytest.raises(ScenarioValidationError, match=r"roads\[0\].risk.fire"):
        normalize_scenario(raw)


def test_normalize_scenario_rejects_unsupported_event_type():
    raw = make_scenario()
    raw["events"] = [
        {
            "event_id": "EVT_BAD",
            "event_type": "meteor",
            "target_id": "A",
            "changes": {"observations.fire": 1.0},
        }
    ]

    with pytest.raises(ScenarioValidationError, match="event_type"):
        normalize_scenario(raw)
