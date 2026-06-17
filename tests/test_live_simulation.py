from copy import deepcopy
import json
from pathlib import Path

import pytest

from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.live_simulation import LiveSimulation
from emergency_commander.random_scenario import generate_random_scenario
from tests.test_pipeline_replanning import scenario_with_higher_utility_sos


ROOT = Path(__file__).resolve().parents[1]


def _base_weights():
    return {
        "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
        "passability": {"road_damage": 0.45, "fire_risk": 0.35, "congestion": 0.20, "drone_confidence": 0.0},
        "life_risk": {"fire": 0.40, "trapped_prob": 0.35, "time_urgency": 0.25},
        "priority": {"trapped_prob": 0.40, "life_risk": 0.30, "time_urgency": 0.20, "accessibility": 0.10},
        "utility": {"alpha": 0.30, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "epsilon": 0.10},
        "astar_risk": {"fire": 0.35, "damage": 0.25, "congestion": 0.20, "secondary": 0.20},
    }


def _staggered_return_scenario():
    def zone(zone_id, node_id, sos, urgency):
        return {
            "zone_id": zone_id,
            "node_id": node_id,
            "observations": {
                "sos_signal": sos,
                "building_collapse": 0.75,
                "smoke": 0.20,
                "fire": 0.20,
                "road_damage": 0.10,
                "human_activity": sos,
                "congestion": 0.10,
                "time_urgency": urgency,
                "drone_confidence": 0.0,
            },
        }

    def road(road_id, start, end, travel_time):
        return {
            "road_id": road_id,
            "from": start,
            "to": end,
            "distance": travel_time,
            "travel_time_base": travel_time,
            "status": "open",
            "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
        }

    return {
        "scenario_id": "staggered_return_case",
        "generated_at": "2026-06-16T12:00:00+08:00",
        "mode": "fixed",
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HQ"},
        "nodes": {
            "HQ": {"x": 0.0, "y": 0.0},
            "ZONE_A": {"x": 12.0, "y": 0.0},
            "ZONE_B": {"x": 1.0, "y": 0.0},
            "ZONE_C": {"x": 2.0, "y": 1.0},
        },
        "config": {
            "weights": _base_weights(),
            "thresholds": {"car_min_passability": 0.45, "drone_recon_priority_risk": 0.70},
        },
        "zones": [
            zone("A", "ZONE_A", 0.95, 0.95),
            zone("B", "ZONE_B", 0.90, 0.90),
            zone("C", "ZONE_C", 0.85, 0.85),
        ],
        "roads": [
            road("to_a", "HQ", "ZONE_A", 12.0),
            road("to_b", "HQ", "ZONE_B", 1.0),
            road("to_c", "HQ", "ZONE_C", 2.2),
        ],
        "air_routes": [],
        "units": [
            {
                "unit_id": "RescueCar-1",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 4.0,
                "can_transport": True,
                "capacity": 4,
                "service_time": 0.0,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
            {
                "unit_id": "RescueCar-2",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 0.5,
                "can_transport": True,
                "capacity": 4,
                "service_time": 0.0,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
        ],
        "events": [],
    }


def _drone_recon_scenario():
    return {
        "scenario_id": "drone_recon_case",
        "generated_at": "2026-06-16T13:00:00+08:00",
        "mode": "fixed",
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HQ"},
        "nodes": {
            "HQ": {"x": 0.0, "y": 0.0},
            "ZONE_A": {"x": 3.0, "y": 0.0},
        },
        "config": {
            "weights": _base_weights(),
            "thresholds": {"car_min_passability": 0.45, "drone_recon_priority_risk": 0.70},
        },
        "zones": [
            {
                "zone_id": "A",
                "node_id": "ZONE_A",
                "observations": {
                    "sos_signal": 0.70,
                    "building_collapse": 0.70,
                    "smoke": 0.70,
                    "fire": 0.85,
                    "road_damage": 0.90,
                    "human_activity": 0.70,
                    "congestion": 0.75,
                    "time_urgency": 0.85,
                    "drone_confidence": 0.0,
                },
            }
        ],
        "roads": [],
        "air_routes": [
            {
                "road_id": "air_a",
                "from": "HQ",
                "to": "ZONE_A",
                "distance": 3.0,
                "travel_time_base": 3.0,
                "status": "open",
                "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
            }
        ],
        "units": [
            {
                "unit_id": "Drone-1",
                "type": "drone",
                "start_node": "HQ",
                "speed": 3.0,
                "can_transport": False,
                "capacity": 0,
                "service_time": 0.5,
                "constraints": {},
            }
        ],
        "events": [],
    }


def _advance_to_phase(session, phase, limit=30):
    for _ in range(limit):
        if session.phase == phase:
            return
        session.step()
    raise AssertionError(f"session did not reach phase {phase}")


def test_live_session_starts_without_precomputing_and_advances_phases():
    session = LiveSimulation.create(generate_random_scenario(11), seed=11)

    assert session.phase == "validate"
    assert session.clock_minutes == 0
    assert session.timeline == []
    assert session.current_plan == {}
    assert session.calculation_history == []

    observed = []
    for _ in range(7):
        observed.append(session.phase)
        session.step()

    assert observed == [
        "validate",
        "infer",
        "prioritize",
        "route",
        "utility",
        "allocate",
        "execute",
    ]
    assert session.current_plan["assignments"]
    assert [entry["phase"] for entry in session.algorithm_log[:6]] == observed[:6]
    assert [entry["phase"] for entry in session.calculation_history] == observed
    assert set(session.calculation_history[0]) == {
        "index",
        "phase",
        "title",
        "clock_minutes",
        "summary",
        "focus",
        "inputs",
        "operations",
        "outputs",
    }
    assert session.calculation_history[1]["outputs"]["zones"]
    route_record = session.calculation_history[3]
    assert route_record["outputs"]["candidates"]
    first_feasible = next(
        candidate
        for candidate in route_record["outputs"]["candidates"]
        if candidate.get("route")
    )
    assert route_record["focus"]["roads"] == first_feasible["route"]["road_ids"]
    assert route_record["focus"]["zones"] == [first_feasible["target_zone"]]


def test_learned_cpt_inference_record_exposes_difference_from_expert_cpt():
    payload = json.loads(
        (ROOT / "artifacts" / "full_bayesian_experiment" / "learned_network.json")
        .read_text(encoding="utf-8")
    )
    learned_network = DiscreteBayesianNetwork.from_dict(payload)
    session = LiveSimulation.create(
        generate_random_scenario(20260616, mode="learned"),
        seed=20260616,
        model_name="learned_cpt",
    )

    session.step(network=learned_network)
    session.step(network=learned_network)

    record = session.calculation_history[-1]
    comparison = record["outputs"]["model_comparison"]
    assert record["inputs"]["model"] == "learned_cpt"
    assert comparison["baseline_model"] == "expert_cpt"
    assert comparison["active_model"] == "learned_cpt"
    assert comparison["max_abs_priority_delta"] >= 0.01
    assert any(abs(row["priority_delta"]) >= 0.01 for row in comparison["zones"])


def test_execute_step_changes_clock_and_unit_position():
    session = LiveSimulation.create(generate_random_scenario(12), seed=12)
    _advance_to_phase(session, "execute")
    before = {
        unit_id: dict(state["position"])
        for unit_id, state in session.unit_states.items()
    }

    session.step(execution_minutes=1.0)

    assert session.clock_minutes == 1.0
    assert any(
        state["position"] != before[unit_id]
        for unit_id, state in session.unit_states.items()
    )
    assert session.timeline[-1]["clock_minutes"] == 1.0


def test_execute_can_advance_directly_to_next_unit_transition():
    session = LiveSimulation.create(generate_random_scenario(120), seed=120)
    _advance_to_phase(session, "execute")
    expected = min(
        state["remaining_travel"]
        for state in session.unit_states.values()
        if state["status"] in {"en_route", "returning"}
        and state["remaining_travel"] > 0
    )

    session.step(to_next_transition=True)

    assert session.clock_minutes == pytest.approx(expected)
    record = session.calculation_history[-1]
    assert record["phase"] == "execute"
    assert record["inputs"]["elapsed_minutes"] == pytest.approx(expected)
    assert any(
        before["status"] != after["status"]
        for before, after in zip(
            record["operations"]["before"].values(),
            record["operations"]["after"].values(),
        )
    )


def test_live_session_round_trips_without_recomputing():
    session = LiveSimulation.create(generate_random_scenario(13), seed=13)
    _advance_to_phase(session, "execute")
    payload = session.to_dict()

    restored = LiveSimulation.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.phase == "execute"
    assert restored.calculation_history == session.calculation_history


@pytest.mark.parametrize(
    "event_type", ["road_collapse", "fire_spread", "new_sos", "drone_update"]
)
def test_supported_event_interrupts_and_replans(event_type):
    session = LiveSimulation.create(generate_random_scenario(21), seed=21)
    _advance_to_phase(session, "execute")
    old_plan = deepcopy(session.current_plan)

    event = session.inject_event(event_type)

    assert event["event_type"] == event_type
    assert session.phase == "replan"
    assert len(session.event_log) == 1

    session.step()
    assert session.phase == "infer"
    _advance_to_phase(session, "execute")
    assert session.replan_log[-1]["old_plan"] == old_plan
    assert session.replan_log[-1]["new_plan"] == session.current_plan
    assert session.status == "running"


def test_manual_event_target_is_used_and_duplicate_event_is_not_reapplied():
    session = LiveSimulation.create(generate_random_scenario(22), seed=22)
    _advance_to_phase(session, "execute")
    target = session.available_event_targets("fire_spread")[0]

    event = session.inject_event("fire_spread", target_id=target)
    fire = next(
        zone for zone in session.scenario["zones"] if zone["zone_id"] == target
    )["observations"]["fire"]

    with pytest.raises(ValueError, match="already applied"):
        session.inject_event_payload(event)
    assert next(
        zone for zone in session.scenario["zones"] if zone["zone_id"] == target
    )["observations"]["fire"] == fire


def test_road_collapse_invalidates_only_tasks_using_that_road():
    session = LiveSimulation.create(generate_random_scenario(23), seed=23)
    _advance_to_phase(session, "execute")
    active_ground = {
        unit_id: deepcopy(state["current_task"])
        for unit_id, state in session.unit_states.items()
        if state.get("current_task")
        and state["current_task"]["route"]["route_layer"] == "ground"
    }
    assert len(active_ground) >= 2
    affected_unit, affected_task = next(iter(active_ground.items()))
    target_road = affected_task["route"]["road_ids"][0]

    session.inject_event("road_collapse", target_id=target_road)

    assert session.unit_states[affected_unit]["current_task"] is None
    assert all(
        target_road not in route["road_ids"]
        for route in session.current_plan["routes"]
    )
    for unit_id, task in active_ground.items():
        if target_road not in task["route"]["road_ids"]:
            assert session.unit_states[unit_id]["current_task"] == task


def test_road_collapse_replan_never_reuses_blocked_road():
    session = LiveSimulation.create(generate_random_scenario(21), seed=21)
    _advance_to_phase(session, "execute")
    target_road = next(
        road_id
        for state in session.unit_states.values()
        if state.get("current_task")
        and state["current_task"]["route"].get("route_layer") == "ground"
        for road_id in state["current_task"]["route"]["road_ids"]
    )

    session.inject_event("road_collapse", target_id=target_road)
    while session.phase != "execute":
        session.step()

    assert next(
        road for road in session.scenario["roads"] if road["road_id"] == target_road
    )["status"] == "blocked"
    assert all(
        target_road not in route["road_ids"]
        for route in session.current_plan["routes"]
    )
    assert all(
        target_road not in (state.get("current_task") or {}).get("route", {}).get("road_ids", [])
        for state in session.unit_states.values()
    )


def test_road_collapse_reroutes_returning_unit_away_from_blocked_road():
    session = LiveSimulation.create(generate_random_scenario(1), seed=1)
    _advance_to_phase(session, "execute")
    returning_unit = None
    for _ in range(80):
        returning = [
            state
            for state in session.unit_states.values()
            if state["status"] == "returning"
            and state.get("current_task")
            and state["current_task"]["route"].get("route_layer") == "ground"
            and state["current_task"]["route"]["road_ids"]
        ]
        if returning:
            returning_unit = returning[0]
            break
        session.step(to_next_transition=True)
    assert returning_unit is not None
    target_road = returning_unit["current_task"]["route"]["road_ids"][0]
    unit_id = returning_unit["unit_id"]

    session.inject_event("road_collapse", target_id=target_road)

    rerouted = session.unit_states[unit_id]
    assert rerouted["status"] == "returning"
    assert target_road not in rerouted["current_task"]["route"]["road_ids"]
    assert all(
        target_road not in route["road_ids"]
        for route in session.current_plan["routes"]
    )


def test_live_new_sos_replans_active_car_to_higher_utility_zone():
    scenario = scenario_with_higher_utility_sos()
    event = scenario["events"][0]
    scenario["events"] = []
    session = LiveSimulation.create(scenario, seed=20260616)
    _advance_to_phase(session, "execute")
    assert session.current_plan["assignments"][0]["target_zone"] == "A"

    session.step(execution_minutes=1.0)
    session.inject_event_payload(event)
    while session.phase != "execute":
        session.step()

    assignment = session.current_plan["assignments"][0]
    route = session.current_plan["routes"][0]
    assert assignment["target_zone"] == "B"
    assert route["path"][0].startswith("__unit_RescueCar-1_position")
    assert session.unit_states["RescueCar-1"]["current_node"].startswith(
        "__unit_RescueCar-1_position"
    )


def test_idle_unit_replans_without_waiting_for_all_other_units_to_return():
    session = LiveSimulation.create(_staggered_return_scenario(), seed=20260616)
    _advance_to_phase(session, "execute")

    for _ in range(20):
        session.step(to_next_transition=True)
        redeployed = [
            state
            for state in session.unit_states.values()
            if state["status"] == "en_route"
            and state.get("delivered_targets")
            and (state.get("current_task") or {}).get("target_zone") == "A"
        ]
        active = [
            state
            for state in session.unit_states.values()
            if state["status"] in {"en_route", "rescuing", "returning"}
        ]
        if redeployed and len(active) >= 2 and len(session.completed_zones()) < len(session.scenario["zones"]):
            assert session.phase == "execute"
            assert not any(
                record["phase"] in {"infer", "prioritize", "route", "utility", "allocate"}
                and record["clock_minutes"] == session.clock_minutes
                for record in session.calculation_history
            )
            return

    pytest.fail("idle delivered unit was not redeployed while another unit was active")


def test_drone_recon_automatically_transmits_intel_and_triggers_replan():
    session = LiveSimulation.create(_drone_recon_scenario(), seed=20260616)
    _advance_to_phase(session, "execute")
    assert session.current_plan["assignments"][0]["mission_type"] == "reconnaissance"
    assert session.scenario["zones"][0]["observations"]["drone_confidence"] == 0.0

    session.step(to_next_transition=True)
    session.step(to_next_transition=True)

    observations = session.scenario["zones"][0]["observations"]
    assert observations["drone_confidence"] == 1.0
    assert observations["road_damage"] < 0.90
    assert observations["congestion"] < 0.75
    assert session.phase == "replan"
    assert session.event_log[-1]["event_type"] == "drone_update"
    assert session.event_log[-1]["source"] == "automatic_drone_recon"


def test_paused_session_does_not_advance_and_can_resume():
    session = LiveSimulation.create(generate_random_scenario(31), seed=31)
    _advance_to_phase(session, "execute")
    session.pause()
    frozen = session.to_dict()

    session.step()

    assert session.to_dict() == frozen
    session.resume()
    session.step()
    assert session.clock_minutes == 1.0


def test_event_can_be_queued_while_paused_then_replans_after_resume():
    session = LiveSimulation.create(generate_random_scenario(32), seed=32)
    _advance_to_phase(session, "execute")
    session.pause()

    session.inject_event("fire_spread")

    assert session.status == "paused"
    assert session.phase == "replan"
    session.resume()
    session.step()
    assert session.phase == "infer"


def test_final_report_counts_delivered_rescues_not_drone_recon():
    session = LiveSimulation.create(generate_random_scenario(33), seed=33)
    for _ in range(500):
        if session.status == "completed":
            break
        session.step()
    assert session.status == "completed"

    report = session.build_result()
    delivered = {
        zone_id
        for state in session.unit_states.values()
        for zone_id in state.get("delivered_targets", [])
    }

    assert set(report["completed_zones"]) == delivered
    assert report["rescued_people"] == sum(
        state.get("rescued_people", 0) for state in session.unit_states.values()
    )
    assert report["end_reason"] in {
        "all_rescues_complete",
        "no_feasible_tasks",
        "timeout",
    }
    assert report["algorithm_log"]
    assert report["timeline"]
    assert report["seed"] == 33
