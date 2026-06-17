from __future__ import annotations

from typing import Any

from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.expert_cpts import build_expert_network


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def discretize_level(value: float) -> str:
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"


def _drone_report(observations: dict[str, Any]) -> str:
    explicit = observations.get("drone_road_report")
    if explicit in {"blocked", "uncertain", "open"}:
        return explicit
    confidence = float(observations.get("drone_confidence", 0.0))
    if confidence < 0.60:
        return "uncertain"
    if observations["road_damage"] >= 2.0 / 3.0 or observations["fire"] >= 0.80:
        return "blocked"
    return "open"


def observations_to_evidence(observations: dict[str, Any]) -> dict[str, str]:
    hazard = float(
        observations.get(
            "hazard_intensity",
            max(
                observations["building_collapse"],
                observations["fire"],
                observations["road_damage"],
            ),
        )
    )
    return {
        "hazard_intensity": discretize_level(hazard),
        "building_damage": discretize_level(observations["building_collapse"]),
        "fire_severity": discretize_level(observations["fire"]),
        "smoke_level": discretize_level(observations["smoke"]),
        "sos_signal": discretize_level(observations["sos_signal"]),
        "human_activity": discretize_level(observations["human_activity"]),
        "road_damage": discretize_level(observations["road_damage"]),
        "congestion": discretize_level(observations["congestion"]),
        "drone_road_report": _drone_report(observations),
    }


def assess_zones(
    scenario: dict[str, Any],
    network: DiscreteBayesianNetwork | None = None,
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Infer zone posteriors and calculate downstream risk and priority."""
    network = network or build_expert_network()
    if model_name is None:
        model_name = "learned_cpt" if scenario.get("run_mode") == "learned" else "expert_cpt"
    weights = scenario["config"]["weights"]
    life_weights = weights["life_risk"]
    priority_weights = weights["priority"]
    assessments: list[dict[str, Any]] = []

    trapped_evidence_names = {
        "hazard_intensity",
        "building_damage",
        "fire_severity",
        "smoke_level",
        "sos_signal",
        "human_activity",
    }
    road_evidence_names = {
        "hazard_intensity",
        "fire_severity",
        "road_damage",
        "congestion",
        "drone_road_report",
    }

    for zone in scenario["zones"]:
        obs = zone["observations"]
        evidence = observations_to_evidence(obs)
        trapped_evidence = {
            name: state for name, state in evidence.items() if name in trapped_evidence_names
        }
        road_evidence = {
            name: state for name, state in evidence.items() if name in road_evidence_names
        }
        trapped_prob = network.query("trapped_people", trapped_evidence)["yes"]
        passability_prob = network.query("road_passable", road_evidence)["yes"]
        trapped_explanation = network.explain(
            "trapped_people", "yes", trapped_evidence
        )
        road_explanation = network.explain("road_passable", "yes", road_evidence)
        life_risk = _clip(
            life_weights["fire"] * obs["fire"]
            + life_weights["trapped_prob"] * trapped_prob
            + life_weights["time_urgency"] * obs["time_urgency"]
        )
        priority_score = _clip(
            priority_weights["trapped_prob"] * trapped_prob
            + priority_weights["life_risk"] * life_risk
            + priority_weights["time_urgency"] * obs["time_urgency"]
            + priority_weights["accessibility"] * passability_prob
        )
        assessments.append(
            {
                "zone_id": zone["zone_id"],
                "node_id": zone["node_id"],
                "trapped_prob": round(trapped_prob, 6),
                "passability_prob": round(passability_prob, 6),
                "life_risk": round(life_risk, 6),
                "priority_score": round(priority_score, 6),
                "inference_model": model_name,
                "bayesian_evidence": evidence,
                "trapped_explanation": trapped_explanation,
                "passability_explanation": road_explanation,
            }
        )

    return sorted(assessments, key=lambda item: item["priority_score"], reverse=True)

