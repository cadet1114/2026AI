import math

import pytest

from emergency_commander.bayesian_network import (
    BayesianNetworkError,
    DiscreteBayesianNetwork,
    fit_cpts,
)
from emergency_commander.expert_cpts import build_expert_network


def test_network_rejects_cpt_row_that_does_not_sum_to_one():
    with pytest.raises(BayesianNetworkError, match="sum to 1"):
        DiscreteBayesianNetwork.from_dict(
            {
                "nodes": [
                    {
                        "name": "rain",
                        "states": ["no", "yes"],
                        "parents": [],
                        "cpt": {"": {"no": 0.8, "yes": 0.3}},
                    }
                ]
            }
        )


def test_exact_inference_updates_trapped_posterior_from_multiple_evidence_nodes():
    network = build_expert_network()

    low_risk = network.query(
        "trapped_people",
        {
            "hazard_intensity": "low",
            "building_damage": "low",
            "fire_severity": "low",
            "smoke_level": "low",
            "sos_signal": "low",
            "human_activity": "low",
        },
    )
    high_risk = network.query(
        "trapped_people",
        {
            "hazard_intensity": "high",
            "building_damage": "high",
            "fire_severity": "high",
            "smoke_level": "high",
            "sos_signal": "high",
            "human_activity": "high",
        },
    )

    assert high_risk["yes"] > 0.85
    assert low_risk["yes"] < 0.20
    assert high_risk["yes"] > low_risk["yes"]
    assert sum(high_risk.values()) == pytest.approx(1.0)


def test_exact_inference_marginalizes_missing_evidence_and_reports_contributions():
    network = build_expert_network()
    evidence = {"sos_signal": "high", "building_damage": "high"}

    posterior = network.query("trapped_people", evidence)
    explanation = network.explain("trapped_people", "yes", evidence)

    assert 0.0 < posterior["yes"] < 1.0
    assert explanation["posterior"] == pytest.approx(posterior["yes"])
    assert {item["evidence"] for item in explanation["contributions"]} == set(evidence)
    assert all(math.isfinite(item["delta"]) for item in explanation["contributions"])


def test_fit_cpts_learns_every_node_with_dirichlet_smoothing():
    expert = build_expert_network()
    records = []
    for index in range(40):
        positive = index < 30
        records.append(
            {
                "hazard_intensity": "high" if positive else "low",
                "building_damage": "high" if positive else "low",
                "fire_severity": "medium" if positive else "low",
                "smoke_level": "medium" if positive else "low",
                "sos_signal": "high" if positive else "low",
                "human_activity": "high" if positive else "low",
                "trapped_people": "yes" if positive else "no",
                "road_damage": "medium" if positive else "low",
                "congestion": "medium" if positive else "low",
                "drone_road_report": "uncertain" if positive else "open",
                "road_passable": "yes",
                "sample_weight": 1.0,
            }
        )

    learned = fit_cpts(expert, records, prior_strength=1.0)

    assert set(learned.node_names) == set(expert.node_names)
    learned_high = learned.query(
        "trapped_people",
        {
            "building_damage": "high",
            "fire_severity": "medium",
            "smoke_level": "medium",
            "sos_signal": "high",
            "human_activity": "high",
        },
    )
    assert learned_high["yes"] > 0.8
    for node in learned.nodes:
        for row in node.cpt.values():
            assert sum(row.values()) == pytest.approx(1.0)
            assert all(probability > 0.0 for probability in row.values())


def test_network_round_trip_preserves_posteriors():
    network = build_expert_network()
    restored = DiscreteBayesianNetwork.from_dict(network.to_dict())
    evidence = {"road_damage": "high", "fire_severity": "high", "congestion": "high"}

    assert restored.query("road_passable", evidence) == pytest.approx(
        network.query("road_passable", evidence)
    )
