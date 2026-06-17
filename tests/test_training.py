import copy

import pytest

from emergency_commander.training import (
    apply_probability_weights,
    brier_scores,
    compare_weight_sets,
    generate_synthetic_dataset,
    train_probability_weights,
)


FIXED_WEIGHTS = {
    "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
    "passability": {"road_damage": 0.45, "fire_risk": 0.35, "congestion": 0.20, "drone_confidence": 0.0},
}


def test_synthetic_dataset_is_reproducible_and_has_binary_labels():
    first = generate_synthetic_dataset(sample_count=120, seed=7)
    second = generate_synthetic_dataset(sample_count=120, seed=7)

    assert first == second
    assert len(first["samples"]) == 120
    assert {sample["trapped_ground_truth"] for sample in first["samples"]} <= {0, 1}
    assert {sample["passable_ground_truth"] for sample in first["samples"]} <= {0, 1}


def test_learned_weights_are_normalized_and_improve_validation_brier_score():
    dataset = generate_synthetic_dataset(sample_count=300, seed=11)
    train_samples = dataset["samples"][:240]
    validation_samples = dataset["samples"][240:]

    learned = train_probability_weights(train_samples)
    fixed_score = brier_scores(validation_samples, FIXED_WEIGHTS)
    learned_score = brier_scores(validation_samples, learned)

    assert sum(learned["trapped"].values()) == pytest.approx(1.0)
    assert sum(learned["passability"].values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for group in learned.values() for value in group.values())
    assert learned_score["mean"] < fixed_score["mean"]
    assert learned != FIXED_WEIGHTS


def test_apply_probability_weights_only_replaces_learnable_groups():
    scenario = {
        "config": {
            "weights": {
                **copy.deepcopy(FIXED_WEIGHTS),
                "utility": {"alpha": 0.3},
                "astar_risk": {"fire": 0.35},
            }
        }
    }
    learned = {
        "trapped": {"sos": 0.5, "collapse": 0.2, "human_activity": 0.2, "smoke": 0.1},
        "passability": {"road_damage": 0.5, "fire_risk": 0.2, "congestion": 0.2, "drone_confidence": 0.1},
    }

    updated = apply_probability_weights(scenario, learned)

    assert updated["config"]["weights"]["trapped"] == learned["trapped"]
    assert updated["config"]["weights"]["utility"] == {"alpha": 0.3}
    assert scenario["config"]["weights"]["trapped"] == FIXED_WEIGHTS["trapped"]


def test_compare_weight_sets_reports_case_level_improvements():
    samples = generate_synthetic_dataset(sample_count=100, seed=3)["samples"][:5]
    learned = train_probability_weights(generate_synthetic_dataset(sample_count=300, seed=11)["samples"])

    comparison = compare_weight_sets(samples, FIXED_WEIGHTS, learned)

    assert len(comparison["cases"]) == 5
    assert all("fixed_error" in case and "learned_error" in case for case in comparison["cases"])
    assert comparison["summary"]["case_count"] == 5
    assert comparison["summary"]["improved_cases"] >= 0
