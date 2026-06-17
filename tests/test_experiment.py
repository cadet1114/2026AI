from emergency_commander.experiment import (
    evaluate_cross_validation,
    generate_hybrid_dataset,
)
from emergency_commander.public_data import map_usgs_hazard_intensity


def public_rows():
    return [
        {"id": "eq-low", "mag": 4.1, "depth": 12.0},
        {"id": "eq-mid", "mag": 5.6, "depth": 18.0},
        {"id": "eq-high", "mag": 7.2, "depth": 10.0},
    ]


def test_usgs_mapping_uses_magnitude_and_depth():
    assert map_usgs_hazard_intensity(4.1, 10.0) == "low"
    assert map_usgs_hazard_intensity(5.6, 10.0) == "medium"
    assert map_usgs_hazard_intensity(7.1, 10.0) == "high"
    assert map_usgs_hazard_intensity(5.4, 250.0) == "low"


def test_hybrid_dataset_is_reproducible_and_tracks_provenance_missingness_noise():
    first = generate_hybrid_dataset(
        public_rows(), sample_count=240, seed=13, missing_rate=0.15, label_noise=0.05
    )
    second = generate_hybrid_dataset(
        public_rows(), sample_count=240, seed=13, missing_rate=0.15, label_noise=0.05
    )

    assert first == second
    assert len(first["records"]) == 240
    assert first["metadata"]["public_anchor_rows"] == 3
    assert first["metadata"]["label_source"] == "bayesian_ancestral_simulation"
    assert all(record["provenance"]["hazard_intensity"] == "USGS" for record in first["records"])
    assert all(record["provenance"]["trapped_people"] == "simulated" for record in first["records"])
    assert any("sos_signal" not in record for record in first["records"])
    assert {record["trapped_people"] for record in first["records"]} == {"no", "yes"}


def test_cross_validation_reports_calibration_classification_and_robustness():
    dataset = generate_hybrid_dataset(
        public_rows(), sample_count=600, seed=21, missing_rate=0.08, label_noise=0.03
    )

    result = evaluate_cross_validation(dataset["records"], folds=3, seed=21)

    assert result["folds"] == 3
    assert set(result["aggregate"]) == {"expert_cpt", "learned_cpt"}
    for model_metrics in result["aggregate"].values():
        for target in ("trapped_people", "road_passable"):
            metrics = model_metrics[target]
            assert 0.0 <= metrics["brier"] <= 1.0
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert metrics["calibration_bins"]
    assert result["public_anchor_report"]["target_labels_available"] is False
    assert result["robustness"]["missing_evidence"]["rates"] == [0.0, 0.2, 0.4]
    assert result["aggregate"]["learned_cpt"]["trapped_people"]["brier"] <= 0.30
