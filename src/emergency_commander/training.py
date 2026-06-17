from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


TRAPPED_FEATURES = ("sos_signal", "building_collapse", "human_activity", "smoke")
TRAPPED_KEYS = ("sos", "collapse", "human_activity", "smoke")
PASSABILITY_FEATURES = ("road_damage", "fire", "congestion", "drone_confidence")
PASSABILITY_KEYS = ("road_damage", "fire_risk", "congestion", "drone_confidence")

HIDDEN_TRAPPED_WEIGHTS = np.array([0.55, 0.15, 0.25, 0.05], dtype=float)
HIDDEN_PASSABILITY_WEIGHTS = np.array([0.25, 0.50, 0.10, 0.15], dtype=float)

DEFAULT_PROBABILITY_WEIGHTS = {
    "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
    "passability": {
        "road_damage": 0.45,
        "fire_risk": 0.35,
        "congestion": 0.20,
        "drone_confidence": 0.0,
    },
}


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def generate_synthetic_dataset(*, sample_count: int = 200, seed: int = 42) -> dict[str, Any]:
    """Generate reproducible semi-synthetic samples with hidden expert weights."""
    if not 100 <= sample_count <= 300:
        raise ValueError("sample_count must be between 100 and 300")
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []

    for sample_id in range(sample_count):
        values = rng.uniform(0.0, 1.0, size=8)
        observations = {
            "sos_signal": float(values[0]),
            "building_collapse": float(values[1]),
            "human_activity": float(values[2]),
            "smoke": float(values[3]),
            "road_damage": float(values[4]),
            "fire": float(values[5]),
            "congestion": float(values[6]),
            "drone_confidence": float(values[7]),
        }
        trapped_probability = float(values[:4] @ HIDDEN_TRAPPED_WEIGHTS)
        passability_probability = _clip_probability(
            1.0
            - float(values[4:7] @ HIDDEN_PASSABILITY_WEIGHTS[:3])
            + float(values[7] * HIDDEN_PASSABILITY_WEIGHTS[3])
        )
        samples.append(
            {
                "sample_id": f"S{sample_id + 1:03d}",
                "observations": observations,
                "trapped_ground_truth": int(rng.random() < trapped_probability),
                "passable_ground_truth": int(rng.random() < passability_probability),
            }
        )

    return {
        "metadata": {
            "source": "semi_synthetic_expert_rules",
            "sample_count": sample_count,
            "seed": seed,
        },
        "samples": samples,
    }


def _normalized_nonnegative_least_squares(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(features, targets, rcond=None)
    coefficients = np.clip(coefficients, 0.0, None)
    total = float(coefficients.sum())
    if total == 0.0:
        return np.full(features.shape[1], 1.0 / features.shape[1])
    return coefficients / total


def train_probability_weights(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Fit the two learnable linear probability groups from labeled samples."""
    if len(samples) < 20:
        raise ValueError("at least 20 labeled samples are required")
    trapped_x = np.array(
        [[sample["observations"][name] for name in TRAPPED_FEATURES] for sample in samples],
        dtype=float,
    )
    trapped_y = np.array([sample["trapped_ground_truth"] for sample in samples], dtype=float)
    passability_x = np.array(
        [
            [
                sample["observations"]["road_damage"],
                sample["observations"]["fire"],
                sample["observations"]["congestion"],
                -sample["observations"]["drone_confidence"],
            ]
            for sample in samples
        ],
        dtype=float,
    )
    passability_risk_y = 1.0 - np.array(
        [sample["passable_ground_truth"] for sample in samples], dtype=float
    )

    trapped_weights = _normalized_nonnegative_least_squares(trapped_x, trapped_y)
    passability_weights = _normalized_nonnegative_least_squares(
        passability_x, passability_risk_y
    )
    return {
        "trapped": {
            key: round(float(value), 8)
            for key, value in zip(TRAPPED_KEYS, trapped_weights, strict=True)
        },
        "passability": {
            key: round(float(value), 8)
            for key, value in zip(PASSABILITY_KEYS, passability_weights, strict=True)
        },
    }


def _predict(sample: dict[str, Any], weights: dict[str, dict[str, float]]) -> tuple[float, float]:
    obs = sample["observations"]
    trapped = _clip_probability(
        weights["trapped"]["sos"] * obs["sos_signal"]
        + weights["trapped"]["collapse"] * obs["building_collapse"]
        + weights["trapped"]["human_activity"] * obs["human_activity"]
        + weights["trapped"]["smoke"] * obs["smoke"]
    )
    passability = _clip_probability(
        1.0
        - weights["passability"]["road_damage"] * obs["road_damage"]
        - weights["passability"]["fire_risk"] * obs["fire"]
        - weights["passability"]["congestion"] * obs["congestion"]
        + weights["passability"]["drone_confidence"] * obs["drone_confidence"]
    )
    return trapped, passability


def brier_scores(
    samples: list[dict[str, Any]], weights: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Return lower-is-better probability calibration scores."""
    trapped_errors = []
    passability_errors = []
    for sample in samples:
        trapped, passability = _predict(sample, weights)
        trapped_errors.append((trapped - sample["trapped_ground_truth"]) ** 2)
        passability_errors.append((passability - sample["passable_ground_truth"]) ** 2)
    trapped_score = float(np.mean(trapped_errors))
    passability_score = float(np.mean(passability_errors))
    return {
        "trapped": round(trapped_score, 8),
        "passability": round(passability_score, 8),
        "mean": round((trapped_score + passability_score) / 2.0, 8),
    }


def compare_weight_sets(
    samples: list[dict[str, Any]],
    fixed_weights: dict[str, dict[str, float]],
    learned_weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Create case-level prediction evidence for fixed vs learned parameters."""
    cases = []
    improved_cases = 0
    for index, sample in enumerate(samples):
        fixed_trapped, fixed_passability = _predict(sample, fixed_weights)
        learned_trapped, learned_passability = _predict(sample, learned_weights)
        trapped_label = sample["trapped_ground_truth"]
        passability_label = sample["passable_ground_truth"]
        fixed_error = (
            (fixed_trapped - trapped_label) ** 2
            + (fixed_passability - passability_label) ** 2
        ) / 2.0
        learned_error = (
            (learned_trapped - trapped_label) ** 2
            + (learned_passability - passability_label) ** 2
        ) / 2.0
        improved = learned_error < fixed_error
        improved_cases += int(improved)
        cases.append(
            {
                "case_id": sample.get("case_id", sample.get("sample_id", f"case_{index + 1}")),
                "description": sample.get("description", ""),
                "labels": {
                    "trapped_ground_truth": trapped_label,
                    "passable_ground_truth": passability_label,
                },
                "fixed_prediction": {
                    "trapped_prob": round(fixed_trapped, 6),
                    "passability_prob": round(fixed_passability, 6),
                },
                "learned_prediction": {
                    "trapped_prob": round(learned_trapped, 6),
                    "passability_prob": round(learned_passability, 6),
                },
                "fixed_error": round(fixed_error, 8),
                "learned_error": round(learned_error, 8),
                "learned_improved": improved,
            }
        )
    return {
        "summary": {
            "case_count": len(cases),
            "improved_cases": improved_cases,
            "fixed_brier": brier_scores(samples, fixed_weights),
            "learned_brier": brier_scores(samples, learned_weights),
        },
        "cases": cases,
    }


def apply_probability_weights(
    scenario: dict[str, Any], learned_weights: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Return a learned-mode scenario while preserving fixed downstream weights."""
    updated = deepcopy(scenario)
    for group, required_keys in (
        ("trapped", set(TRAPPED_KEYS)),
        ("passability", set(PASSABILITY_KEYS)),
    ):
        if set(learned_weights.get(group, {})) != required_keys:
            raise ValueError(f"learned {group} weights do not match the required schema")
        updated["config"]["weights"][group] = deepcopy(learned_weights[group])
    updated.pop("mode", None)
    updated["run_mode"] = "learned"
    return updated
