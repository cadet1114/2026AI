from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from emergency_commander.bayesian_network import (
    BayesianNode,
    DiscreteBayesianNetwork,
    fit_cpts,
)
from emergency_commander.expert_cpts import build_expert_network
from emergency_commander.public_data import map_usgs_hazard_intensity


TRAPPED_EVIDENCE = (
    "hazard_intensity",
    "building_damage",
    "fire_severity",
    "smoke_level",
    "sos_signal",
    "human_activity",
)
ROAD_EVIDENCE = (
    "hazard_intensity",
    "fire_severity",
    "road_damage",
    "congestion",
    "drone_road_report",
)
EVIDENCE_NODES = tuple(dict.fromkeys((*TRAPPED_EVIDENCE, *ROAD_EVIDENCE)))


def _simulation_network() -> DiscreteBayesianNetwork:
    """Create a documented distribution shift so learning has a real calibration task."""
    expert = build_expert_network()
    nodes = []
    for node in expert.nodes:
        cpt = deepcopy(node.cpt)
        if node.name == "trapped_people":
            for key, row in cpt.items():
                sos = key[3]
                activity = key[4]
                shift = (0.10 if sos == "high" else -0.03 if sos == "low" else 0.0)
                shift += 0.05 if activity == "high" else 0.0
                yes = min(0.985, max(0.015, row["yes"] + shift))
                cpt[key] = {"no": 1.0 - yes, "yes": yes}
        elif node.name == "road_passable":
            for key, row in cpt.items():
                fire = key[1]
                report = key[3]
                shift = -0.12 if fire == "high" else 0.0
                shift += 0.08 if report == "open" else -0.05 if report == "blocked" else 0.0
                yes = min(0.985, max(0.015, row["yes"] + shift))
                cpt[key] = {"no": 1.0 - yes, "yes": yes}
        nodes.append(BayesianNode(node.name, node.states, node.parents, cpt))
    return DiscreteBayesianNetwork(nodes)


def _sample_state(probabilities: dict[str, float], rng: np.random.Generator) -> str:
    states = list(probabilities)
    values = np.array([probabilities[state] for state in states], dtype=float)
    return str(rng.choice(states, p=values / values.sum()))


def _flip_binary(value: str) -> str:
    return "no" if value == "yes" else "yes"


def generate_hybrid_dataset(
    public_rows: list[dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
    missing_rate: float,
    label_noise: float,
) -> dict[str, Any]:
    if not public_rows:
        raise ValueError("at least one public anchor row is required")
    if not 0.0 <= missing_rate < 1.0 or not 0.0 <= label_noise < 0.5:
        raise ValueError("invalid missingness or label noise")
    rng = np.random.default_rng(seed)
    generator = _simulation_network()
    records = []
    for index in range(sample_count):
        anchor = public_rows[index % len(public_rows)]
        assignment = {
            "hazard_intensity": map_usgs_hazard_intensity(
                float(anchor["mag"]), float(anchor["depth"])
            )
        }
        for node in generator.nodes[1:]:
            key = tuple(assignment[parent] for parent in node.parents)
            assignment[node.name] = _sample_state(node.cpt[key], rng)

        for target in ("trapped_people", "road_passable"):
            if rng.random() < label_noise:
                assignment[target] = _flip_binary(assignment[target])

        record: dict[str, Any] = dict(assignment)
        for evidence_name in EVIDENCE_NODES:
            if evidence_name != "hazard_intensity" and rng.random() < missing_rate:
                record.pop(evidence_name, None)
        record.update(
            {
                "record_id": f"HYB-{index + 1:06d}",
                "sample_weight": 1.0,
                "public_anchor": {
                    "dataset": "USGS Earthquake Catalog",
                    "id": anchor.get("id", ""),
                    "magnitude": float(anchor["mag"]),
                    "depth_km": float(anchor["depth"]),
                },
                "provenance": {
                    "hazard_intensity": "USGS",
                    "network_structure": "expert",
                    "trapped_people": "simulated",
                    "road_passable": "simulated",
                },
            }
        )
        records.append(record)
    return {
        "metadata": {
            "seed": seed,
            "sample_count": sample_count,
            "public_anchor_rows": len(public_rows),
            "missing_rate": missing_rate,
            "label_noise": label_noise,
            "label_source": "bayesian_ancestral_simulation",
            "truth_claim": "Only hazard intensity is public-data anchored; target labels are simulated.",
        },
        "records": records,
    }


def _evidence(record: dict[str, Any], target: str) -> dict[str, str]:
    names = TRAPPED_EVIDENCE if target == "trapped_people" else ROAD_EVIDENCE
    return {name: record[name] for name in names if name in record}


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    if len(set(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, probabilities))
    else:
        roc_auc = 0.5
    fraction_positive, mean_predicted = calibration_curve(
        labels, probabilities, n_bins=8, strategy="quantile"
    )
    return {
        "brier": round(float(brier_score_loss(labels, probabilities)), 8),
        "accuracy": round(float(accuracy_score(labels, predictions)), 8),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 8),
        "roc_auc": round(roc_auc, 8),
        "calibration_bins": [
            {
                "mean_predicted": round(float(predicted), 8),
                "fraction_positive": round(float(actual), 8),
            }
            for predicted, actual in zip(mean_predicted, fraction_positive, strict=True)
        ],
    }


def _evaluate_model(
    network: DiscreteBayesianNetwork, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for target in ("trapped_people", "road_passable"):
        labels = [int(record[target] == "yes") for record in records]
        probabilities = [
            network.query(target, _evidence(record, target))["yes"] for record in records
        ]
        output[target] = _metrics(labels, probabilities)
    return output


def _drop_evidence(
    records: list[dict[str, Any]], rate: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    copied = deepcopy(records)
    for record in copied:
        for name in EVIDENCE_NODES:
            if name != "hazard_intensity" and rng.random() < rate:
                record.pop(name, None)
    return copied


def evaluate_cross_validation(
    records: list[dict[str, Any]], *, folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    expert = build_expert_network()
    strata = np.array(
        [f"{record['trapped_people']}|{record['road_passable']}" for record in records]
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    per_fold = []
    out_of_fold: dict[str, dict[str, list[float] | list[int]]] = {
        model: {
            f"{target}_{kind}": []
            for target in ("trapped_people", "road_passable")
            for kind in ("labels", "probabilities")
        }
        for model in ("expert_cpt", "learned_cpt")
    }
    indices = np.arange(len(records))
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(indices, strata), start=1):
        train = [records[int(index)] for index in train_indices]
        test = [records[int(index)] for index in test_indices]
        learned = fit_cpts(expert, train, prior_strength=2.0)
        fold_result = {"fold": fold_index, "test_samples": len(test), "models": {}}
        for model_name, network in (("expert_cpt", expert), ("learned_cpt", learned)):
            model_metrics = {}
            for target in ("trapped_people", "road_passable"):
                labels = [int(record[target] == "yes") for record in test]
                probabilities = [
                    network.query(target, _evidence(record, target))["yes"]
                    for record in test
                ]
                model_metrics[target] = _metrics(labels, probabilities)
                out_of_fold[model_name][f"{target}_labels"].extend(labels)
                out_of_fold[model_name][f"{target}_probabilities"].extend(probabilities)
            fold_result["models"][model_name] = model_metrics
        per_fold.append(fold_result)

    aggregate = {}
    for model_name, values in out_of_fold.items():
        aggregate[model_name] = {}
        for target in ("trapped_people", "road_passable"):
            aggregate[model_name][target] = _metrics(
                values[f"{target}_labels"], values[f"{target}_probabilities"]
            )

    final_learned = fit_cpts(expert, records, prior_strength=2.0)
    rng = np.random.default_rng(seed + 1000)
    rates = [0.0, 0.2, 0.4]
    robustness = {
        str(rate): _evaluate_model(final_learned, _drop_evidence(records, rate, rng))
        for rate in rates
    }
    hazard_counts = Counter(record["hazard_intensity"] for record in records)
    public_ids = {record["public_anchor"]["id"] for record in records}
    return {
        "folds": folds,
        "per_fold": per_fold,
        "aggregate": aggregate,
        "robustness": {
            "missing_evidence": {"rates": rates, "metrics": robustness}
        },
        "public_anchor_report": {
            "dataset": "USGS Earthquake Catalog",
            "unique_public_rows": len(public_ids),
            "hazard_distribution": dict(sorted(hazard_counts.items())),
            "target_labels_available": False,
            "note": "USGS anchors hazard intensity only; rescue target metrics use simulated labels.",
        },
        "learned_network": final_learned.to_dict(),
    }

