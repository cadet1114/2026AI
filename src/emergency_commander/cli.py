from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Sequence

import yaml

from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.experiment import evaluate_cross_validation, generate_hybrid_dataset
from emergency_commander.pipeline import run_pipeline
from emergency_commander.public_data import fetch_usgs_catalog, load_usgs_catalog
from emergency_commander.training import (
    DEFAULT_PROBABILITY_WEIGHTS,
    apply_probability_weights,
    brier_scores,
    compare_weight_sets,
    generate_synthetic_dataset,
    train_probability_weights,
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Emergency Commander algorithm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="generate semi-synthetic labeled data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--samples", type=int, default=200)
    generate.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="fit learnable probability weights")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--metrics", required=True)
    train.add_argument("--validation-ratio", type=float, default=0.2)

    run = subparsers.add_parser("run", help="run the complete decision pipeline")
    run.add_argument("--scenario", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--mode", choices=("fixed", "learned"), required=True)
    run.add_argument("--weights", help="learned_weights.json, required in learned mode")
    run.add_argument("--model", help="learned Bayesian network JSON, required for CPT learned mode")
    run.add_argument("--no-events", action="store_true", help="skip dynamic event processing")

    compare = subparsers.add_parser("compare", help="compare fixed and learned weights per case")
    compare.add_argument("--data", required=True)
    compare.add_argument("--weights", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--limit", type=int)

    public = subparsers.add_parser("download-public", help="download bounded official USGS data")
    public.add_argument("--output", required=True)
    public.add_argument("--metadata", required=True)
    public.add_argument("--start-time", default="2024-01-01")
    public.add_argument("--end-time", default="2025-12-31")
    public.add_argument("--minimum-magnitude", type=float, default=4.5)
    public.add_argument("--limit", type=int, default=2000)

    experiment = subparsers.add_parser(
        "run-experiment", help="run full hybrid CPT learning and evaluation"
    )
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--public-data", required=True)
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    dataset = generate_synthetic_dataset(sample_count=args.samples, seed=args.seed)
    _write_json(args.output, dataset)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    if not 0.05 <= args.validation_ratio <= 0.5:
        raise ValueError("validation-ratio must be between 0.05 and 0.5")
    dataset = _read_json(args.data)
    samples = dataset["samples"]
    split_index = int(len(samples) * (1.0 - args.validation_ratio))
    train_samples = samples[:split_index]
    validation_samples = samples[split_index:]
    learned_weights = train_probability_weights(train_samples)
    metrics = {
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "fixed_validation_brier": brier_scores(validation_samples, DEFAULT_PROBABILITY_WEIGHTS),
        "learned_validation_brier": brier_scores(validation_samples, learned_weights),
    }
    _write_json(args.output, learned_weights)
    _write_json(args.metrics, metrics)
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    scenario = _read_json(args.scenario)
    network = None
    model_name = "expert_cpt"
    if args.mode == "learned":
        if args.model:
            network = DiscreteBayesianNetwork.from_dict(_read_json(args.model))
            scenario["mode"] = "learned"
            scenario["run_mode"] = "learned"
            model_name = "learned_cpt"
        elif args.weights:
            scenario = apply_probability_weights(scenario, _read_json(args.weights))
            model_name = "expert_cpt_with_legacy_linear_config"
        else:
            raise ValueError("--model is required when --mode learned")
    else:
        scenario["mode"] = "fixed"
        scenario["run_mode"] = "fixed"
    output = run_pipeline(
        scenario,
        process_events=not args.no_events,
        network=network,
        model_name=model_name,
    )
    _write_json(args.output, output)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    samples = _read_json(args.data)["samples"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    learned_weights = _read_json(args.weights)
    comparison = compare_weight_sets(samples, DEFAULT_PROBABILITY_WEIGHTS, learned_weights)
    _write_json(args.output, comparison)
    return 0


def _run_download_public(args: argparse.Namespace) -> int:
    metadata = fetch_usgs_catalog(
        args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        minimum_magnitude=args.minimum_magnitude,
        limit=args.limit,
    )
    _write_json(args.metadata, metadata)
    return 0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _experiment_report(metrics: dict[str, Any], metadata: dict[str, Any]) -> str:
    expert = metrics["aggregate"]["expert_cpt"]
    learned = metrics["aggregate"]["learned_cpt"]
    lines = [
        "# Full Bayesian Network Experiment Report",
        "",
        "## Data Truth Boundary",
        "",
        "USGS records anchor only `hazard_intensity`. All trapped-person and road-passability labels are generated by documented Bayesian ancestral simulation; they are not claimed as real rescue labels.",
        "",
        "## Runtime",
        "",
        f"- Samples: `{metadata['sample_count']}`",
        f"- Elapsed seconds: `{metadata['elapsed_seconds']}`",
        f"- Peak traced memory MB: `{metadata['peak_memory_mb']}`",
        "",
        "## Held-out Five-fold Metrics",
        "",
        "| Target | Model | Brier | Accuracy | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in ("trapped_people", "road_passable"):
        for label, values in (("Expert CPT", expert[target]), ("Learned CPT", learned[target])):
            lines.append(
                f"| {target} | {label} | {values['brier']:.4f} | {values['accuracy']:.4f} | {values['f1']:.4f} | {values['roc_auc']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Public Anchor Coverage",
            "",
            f"- Unique USGS rows: `{metrics['public_anchor_report']['unique_public_rows']}`",
            f"- Hazard distribution: `{metrics['public_anchor_report']['hazard_distribution']}`",
            "- Target-label metrics are intentionally reported only for simulated labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_full_experiment(args: argparse.Namespace) -> int:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    evaluation_config = config["evaluation"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = load_usgs_catalog(args.public_data)

    tracemalloc.start()
    started = time.perf_counter()
    dataset = generate_hybrid_dataset(
        public_rows,
        sample_count=int(dataset_config["sample_count"]),
        seed=int(dataset_config["seed"]),
        missing_rate=float(dataset_config["missing_rate"]),
        label_noise=float(dataset_config["label_noise"]),
    )
    metrics = evaluate_cross_validation(
        dataset["records"],
        folds=int(evaluation_config["folds"]),
        seed=int(dataset_config["seed"]),
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = {
        **dataset["metadata"],
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_mb": round(peak / (1024 * 1024), 3),
    }
    _write_jsonl(output_dir / "hybrid_dataset.jsonl", dataset["records"])
    _write_json(output_dir / "experiment_metrics.json", metrics)
    _write_json(output_dir / "learned_network.json", metrics["learned_network"])
    _write_json(output_dir / "runtime.json", runtime)
    _write_json(output_dir / "config_snapshot.json", config)
    (output_dir / "experiment_report.md").write_text(
        _experiment_report(metrics, runtime), encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate-data":
        return _run_generate(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "run":
        return _run_pipeline(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "download-public":
        return _run_download_public(args)
    if args.command == "run-experiment":
        return _run_full_experiment(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
