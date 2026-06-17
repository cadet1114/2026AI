import json
from pathlib import Path

from emergency_commander.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = PROJECT_ROOT / "examples" / "scenario_input.json"


def test_cli_generates_trains_and_runs_both_modes(tmp_path):
    dataset_path = tmp_path / "training_data.json"
    weights_path = tmp_path / "learned_weights.json"
    metrics_path = tmp_path / "training_metrics.json"
    fixed_output = tmp_path / "fixed_output.json"
    learned_output = tmp_path / "learned_output.json"
    comparison_output = tmp_path / "comparison.json"

    assert main(["generate-data", "--output", str(dataset_path), "--samples", "200", "--seed", "19"]) == 0
    assert main([
        "train",
        "--data", str(dataset_path),
        "--output", str(weights_path),
        "--metrics", str(metrics_path),
    ]) == 0
    assert main([
        "run",
        "--scenario", str(SCENARIO),
        "--output", str(fixed_output),
        "--mode", "fixed",
    ]) == 0
    assert main([
        "run",
        "--scenario", str(SCENARIO),
        "--output", str(learned_output),
        "--mode", "learned",
        "--weights", str(weights_path),
    ]) == 0
    assert main([
        "compare",
        "--data", str(dataset_path),
        "--weights", str(weights_path),
        "--output", str(comparison_output),
        "--limit", "6",
    ]) == 0

    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_output.read_text(encoding="utf-8"))
    learned = json.loads(learned_output.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_output.read_text(encoding="utf-8"))

    assert set(weights) == {"trapped", "passability"}
    assert metrics["learned_validation_brier"]["mean"] < metrics["fixed_validation_brier"]["mean"]
    assert fixed["run_mode"] == "fixed"
    assert learned["run_mode"] == "learned"
    assert fixed.keys() == learned.keys()
    assert fixed["weights_used"]["trapped"] != learned["weights_used"]["trapped"]
    assert comparison["summary"]["case_count"] == 6
