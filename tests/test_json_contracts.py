import json
from pathlib import Path

import pytest

from emergency_commander.contracts import (
    ContractValidationError,
    validate_decision_output,
    validate_scenario,
)
from emergency_commander.pipeline import run_pipeline
from emergency_commander.input_adapter import ScenarioValidationError, normalize_scenario


ROOT = Path(__file__).resolve().parents[1]


def load_scenario():
    return json.loads(
        (ROOT / "examples" / "scenario_input.json").read_text(encoding="utf-8")
    )


def test_example_scenario_matches_published_schema():
    validate_scenario(load_scenario())


def test_schema_rejects_unknown_top_level_field():
    scenario = load_scenario()
    scenario["silent_typo"] = True

    with pytest.raises(ContractValidationError, match="silent_typo"):
        validate_scenario(scenario)


def test_normalization_enforces_published_schema():
    scenario = load_scenario()
    scenario["silent_typo"] = True

    with pytest.raises(ScenarioValidationError, match="silent_typo"):
        normalize_scenario(scenario)


def test_pipeline_output_matches_published_schema():
    output = run_pipeline(load_scenario(), process_events=True)

    validate_decision_output(output)
    assert "utility_matrix" in output
    assert output["timeline"][0]["plan"]["utility_matrix"]
    assert output["timeline"][0]["scenario_state"]["roads"]
    feasible = next(
        item
        for item in output["timeline"][0]["plan"]["utility_matrix"]
        if item["feasible"]
    )
    assert feasible["utility_breakdown"]
    assert feasible["explanation"]
