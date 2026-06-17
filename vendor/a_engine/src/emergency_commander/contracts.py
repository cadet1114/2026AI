from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


class ContractValidationError(ValueError):
    """Raised when a JSON document violates a published project contract."""


def validate_document(document: dict[str, Any], schema_name: str) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "document"
    raise ContractValidationError(f"{path}: {error.message}")


def validate_scenario(document: dict[str, Any]) -> None:
    validate_document(document, "scenario.schema.json")


def validate_decision_output(document: dict[str, Any]) -> None:
    validate_document(document, "decision_output.schema.json")
