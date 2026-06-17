from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_EVENT_TYPES = {"road_collapse", "drone_update", "new_sos", "fire_spread"}


def _set_dot_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _find_target(scenario: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    target_id = event["target_id"]
    event_type = event["event_type"]
    if event_type == "road_collapse":
        collections = (("roads", "road_id"),)
    else:
        collections = (("zones", "zone_id"), ("roads", "road_id"))

    for collection_name, id_field in collections:
        for item in scenario.get(collection_name, []):
            if item.get(id_field) == target_id:
                return item
    raise ValueError(f"event target '{target_id}' was not found")


def apply_event(scenario: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Return a copied scenario with one supported event applied."""
    event_type = event.get("event_type")
    if event_type not in SUPPORTED_EVENT_TYPES:
        raise ValueError(f"unsupported event_type '{event_type}'")
    updated = deepcopy(scenario)
    target = _find_target(updated, event)
    for path, value in event.get("changes", {}).items():
        _set_dot_path(target, path, value)
    return updated

