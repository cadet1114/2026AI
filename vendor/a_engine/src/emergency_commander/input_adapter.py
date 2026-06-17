from __future__ import annotations

from copy import deepcopy
from typing import Any

from emergency_commander.contracts import ContractValidationError, validate_scenario
from emergency_commander.replanning import SUPPORTED_EVENT_TYPES


class ScenarioValidationError(ValueError):
    """Raised when a scenario does not match the algorithm input contract."""


REQUIRED_OBSERVATIONS = (
    "sos_signal",
    "building_collapse",
    "smoke",
    "fire",
    "road_damage",
    "human_activity",
    "congestion",
    "time_urgency",
    "drone_confidence",
)

ROAD_RISK_FIELDS = ("fire", "damage", "congestion", "secondary_disaster")

WEIGHT_FIELDS = {
    "trapped": ("sos", "collapse", "human_activity", "smoke"),
    "passability": ("road_damage", "fire_risk", "congestion", "drone_confidence"),
    "life_risk": ("fire", "trapped_prob", "time_urgency"),
    "priority": ("trapped_prob", "life_risk", "time_urgency", "accessibility"),
    "utility": ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"),
    "astar_risk": ("fire", "damage", "congestion", "secondary"),
}


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ScenarioValidationError(f"{context} is missing required field '{key}'")
    return mapping[key]


def _unit_interval(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScenarioValidationError(f"{field} must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ScenarioValidationError(f"{field} must be within [0, 1]")
    return numeric


def _positive_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ScenarioValidationError(f"{field} must be a positive number")
    return float(value)


def _nonnegative_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
        raise ScenarioValidationError(f"{field} must be a nonnegative number")
    return float(value)


def normalize_scenario(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Qwen or preset JSON without mutating the caller."""
    if not isinstance(raw, dict):
        raise ScenarioValidationError("scenario must be a JSON object")

    scenario = deepcopy(raw)
    _require(scenario, "scenario_id", "scenario")
    _require(scenario, "command_center", "scenario")
    _require(scenario, "config", "scenario")

    mode = scenario.pop("mode", scenario.get("run_mode", "fixed"))
    if mode not in {"fixed", "learned"}:
        raise ScenarioValidationError("run_mode must be 'fixed' or 'learned'")
    scenario["run_mode"] = mode

    scenario.setdefault("hospital", {})
    scenario.setdefault("nodes", {})
    scenario.setdefault("roads", [])
    scenario.setdefault("air_routes", [])
    scenario.setdefault("units", [])
    scenario.setdefault("events", [])
    zones = _require(scenario, "zones", "scenario")
    if not isinstance(zones, list) or not zones:
        raise ScenarioValidationError("zones must be a non-empty array")

    seen_zone_ids: set[str] = set()
    for index, zone in enumerate(zones):
        context = f"zones[{index}]"
        zone_id = _require(zone, "zone_id", context)
        _require(zone, "node_id", context)
        if zone_id in seen_zone_ids:
            raise ScenarioValidationError(f"duplicate zone_id '{zone_id}'")
        seen_zone_ids.add(zone_id)
        observations = _require(zone, "observations", context)
        for field in REQUIRED_OBSERVATIONS:
            observations[field] = _unit_interval(
                _require(observations, field, f"{context}.observations"),
                f"{context}.observations.{field}",
            )
        zone.setdefault("labels", {})
        if "hazard_intensity" in observations:
            observations["hazard_intensity"] = _unit_interval(
                observations["hazard_intensity"],
                f"{context}.observations.hazard_intensity",
            )
        if observations.get("drone_road_report") not in {
            None,
            "blocked",
            "uncertain",
            "open",
        }:
            raise ScenarioValidationError(
                f"{context}.observations.drone_road_report is invalid"
            )

    weights = _require(scenario["config"], "weights", "config")
    for group, fields in WEIGHT_FIELDS.items():
        group_weights = _require(weights, group, "config.weights")
        if group == "utility":
            group_weights.setdefault("zeta", 0.10)
        for field in fields:
            group_weights[field] = _nonnegative_number(
                _require(group_weights, field, f"config.weights.{group}"),
                f"config.weights.{group}.{field}",
            )
    scenario["config"].setdefault("thresholds", {})
    scenario["config"]["thresholds"].setdefault("car_min_passability", 0.45)
    scenario["config"]["thresholds"].setdefault("drone_recon_priority_risk", 0.70)

    for node_id, coordinates in scenario["nodes"].items():
        if not isinstance(coordinates, dict):
            raise ScenarioValidationError(f"nodes.{node_id} must be an object")
        for axis in ("x", "y"):
            value = _require(coordinates, axis, f"nodes.{node_id}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ScenarioValidationError(f"nodes.{node_id}.{axis} must be numeric")
            coordinates[axis] = float(value)

    seen_road_ids: set[str] = set()
    for collection_name in ("roads", "air_routes"):
        for index, road in enumerate(scenario[collection_name]):
            context = f"{collection_name}[{index}]"
            road_id = _require(road, "road_id", context)
            if road_id in seen_road_ids:
                raise ScenarioValidationError(f"duplicate road_id '{road_id}'")
            seen_road_ids.add(road_id)
            start_node = _require(road, "from", context)
            end_node = _require(road, "to", context)
            if scenario["nodes"] and (
                start_node not in scenario["nodes"]
                or end_node not in scenario["nodes"]
            ):
                raise ScenarioValidationError(
                    f"{context} references a node without coordinates"
                )
            road["distance"] = _positive_number(
                _require(road, "distance", context), f"{context}.distance"
            )
            road["travel_time_base"] = _positive_number(
                _require(road, "travel_time_base", context),
                f"{context}.travel_time_base",
            )
            status = road.setdefault("status", "open")
            if status not in {"open", "blocked"}:
                raise ScenarioValidationError(
                    f"{context}.status must be 'open' or 'blocked'"
                )
            road.setdefault("bidirectional", True)
            risk = _require(road, "risk", context)
            for field in ROAD_RISK_FIELDS:
                risk[field] = _unit_interval(
                    _require(risk, field, f"{context}.risk"),
                    f"{context}.risk.{field}",
                )
            road.setdefault("labels", {})

    seen_unit_ids: set[str] = set()
    for index, unit in enumerate(scenario["units"]):
        context = f"units[{index}]"
        unit_id = _require(unit, "unit_id", context)
        if unit_id in seen_unit_ids:
            raise ScenarioValidationError(f"duplicate unit_id '{unit_id}'")
        seen_unit_ids.add(unit_id)
        unit_type = _require(unit, "type", context)
        if unit_type not in {"rescue_car", "drone"}:
            raise ScenarioValidationError(f"{context}.type must be 'rescue_car' or 'drone'")
        _require(unit, "start_node", context)
        unit["speed"] = _positive_number(_require(unit, "speed", context), f"{context}.speed")
        if not isinstance(_require(unit, "can_transport", context), bool):
            raise ScenarioValidationError(f"{context}.can_transport must be boolean")
        constraints = unit.setdefault("constraints", {})
        for field in ("max_fire_risk", "min_passability"):
            if field in constraints:
                constraints[field] = _unit_interval(constraints[field], f"{context}.constraints.{field}")
        capacity = unit.setdefault("capacity", 4 if unit_type == "rescue_car" else 0)
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ScenarioValidationError(f"{context}.capacity must be a nonnegative integer")
        unit["service_time"] = _nonnegative_number(
            unit.setdefault("service_time", 1.0 if unit_type == "rescue_car" else 0.5),
            f"{context}.service_time",
        )
        unit["resource_cost"] = _unit_interval(
            unit.setdefault("resource_cost", 0.55 if unit_type == "rescue_car" else 0.25),
            f"{context}.resource_cost",
        )
        if scenario["nodes"] and unit["start_node"] not in scenario["nodes"]:
            raise ScenarioValidationError(f"{context}.start_node has no coordinates")

    seen_event_ids: set[str] = set()
    for index, event in enumerate(scenario["events"]):
        context = f"events[{index}]"
        event_id = _require(event, "event_id", context)
        if event_id in seen_event_ids:
            raise ScenarioValidationError(f"duplicate event_id '{event_id}'")
        seen_event_ids.add(event_id)
        event_type = _require(event, "event_type", context)
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise ScenarioValidationError(f"{context}.event_type is unsupported")
        _require(event, "target_id", context)
        changes = _require(event, "changes", context)
        if not isinstance(changes, dict) or not changes:
            raise ScenarioValidationError(f"{context}.changes must be a non-empty object")
        elapsed = event.setdefault("elapsed_minutes", 0.0)
        event["elapsed_minutes"] = _nonnegative_number(
            elapsed, f"{context}.elapsed_minutes"
        )

    try:
        validate_scenario(scenario)
    except ContractValidationError as error:
        raise ScenarioValidationError(f"scenario schema violation: {error}") from error
    return scenario
