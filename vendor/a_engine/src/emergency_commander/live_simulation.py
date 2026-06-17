from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.expert_cpts import build_expert_network
from emergency_commander.inference import assess_zones
from emergency_commander.input_adapter import normalize_scenario
from emergency_commander.pipeline import _invalidate_affected_missions, _public_plan
from emergency_commander.replanning import SUPPORTED_EVENT_TYPES, apply_event
from emergency_commander.simulation import (
    advance_unit_states,
    initialize_unit_states,
    start_assignments,
)


PHASES = (
    "validate",
    "infer",
    "prioritize",
    "route",
    "utility",
    "allocate",
    "execute",
    "replan",
    "complete",
)


@dataclass
class LiveSimulation:
    scenario: dict[str, Any]
    seed: int
    model_name: str = "expert_cpt"
    phase: str = "validate"
    status: str = "running"
    clock_minutes: float = 0.0
    step_count: int = 0
    execution_step_minutes: float = 1.0
    max_minutes: float = 120.0
    max_steps: int = 500
    unit_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    assessments: list[dict[str, Any]] = field(default_factory=list)
    utility_matrix: list[dict[str, Any]] = field(default_factory=list)
    current_plan: dict[str, Any] = field(default_factory=dict)
    initial_plan: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    algorithm_log: list[dict[str, Any]] = field(default_factory=list)
    calculation_history: list[dict[str, Any]] = field(default_factory=list)
    replan_log: list[dict[str, Any]] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    end_reason: str | None = None
    last_error: str | None = None
    replan_context: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        scenario: dict[str, Any],
        *,
        seed: int,
        model_name: str | None = None,
    ) -> "LiveSimulation":
        selected_model = model_name or (
            "learned_cpt" if scenario.get("run_mode") == "learned" else "expert_cpt"
        )
        return cls(
            scenario=deepcopy(scenario),
            seed=int(seed),
            model_name=selected_model,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveSimulation":
        return cls(**deepcopy(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def completed_zones(self) -> set[str]:
        return {
            zone_id
            for state in self.unit_states.values()
            for zone_id in state.get("delivered_targets", [])
        }

    def pause(self) -> None:
        if self.status == "running":
            self.status = "paused"

    def resume(self) -> None:
        if self.status == "paused":
            self.status = "running"

    def build_result(self) -> dict[str, Any]:
        completed = sorted(self.completed_zones())
        all_zones = [zone["zone_id"] for zone in self.scenario["zones"]]
        units = []
        for unit_id, state in sorted(self.unit_states.items()):
            units.append(
                {
                    "unit_id": unit_id,
                    "type": state["type"],
                    "final_status": state["status"],
                    "final_node": state["current_node"],
                    "completed_missions": state["completed_missions"],
                    "recon_targets": deepcopy(state.get("completed_targets", [])),
                    "delivered_targets": deepcopy(state.get("delivered_targets", [])),
                    "rescued_people": int(state.get("rescued_people", 0)),
                    "travel_minutes": round(float(state.get("travel_minutes", 0.0)), 6),
                }
            )
        return {
            "scenario_id": self.scenario["scenario_id"],
            "seed": self.seed,
            "model_name": self.model_name,
            "status": self.status,
            "end_reason": self.end_reason,
            "simulation_clock": round(self.clock_minutes, 6),
            "completed_zones": completed,
            "incomplete_zones": [zone for zone in all_zones if zone not in completed],
            "rescued_people": sum(
                int(state.get("rescued_people", 0))
                for state in self.unit_states.values()
            ),
            "units": units,
            "initial_plan": deepcopy(self.initial_plan),
            "final_plan": deepcopy(self.current_plan),
            "zone_assessment": deepcopy(self.assessments),
            "utility_matrix": deepcopy(self.utility_matrix),
            "events": deepcopy(self.event_log),
            "replan_log": deepcopy(self.replan_log),
            "timeline": deepcopy(self.timeline),
            "algorithm_log": deepcopy(self.algorithm_log),
            "calculation_history": deepcopy(self.calculation_history),
        }

    def _active_zones(self) -> set[str]:
        return {
            state["current_task"].get("target_zone")
            or state["current_task"].get("origin_zone")
            for state in self.unit_states.values()
            if state.get("current_task")
            and (
                state["current_task"].get("target_zone") is not None
                or state["current_task"].get("origin_zone") is not None
            )
        }

    def _has_idle_planning_work(self) -> bool:
        if not any(state["status"] == "idle" for state in self.unit_states.values()):
            return False
        completed = self.completed_zones()
        active = self._active_zones()
        for assessment in self.assessments:
            zone_id = assessment["zone_id"]
            if zone_id in completed or zone_id in active:
                continue
            for state in self.unit_states.values():
                if (
                    state["status"] == "idle"
                    and zone_id not in state.get("completed_targets", [])
                ):
                    return True
        return False

    def _planning_scenario(self) -> dict[str, Any]:
        planning = deepcopy(self.scenario)
        planning["units"] = []
        for unit in self.scenario["units"]:
            state = self.unit_states[unit["unit_id"]]
            if state["status"] != "idle":
                continue
            available = deepcopy(unit)
            available["start_node"] = state["current_node"]
            if state.get("_temporary_route_edges"):
                available["_temporary_route_edges"] = deepcopy(
                    state["_temporary_route_edges"]
                )
            planning["units"].append(available)
        return planning

    def _eligible_assessments(self) -> list[dict[str, Any]]:
        excluded = self.completed_zones() | self._active_zones()
        return [
            assessment
            for assessment in self.assessments
            if assessment["zone_id"] not in excluded
        ]

    def _filtered_matrix(self, matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in matrix
            if candidate["target_zone"]
            not in self.unit_states[candidate["unit_id"]].get("completed_targets", [])
        ]

    def _assign_idle_units_without_phase_loop(self) -> list[dict[str, Any]]:
        planning = self._planning_scenario()
        eligible = self._eligible_assessments()
        if not planning["units"] or not eligible:
            return []
        matrix = self._filtered_matrix(
            build_utility_matrix(planning, eligible, include_trace=True)
        )
        assignments = allocate_tasks(planning, matrix) if matrix else []
        self._add_estimated_people(assignments)
        start_assignments(self.unit_states, assignments, self.scenario)
        if assignments:
            self.utility_matrix = matrix
        return assignments

    def _build_drone_intel_event(self, unit_id: str, zone_id: str) -> dict[str, Any]:
        zone = next(item for item in self.scenario["zones"] if item["zone_id"] == zone_id)
        obs = zone["observations"]
        return {
            "event_id": f"AUTO_{len(self.event_log) + 1:03d}_DRONE_UPDATE",
            "event_type": "drone_update",
            "source": "automatic_drone_recon",
            "trigger_step": self.step_count,
            "elapsed_minutes": 0.0,
            "target_id": zone_id,
            "unit_id": unit_id,
            "changes": {
                "observations.drone_confidence": 1.0,
                "observations.road_damage": max(0.0, round(obs["road_damage"] - 0.20, 3)),
                "observations.congestion": max(0.0, round(obs["congestion"] - 0.12, 3)),
            },
            "description": f"{unit_id} 完成 {zone_id} 区侦察并自动回传道路情报",
        }

    def _apply_automatic_drone_intel(
        self, before_states: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        for unit_id, state in self.unit_states.items():
            if state["type"] != "drone":
                continue
            before_completed = set(before_states.get(unit_id, {}).get("completed_targets", []))
            new_targets = [
                zone_id
                for zone_id in state.get("completed_targets", [])
                if zone_id not in before_completed
            ]
            for zone_id in new_targets:
                zone = next(
                    item for item in self.scenario["zones"] if item["zone_id"] == zone_id
                )
                if zone["observations"].get("drone_confidence", 0.0) >= 1.0:
                    continue
                old_plan = deepcopy(self.current_plan)
                event = self._build_drone_intel_event(unit_id, zone_id)
                self.scenario = normalize_scenario(apply_event(self.scenario, event))
                _invalidate_affected_missions(self.unit_states, event, self.scenario)
                self.utility_matrix = []
                self.current_plan = _public_plan(
                    self.assessments, self.unit_states, self.utility_matrix
                )
                self.event_log.append(deepcopy(event))
                self.replan_context = {
                    "trigger_event": deepcopy(event),
                    "clock_minutes": round(self.clock_minutes, 6),
                    "old_plan": old_plan,
                }
                self.phase = "replan"
                self._log(
                    "replan",
                    "无人机自动情报",
                    f"{event['description']}，准备更新推理与任务分配",
                )
                self._record_calculation(
                    "replan",
                    "无人机情报自动回传",
                    event["description"],
                    focus={"zones": [zone_id], "units": [unit_id]},
                    inputs={"event": event, "old_plan": old_plan},
                    operations={
                        "source": "automatic_drone_recon",
                        "changes": event["changes"],
                    },
                    outputs={"next_phase": "replan"},
                )
                return event
        return None

    def _log(self, phase: str, title: str, summary: str) -> None:
        self.algorithm_log.append(
            {
                "index": len(self.algorithm_log) + 1,
                "phase": phase,
                "title": title,
                "summary": summary,
                "clock_minutes": round(self.clock_minutes, 6),
            }
        )

    def _record_calculation(
        self,
        phase: str,
        title: str,
        summary: str,
        *,
        focus: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        operations: dict[str, Any] | list[Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        self.calculation_history.append(
            {
                "index": len(self.calculation_history) + 1,
                "phase": phase,
                "title": title,
                "clock_minutes": round(self.clock_minutes, 6),
                "summary": summary,
                "focus": deepcopy(focus or {}),
                "inputs": deepcopy(inputs or {}),
                "operations": deepcopy(operations or {}),
                "outputs": deepcopy(outputs or {}),
            }
        )

    def next_transition_minutes(self) -> float | None:
        durations = []
        for state in self.unit_states.values():
            if state["status"] in {"en_route", "returning"}:
                duration = float(state.get("remaining_travel", 0.0))
            elif state["status"] == "rescuing":
                duration = float(state.get("remaining_service", 0.0))
            else:
                continue
            if duration > 1e-9:
                durations.append(duration)
        return min(durations) if durations else None

    def _snapshot(self, *, event: dict[str, Any] | None = None) -> None:
        self.timeline.append(
            {
                "step": len(self.timeline),
                "clock_minutes": round(self.clock_minutes, 6),
                "event": deepcopy(event),
                "phase": self.phase,
                "plan": deepcopy(self.current_plan),
                "unit_states": deepcopy(self.unit_states),
                "scenario_state": deepcopy(self.scenario),
            }
        )

    def _mark_complete(self, reason: str) -> None:
        self.status = "completed"
        self.phase = "complete"
        self.end_reason = reason
        self._log("complete", "仿真结束", reason)

    def _add_estimated_people(self, assignments: list[dict[str, Any]]) -> None:
        assessment_by_zone = {item["zone_id"]: item for item in self.assessments}
        units = {unit["unit_id"]: unit for unit in self.scenario["units"]}
        for assignment in assignments:
            unit = units[assignment["unit_id"]]
            trapped = assessment_by_zone[assignment["target_zone"]]["trapped_prob"]
            capacity = int(unit.get("capacity", 4 if unit["type"] == "rescue_car" else 0))
            assignment["estimated_people"] = (
                max(1, round(trapped * capacity))
                if assignment["mission_type"] == "rescue" and trapped >= 0.5
                else 0
            )

    def available_event_targets(self, event_type: str) -> list[str]:
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"unsupported event_type '{event_type}'")
        if event_type == "road_collapse":
            return [
                road["road_id"]
                for road in self.scenario["roads"]
                if road["status"] == "open"
            ]
        completed = self.completed_zones()
        return [
            zone["zone_id"]
            for zone in self.scenario["zones"]
            if zone["zone_id"] not in completed
        ]

    def select_event_target(self, event_type: str) -> str:
        targets = self.available_event_targets(event_type)
        if not targets:
            raise ValueError(f"no available target for '{event_type}'")
        if event_type == "road_collapse":
            open_targets = set(targets)
            for state in self.unit_states.values():
                task = state.get("current_task")
                if not task or task["route"].get("route_layer", "ground") != "ground":
                    continue
                for road_id in task["route"].get("road_ids", []):
                    if road_id in open_targets:
                        return road_id
            roads = {road["road_id"]: road for road in self.scenario["roads"]}
            return max(
                targets,
                key=lambda road_id: sum(roads[road_id]["risk"].values()),
            )
        zones = {zone["zone_id"]: zone for zone in self.scenario["zones"]}
        assessments = {item["zone_id"]: item for item in self.assessments}
        if event_type == "fire_spread":
            return max(
                targets,
                key=lambda zone_id: assessments.get(zone_id, {}).get(
                    "priority_score", zones[zone_id]["observations"]["time_urgency"]
                ),
            )
        if event_type == "new_sos":
            return min(
                targets,
                key=lambda zone_id: zones[zone_id]["observations"]["sos_signal"],
            )
        return min(
            targets,
            key=lambda zone_id: abs(
                assessments.get(zone_id, {}).get("trapped_prob", 0.5) - 0.5
            ),
        )

    def build_event(self, event_type: str, target_id: str) -> dict[str, Any]:
        if target_id not in self.available_event_targets(event_type):
            raise ValueError(f"event target '{target_id}' is not available")
        event_number = len(self.event_log) + 1
        event_id = f"LIVE_{event_number:03d}_{event_type.upper()}"
        if event_type == "road_collapse":
            changes = {"status": "blocked", "risk.damage": 1.0}
            description = f"道路 {target_id} 发生二次坍塌"
        else:
            zone = next(
                item for item in self.scenario["zones"] if item["zone_id"] == target_id
            )
            obs = zone["observations"]
            if event_type == "fire_spread":
                changes = {
                    "observations.fire": min(1.0, round(obs["fire"] + 0.30, 3)),
                    "observations.smoke": min(1.0, round(obs["smoke"] + 0.18, 3)),
                    "observations.time_urgency": min(
                        1.0, round(obs["time_urgency"] + 0.12, 3)
                    ),
                }
                description = f"{target_id} 区火势快速蔓延"
            elif event_type == "new_sos":
                changes = {
                    "observations.sos_signal": 1.0,
                    "observations.human_activity": min(
                        1.0, round(obs["human_activity"] + 0.22, 3)
                    ),
                    "observations.time_urgency": min(
                        1.0, round(obs["time_urgency"] + 0.15, 3)
                    ),
                }
                description = f"{target_id} 区收到新的高置信求救信号"
            else:
                changes = {
                    "observations.drone_confidence": 1.0,
                    "observations.road_damage": max(
                        0.0, round(obs["road_damage"] - 0.20, 3)
                    ),
                    "observations.congestion": max(
                        0.0, round(obs["congestion"] - 0.12, 3)
                    ),
                }
                description = f"无人机回传 {target_id} 区最新道路情报"
        return {
            "event_id": event_id,
            "event_type": event_type,
            "trigger_step": self.step_count,
            "elapsed_minutes": 0.0,
            "target_id": target_id,
            "changes": changes,
            "description": description,
        }

    def inject_event_payload(self, event: dict[str, Any]) -> None:
        if self.status in {"completed", "error"}:
            raise ValueError("cannot inject an event into a terminal session")
        if not self.initial_plan:
            raise ValueError("events are available after initial allocation")
        if any(item["event_id"] == event["event_id"] for item in self.event_log):
            raise ValueError(f"event '{event['event_id']}' was already applied")
        old_plan = deepcopy(self.current_plan)
        self.scenario = normalize_scenario(apply_event(self.scenario, event))
        _invalidate_affected_missions(self.unit_states, event, self.scenario)
        self.utility_matrix = []
        self.current_plan = _public_plan(
            self.assessments, self.unit_states, self.utility_matrix
        )
        self.event_log.append(deepcopy(event))
        self.replan_context = {
            "trigger_event": deepcopy(event),
            "clock_minutes": round(self.clock_minutes, 6),
            "old_plan": old_plan,
        }
        self.phase = "replan"
        self._log(
            "replan",
            "突发事件",
            f"{event['description']}，准备从当前状态重新规划",
        )
        focus_key = "roads" if event["event_type"] == "road_collapse" else "zones"
        self._record_calculation(
            "replan",
            "突发事件已应用",
            event["description"],
            focus={focus_key: [event["target_id"]]},
            inputs={"event": event, "old_plan": old_plan},
            operations={
                "changes": event["changes"],
                "invalidated_units": [
                    unit_id
                    for unit_id, state in self.unit_states.items()
                    if state["status"] == "idle" and state.get("current_task") is None
                ],
            },
            outputs={"next_phase": "replan"},
        )
        self._snapshot(event=event)

    def inject_event(
        self, event_type: str, target_id: str | None = None
    ) -> dict[str, Any]:
        selected = target_id or self.select_event_target(event_type)
        event = self.build_event(event_type, selected)
        self.inject_event_payload(event)
        return event

    def step(
        self,
        network: DiscreteBayesianNetwork | None = None,
        *,
        execution_minutes: float | None = None,
        to_next_transition: bool = False,
    ) -> None:
        if self.status != "running":
            return
        if self.step_count >= self.max_steps:
            self._mark_complete("timeout")
            return
        try:
            current_phase = self.phase
            if current_phase == "validate":
                raw_counts = {
                    "zones": len(self.scenario.get("zones", [])),
                    "nodes": len(self.scenario.get("nodes", {})),
                    "roads": len(self.scenario.get("roads", [])),
                    "air_routes": len(self.scenario.get("air_routes", [])),
                    "units": len(self.scenario.get("units", [])),
                }
                self.scenario = normalize_scenario(self.scenario)
                self.unit_states = initialize_unit_states(self.scenario)
                summary = "场景契约、单位和路网校验完成"
                self._log("validate", "输入校验", summary)
                self._record_calculation(
                    "validate",
                    "输入校验与归一化",
                    summary,
                    focus={"units": list(self.unit_states)},
                    inputs={"raw_counts": raw_counts},
                    operations={
                        "checks": [
                            "JSON Schema / contract",
                            "road endpoint references",
                            "unit start positions",
                            "probability and configuration defaults",
                        ],
                        "normalization": "normalize_scenario",
                    },
                    outputs={
                        "normalized_counts": raw_counts,
                        "unit_states": self.unit_states,
                    },
                )
                self.phase = "infer"
            elif current_phase == "infer":
                self.assessments = assess_zones(
                    self.scenario, network, model_name=self.model_name
                )
                expert_assessments = assess_zones(
                    self.scenario,
                    build_expert_network(),
                    model_name="expert_cpt",
                )
                expert_by_zone = {
                    item["zone_id"]: item for item in expert_assessments
                }
                summary = f"完成 {len(self.assessments)} 个区域的后验概率计算"
                self._log("infer", "贝叶斯推理", summary)
                zone_records = []
                comparison_rows = []
                for item in self.assessments:
                    expert = expert_by_zone[item["zone_id"]]
                    comparison_rows.append(
                        {
                            "zone_id": item["zone_id"],
                            "expert_trapped_prob": expert["trapped_prob"],
                            "active_trapped_prob": item["trapped_prob"],
                            "trapped_delta": round(
                                item["trapped_prob"] - expert["trapped_prob"], 6
                            ),
                            "expert_passability_prob": expert["passability_prob"],
                            "active_passability_prob": item["passability_prob"],
                            "passability_delta": round(
                                item["passability_prob"]
                                - expert["passability_prob"],
                                6,
                            ),
                            "expert_priority_score": expert["priority_score"],
                            "active_priority_score": item["priority_score"],
                            "priority_delta": round(
                                item["priority_score"] - expert["priority_score"],
                                6,
                            ),
                        }
                    )
                    zone_records.append(
                        {
                            "zone_id": item["zone_id"],
                            "evidence": item["bayesian_evidence"],
                            "trapped_distribution": {
                                "yes": item["trapped_prob"],
                                "no": round(1.0 - item["trapped_prob"], 6),
                            },
                            "passability_distribution": {
                                "yes": item["passability_prob"],
                                "no": round(1.0 - item["passability_prob"], 6),
                            },
                            "trapped_contributions": item["trapped_explanation"][
                                "contributions"
                            ],
                            "passability_contributions": item[
                                "passability_explanation"
                            ]["contributions"],
                        }
                    )
                self._record_calculation(
                    "infer",
                    "贝叶斯精确推理",
                    summary,
                    focus={"zones": [item["zone_id"] for item in self.assessments]},
                    inputs={"model": self.model_name, "zone_count": len(zone_records)},
                    operations={
                        "method": "exact enumeration",
                        "queries": ["trapped_people=yes", "road_passable=yes"],
                    },
                    outputs={"zones": zone_records},
                )
                self.calculation_history[-1]["outputs"][
                    "model_comparison"
                ] = {
                    "baseline_model": "expert_cpt",
                    "active_model": self.model_name,
                    "shared_downstream_weights": [
                        "life_risk",
                        "priority",
                        "utility",
                        "astar_risk",
                    ],
                    "zones": comparison_rows,
                    "max_abs_priority_delta": round(
                        max(
                            abs(row["priority_delta"])
                            for row in comparison_rows
                        )
                        if comparison_rows
                        else 0.0,
                        6,
                    ),
                    "max_abs_trapped_delta": round(
                        max(
                            abs(row["trapped_delta"])
                            for row in comparison_rows
                        )
                        if comparison_rows
                        else 0.0,
                        6,
                    ),
                }
                self.phase = "prioritize"
            elif current_phase == "prioritize":
                self.assessments.sort(
                    key=lambda item: item["priority_score"], reverse=True
                )
                top = self.assessments[0]
                summary = (
                    f"最高优先级为 {top['zone_id']} 区 "
                    f"({top['priority_score']:.1%})"
                )
                self._log("prioritize", "风险排序", summary)
                life_weights = self.scenario["config"]["weights"]["life_risk"]
                priority_weights = self.scenario["config"]["weights"]["priority"]
                zones = {zone["zone_id"]: zone for zone in self.scenario["zones"]}
                calculations = []
                for rank, assessment in enumerate(self.assessments, start=1):
                    observations = zones[assessment["zone_id"]]["observations"]
                    life_terms = {
                        "fire": round(life_weights["fire"] * observations["fire"], 6),
                        "trapped_prob": round(
                            life_weights["trapped_prob"] * assessment["trapped_prob"],
                            6,
                        ),
                        "time_urgency": round(
                            life_weights["time_urgency"]
                            * observations["time_urgency"],
                            6,
                        ),
                    }
                    priority_terms = {
                        "trapped_prob": round(
                            priority_weights["trapped_prob"]
                            * assessment["trapped_prob"],
                            6,
                        ),
                        "life_risk": round(
                            priority_weights["life_risk"] * assessment["life_risk"],
                            6,
                        ),
                        "time_urgency": round(
                            priority_weights["time_urgency"]
                            * observations["time_urgency"],
                            6,
                        ),
                        "accessibility": round(
                            priority_weights["accessibility"]
                            * assessment["passability_prob"],
                            6,
                        ),
                    }
                    calculations.append(
                        {
                            "zone_id": assessment["zone_id"],
                            "rank": rank,
                            "life_terms": life_terms,
                            "life_risk": assessment["life_risk"],
                            "priority_terms": priority_terms,
                            "priority_score": assessment["priority_score"],
                        }
                    )
                self._record_calculation(
                    "prioritize",
                    "风险优先级排序",
                    summary,
                    focus={"zones": [top["zone_id"]]},
                    inputs={
                        "life_weights": life_weights,
                        "priority_weights": priority_weights,
                    },
                    operations={
                        "life_formula": "Σ life_weight × evidence",
                        "priority_formula": "Σ priority_weight × posterior/risk/urgency/accessibility",
                    },
                    outputs={"ranking": calculations},
                )
                self.phase = "route"
            elif current_phase == "route":
                planning = self._planning_scenario()
                eligible = self._eligible_assessments()
                self.utility_matrix = (
                    self._filtered_matrix(
                        build_utility_matrix(planning, eligible, include_trace=True)
                    )
                    if planning["units"] and eligible
                    else []
                )
                feasible = sum(item["feasible"] for item in self.utility_matrix)
                summary = f"生成 {feasible} 条可行候选路线"
                self._log("route", "风险感知 A*", summary)
                candidates = [
                    {
                        "unit_id": item["unit_id"],
                        "target_zone": item["target_zone"],
                        "feasible": item["feasible"],
                        "reason": item["reason"],
                        "route": item.get("route"),
                    }
                    for item in self.utility_matrix
                ]
                route_candidates = [
                    item for item in self.utility_matrix if item.get("route")
                ]
                highlighted = route_candidates[0] if route_candidates else None
                self._record_calculation(
                    "route",
                    "风险感知 A* 搜索",
                    summary,
                    focus={
                        "roads": highlighted["route"]["road_ids"] if highlighted else [],
                        "zones": [highlighted["target_zone"]] if highlighted else [],
                    },
                    inputs={
                        "available_units": [unit["unit_id"] for unit in planning["units"]],
                        "eligible_zones": [item["zone_id"] for item in eligible],
                        "risk_weights": self.scenario["config"]["weights"]["astar_risk"],
                    },
                    operations={
                        "edge_cost": "travel_time / speed × (1 + weighted_road_risk)",
                        "heuristic": "euclidean time lower bound",
                    },
                    outputs={"candidates": candidates},
                )
                self.phase = "utility"
            elif current_phase == "utility":
                self.utility_matrix.sort(
                    key=lambda item: (
                        not item["feasible"],
                        -(item["expected_utility"] or -999.0),
                    )
                )
                feasible = [item for item in self.utility_matrix if item["feasible"]]
                best = feasible[0]["expected_utility"] if feasible else None
                summary = (
                    f"最佳候选期望效用 {best:.3f}"
                    if best is not None
                    else "当前没有可行候选"
                )
                self._log("utility", "期望效用", summary)
                self._record_calculation(
                    "utility",
                    "期望效用分解",
                    summary,
                    focus={
                        "zones": [
                            feasible[0]["target_zone"]
                        ]
                        if feasible
                        else []
                    },
                    inputs={
                        "weights": self.scenario["config"]["weights"]["utility"]
                    },
                    operations={
                        "formula": "benefits(trapped, life risk, accessibility) - costs(arrival, path risk, resource)",
                    },
                    outputs={
                        "candidates": [
                            {
                                "unit_id": item["unit_id"],
                                "target_zone": item["target_zone"],
                                "feasible": item["feasible"],
                                "reason": item["reason"],
                                "inputs": item.get("utility_inputs"),
                                "breakdown": item.get("utility_breakdown"),
                                "expected_utility": item.get("expected_utility"),
                            }
                            for item in self.utility_matrix
                        ]
                    },
                )
                self.phase = "allocate"
            elif current_phase == "allocate":
                planning = self._planning_scenario()
                allocation_result = (
                    allocate_tasks(planning, self.utility_matrix, include_trace=True)
                    if planning["units"] and self.utility_matrix
                    else {
                        "assignments": [],
                        "trace": {
                            "considered": 0,
                            "duplicate_zone_rejections": 0,
                            "ranked_combinations": [],
                            "winning_total": None,
                        },
                    }
                )
                assignments = allocation_result["assignments"]
                self._add_estimated_people(assignments)
                start_assignments(self.unit_states, assignments, self.scenario)
                self.current_plan = _public_plan(
                    self.assessments, self.unit_states, self.utility_matrix
                )
                if not self.initial_plan:
                    self.initial_plan = deepcopy(self.current_plan)
                summary = f"启动 {len(assignments)} 个任务"
                self._log("allocate", "全局任务分配", summary)
                self._record_calculation(
                    "allocate",
                    "全局组合分配",
                    summary,
                    focus={
                        "zones": [item["target_zone"] for item in assignments],
                        "units": [item["unit_id"] for item in assignments],
                        "roads": [
                            road_id
                            for item in assignments
                            for road_id in item["route"]["road_ids"]
                        ],
                    },
                    inputs={
                        "candidate_count": len(self.utility_matrix),
                        "unit_count": len(planning["units"]),
                    },
                    operations=allocation_result["trace"],
                    outputs={"assignments": assignments},
                )
                self._snapshot()
                if self.replan_context is not None:
                    self.replan_log.append(
                        {
                            **deepcopy(self.replan_context),
                            "new_plan": deepcopy(self.current_plan),
                            "reason": self.replan_context["trigger_event"].get(
                                "description",
                                self.replan_context["trigger_event"]["event_type"],
                            ),
                        }
                    )
                    self.replan_context = None
                if not assignments and not any(
                    state["status"] in {"en_route", "rescuing", "returning"}
                    for state in self.unit_states.values()
                ):
                    reason = (
                        "all_rescues_complete"
                        if len(self.completed_zones()) == len(self.scenario["zones"])
                        else "no_feasible_tasks"
                    )
                    self._mark_complete(reason)
                else:
                    self.phase = "execute"
            elif current_phase == "execute":
                if execution_minutes is not None and to_next_transition:
                    raise ValueError(
                        "execution_minutes and to_next_transition are mutually exclusive"
                    )
                elapsed = (
                    self.next_transition_minutes()
                    if to_next_transition
                    else (
                        float(execution_minutes)
                        if execution_minutes is not None
                        else self.execution_step_minutes
                    )
                )
                if elapsed is None or elapsed <= 0:
                    raise ValueError("no positive execution transition is available")
                before_states = deepcopy(self.unit_states)
                advance_unit_states(self.unit_states, elapsed, self.scenario)
                self.clock_minutes = round(self.clock_minutes + elapsed, 6)
                automatic_event = self._apply_automatic_drone_intel(before_states)
                if automatic_event is None and self._has_idle_planning_work():
                    self._assign_idle_units_without_phase_loop()
                self.current_plan = _public_plan(
                    self.assessments, self.unit_states, self.utility_matrix
                )
                summary = f"单位状态推进 {elapsed:.2f} 分钟至 T+{self.clock_minutes:.1f}"
                self._log("execute", "任务执行", summary)
                transitions = [
                    {
                        "unit_id": unit_id,
                        "from": before_states[unit_id]["status"],
                        "to": state["status"],
                    }
                    for unit_id, state in self.unit_states.items()
                    if before_states[unit_id]["status"] != state["status"]
                ]
                self._record_calculation(
                    "execute",
                    "单位状态推进",
                    summary,
                    focus={"units": list(self.unit_states)},
                    inputs={
                        "elapsed_minutes": round(elapsed, 6),
                        "mode": "next_transition" if to_next_transition else "fixed_delta",
                    },
                    operations={
                        "before": before_states,
                        "after": self.unit_states,
                        "transitions": transitions,
                    },
                    outputs={
                        "clock_minutes": self.clock_minutes,
                        "completed_zones": sorted(self.completed_zones()),
                    },
                )
                self._snapshot()
                if automatic_event is not None:
                    self._snapshot(event=automatic_event)
                elif len(self.completed_zones()) == len(self.scenario["zones"]):
                    self._mark_complete("all_rescues_complete")
                elif self.clock_minutes >= self.max_minutes:
                    self._mark_complete("timeout")
                elif not any(
                    state["status"] in {"en_route", "rescuing", "returning"}
                    for state in self.unit_states.values()
                ):
                    self.phase = "infer"
            elif current_phase == "replan":
                summary = "事件已应用，重新计算区域、路线和任务分配"
                self._log("replan", "动态重规划", summary)
                context = deepcopy(self.replan_context or {})
                self._record_calculation(
                    "replan",
                    "动态重规划启动",
                    summary,
                    focus={
                        "roads": [context.get("trigger_event", {}).get("target_id")]
                        if context.get("trigger_event", {}).get("event_type")
                        == "road_collapse"
                        else [],
                        "zones": [context.get("trigger_event", {}).get("target_id")]
                        if context.get("trigger_event", {}).get("event_type")
                        != "road_collapse"
                        and context.get("trigger_event")
                        else [],
                    },
                    inputs=context,
                    operations={"restart_from": "current unit positions and statuses"},
                    outputs={"next_phase": "infer"},
                )
                self.phase = "infer"
            elif current_phase == "complete":
                self._mark_complete(self.end_reason or "all_rescues_complete")
            else:
                raise ValueError(f"unknown phase '{current_phase}'")
            self.step_count += 1
        except Exception as exc:
            self.status = "error"
            self.last_error = f"{self.phase}: {exc}"
            raise
