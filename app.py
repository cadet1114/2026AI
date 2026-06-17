from __future__ import annotations

import html
import json
import secrets
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.live_simulation import LiveSimulation
from emergency_commander.random_scenario import generate_random_scenario
from emergency_commander.visualization import build_map_figure


ROOT = Path(__file__).resolve().parent
EVENTS = (
    ("road_collapse", "道路坍塌", "切断活动路线"),
    ("fire_spread", "火势蔓延", "提高区域风险"),
    ("new_sos", "新增求救", "注入高置信 SOS"),
)
COMMAND_STEPS = (
    ("validate", "01", "输入校验"),
    ("infer", "02", "贝叶斯推断"),
    ("prioritize", "03", "风险排序"),
    ("route", "04", "路线规划"),
    ("allocate", "05", "任务分配"),
    ("complete", "06", "报告生成"),
)
PHASE_TO_COMMAND_STEP = {
    "validate": "validate",
    "infer": "infer",
    "prioritize": "prioritize",
    "route": "route",
    "utility": "allocate",
    "allocate": "allocate",
    "execute": "complete",
    "replan": "route",
    "complete": "complete",
}
STEP_HINTS = {
    "validate": (
        "校验输入 JSON、道路端点和单位起点。",
        "场景节点、道路、灾区和救援单位。",
        "输出结构化校验结果。",
    ),
    "infer": (
        "用 CPT 对每个灾区进行贝叶斯后验推断。",
        "SOS、烟雾、火势、建筑损伤和道路损伤。",
        "输出被困概率与道路可通概率。",
    ),
    "prioritize": (
        "融合被困概率、生命风险、紧迫度和可达性。",
        "贝叶斯推理结果与灾区观测特征。",
        "输出灾区优先级排序。",
    ),
    "route": (
        "按道路风险和通行状态搜索候选路线。",
        "路网、阻断道路、火势风险和单位当前位置。",
        "输出可行路线、总代价与风险。",
    ),
    "utility": (
        "计算候选任务的期望效用。",
        "路线时间、目标优先级、风险和单位能力。",
        "输出效用矩阵。",
    ),
    "allocate": (
        "枚举组合并选择总效用最高的任务分配。",
        "候选任务、单位容量和重复目标约束。",
        "输出最终任务分配。",
    ),
    "execute": (
        "推进救援单位状态机。",
        "当前任务、剩余路程和服务时间。",
        "输出单位状态与救援进度。",
    ),
    "replan": (
        "根据突发事件重新计算风险、路线和任务。",
        "事件变化、旧计划和当前单位位置。",
        "输出重规划后的任务与路线。",
    ),
    "complete": (
        "汇总救援结果并生成可下载报告。",
        "全流程日志、最终单位状态和完成区域。",
        "输出 JSON 与 Markdown 报告。",
    ),
}
MISSION_LABELS = {
    "ground_rescue": "地面救援",
    "air_recon": "无人机侦查",
    "evacuation": "转运撤离",
    "medical_transfer": "医疗转运",
}
NODE_LABELS = {
    "HQ": "指挥中心",
    "HOSPITAL": "医院",
    "AIR_RELAY": "空中中继",
}
VALIDATION_CHECK_LABELS = {
    "JSON Schema / contract": "JSON 结构与字段契约",
    "road endpoint references": "道路端点引用",
    "unit start references": "救援单位起点引用",
    "event target availability": "事件目标可用性",
}
STATUS_LABELS = {
    "ready": "待命",
    "idle": "待命",
    "en_route": "前往灾区",
    "rescuing": "现场救援",
    "returning": "返程转运",
    "completed": "完成",
    "running": "运行中",
    "paused": "暂停",
    "error": "异常",
    "stranded": "受阻",
}
EVENT_LOCKED_LABEL = "待解锁"
PHASE_LABELS = {
    "validate": ("01", "输入校验", "JSON 结构校验 + 归一化"),
    "infer": ("02", "贝叶斯推理", "精确枚举后验概率"),
    "prioritize": ("03", "风险排序", "加权生命风险与优先级"),
    "route": ("04", "候选路线", "风险感知 A*"),
    "utility": ("05", "效用计算", "六项期望效用分解"),
    "allocate": ("06", "全局分配", "组合枚举最大总效用"),
    "execute": ("07", "任务执行", "多单位有限状态机"),
    "replan": ("08", "动态重规划", "事件驱动增量规划"),
    "complete": ("09", "结果汇总", "救援交付与审计"),
}


def _display_zone_id(value: Any) -> str:
    text = str(value)
    if text.startswith("ZONE_"):
        return f"{text.removeprefix('ZONE_')} 区"
    if len(text) == 1 and text.isalpha():
        return f"{text} 区"
    return text


def _display_unit_id(value: Any) -> str:
    text = str(value)
    return text.replace("RescueCar-", "救援车").replace("Drone-", "无人机")


def _display_node_id(value: Any) -> str:
    text = str(value)
    if text.startswith("ZONE_"):
        return _display_zone_id(text)
    return NODE_LABELS.get(text, text)


def _display_route_path(path: list[str]) -> str:
    return " → ".join(_display_node_id(node) for node in path)


def _current_command_step(session: LiveSimulation | None) -> str | None:
    if session is None:
        return None
    return PHASE_TO_COMMAND_STEP.get(session.phase, session.phase)


def _completed_command_steps(session: LiveSimulation | None) -> set[str]:
    if session is None:
        return set()
    completed = {
        PHASE_TO_COMMAND_STEP.get(record["phase"], record["phase"])
        for record in session.calculation_history
    }
    if session.status == "completed":
        completed.add("complete")
    return completed


def _render_progress_bar(session: LiveSimulation | None) -> str:
    current = _current_command_step(session)
    completed = _completed_command_steps(session)
    items = []
    for step_key, number, label in COMMAND_STEPS:
        if step_key in completed and step_key != current:
            state = "done"
            icon = "✓"
        elif step_key == current:
            state = "active"
            icon = "●"
        else:
            state = "todo"
            icon = "○"
        items.append(
            "<div class='progress-step progress-{}'>"
            "<span>{}</span><b>{}</b><small>{}</small></div>".format(
                state,
                icon,
                html.escape(number),
                html.escape(label),
            )
        )
    return "<div class='progress-rail'>" + "".join(items) + "</div>"


def _render_status_capsules(session: LiveSimulation | None, selected_model: str) -> str:
    if session is None:
        capsules = [
            ("等待场景", "standby"),
            ("模型 " + selected_model, "neutral"),
            ("时间 0.0 分钟", "neutral"),
        ]
    else:
        capsules = [
            (f"种子 {session.seed}", "neutral"),
            (f"时间 +{session.clock_minutes:.1f} 分钟", "neutral"),
            (f"模式 {selected_model}", "ok"),
        ]
    return "<div class='status-capsules'>" + "".join(
        f"<span class='status-pill status-{kind}'>{html.escape(label)}</span>"
        for label, kind in capsules
    ) + "</div>"


def _render_map_legend() -> str:
    items = [
        ("#d7e1b2", "地面"),
        ("#6f7c86", "道路"),
        ("#a7b2bc", "建筑"),
        ("#f2c85b", "拥堵"),
        ("#f27f2d", "火势"),
        ("#1c2024", "阻断"),
        ("#21b7ce", "无人机航线"),
        ("#e4583e", "救援路线"),
        ("#6d57a3", "医院"),
        ("#c91525", "灾区目标"),
    ]
    return "<div class='map-legend'>" + "".join(
        f"<span><i style='background:{color}'></i>{label}</span>"
        for color, label in items
    ) + "</div>"


def _phase_state(session: LiveSimulation | None, phase: str) -> str:
    if session is None:
        return "todo"
    seen = {record["phase"] for record in session.calculation_history}
    if phase == session.phase:
        return "active"
    if phase in seen or session.status == "completed":
        return "done"
    return "todo"


def _render_step_stack(session: LiveSimulation | None) -> str:
    phases = ("validate", "infer", "prioritize", "route", "utility", "allocate")
    cards = []
    for phase in phases:
        state = _phase_state(session, phase)
        icon = {"done": "✓", "active": "●", "todo": "○"}[state]
        number, label, algorithm = PHASE_LABELS[phase]
        cards.append(
            "<div class='step-card step-{}'><span>{}</span><div><b>{} {}</b>"
            "<small>{}</small></div></div>".format(
                state,
                icon,
                html.escape(number),
                html.escape(label),
                html.escape(algorithm),
            )
        )
    return "<div class='step-stack'>" + "".join(cards) + "</div>"


def _render_algorithm_hint(session: LiveSimulation | None, record: dict[str, Any] | None) -> str:
    phase = record["phase"] if record else (session.phase if session else "validate")
    description, input_summary, output_summary = STEP_HINTS.get(
        phase, ("等待算法步骤。", "暂无输入。", "等待输出。")
    )
    return (
        "<div class='algorithm-hint'>"
        f"<b>当前算法说明</b><p>{html.escape(description)}</p>"
        f"<b>输入数据摘要</b><p>{html.escape(input_summary)}</p>"
        f"<b>下一步输出</b><p>{html.escape(output_summary)}</p>"
        "</div>"
    )


def _event_status(session: LiveSimulation | None, event_type: str, enabled: bool) -> tuple[str, str]:
    if not enabled or session is None:
        return "待解锁", "locked"
    if any(event.get("event_type") == event_type for event in session.event_log):
        return "已触发", "danger"
    return "可触发", "ready"


def _unit_progress(state: dict[str, Any]) -> int:
    task = state.get("current_task") or {}
    route = task.get("route") or {}
    remaining = float(state.get("remaining_travel") or state.get("remaining_service") or 0.0)
    total = float(route.get("eta") or remaining or 1.0)
    if state.get("status") in {"idle", "ready"}:
        return 0
    if state.get("status") == "completed":
        return 100
    return max(0, min(100, round((1.0 - remaining / max(total, remaining, 1.0)) * 100)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_learned_network() -> DiscreteBayesianNetwork:
    payload = read_json(
        ROOT / "artifacts" / "full_bayesian_experiment" / "learned_network.json"
    )
    return DiscreteBayesianNetwork.from_dict(payload)


def _network_for(session: LiveSimulation) -> DiscreteBayesianNetwork | None:
    return load_learned_network() if session.model_name == "learned_cpt" else None


@st.cache_data
def learned_advantage_metrics() -> dict[str, float]:
    aggregate = read_json(
        ROOT / "artifacts" / "full_bayesian_experiment" / "experiment_metrics.json"
    )["aggregate"]
    expert_trapped = aggregate["expert_cpt"]["trapped_people"]
    learned_trapped = aggregate["learned_cpt"]["trapped_people"]
    expert_road = aggregate["expert_cpt"]["road_passable"]
    learned_road = aggregate["learned_cpt"]["road_passable"]
    return {
        "trapped_f1_delta": learned_trapped["f1"] - expert_trapped["f1"],
        "trapped_accuracy_delta": learned_trapped["accuracy"] - expert_trapped["accuracy"],
        "road_auc_delta": learned_road["roc_auc"] - expert_road["roc_auc"],
        "learned_trapped_f1": learned_trapped["f1"],
        "expert_trapped_f1": expert_trapped["f1"],
    }


def _render_learned_advantage_panel() -> None:
    metrics = learned_advantage_metrics()
    st.markdown(
        "<div class='learned-advantage'><b>学习 CPT 优势</b>"
        f"<span>被困 F1 {metrics['learned_trapped_f1']:.3f} vs "
        f"{metrics['expert_trapped_f1']:.3f}</span>"
        f"<small>被困 F1 +{metrics['trapped_f1_delta']:.3f} · "
        f"准确率 +{metrics['trapped_accuracy_delta']:.3f} · "
        f"道路 ROC-AUC +{metrics['road_auc_delta']:.3f}</small></div>",
        unsafe_allow_html=True,
    )


def _load_session() -> LiveSimulation | None:
    payload = st.session_state.get("live_simulation")
    return LiveSimulation.from_dict(payload) if payload else None


def _save_session(session: LiveSimulation) -> None:
    st.session_state["live_simulation"] = session.to_dict()
    st.session_state["history_index"] = len(session.calculation_history) - 1


def _event_target_key(event_type: str) -> str:
    return f"event_target_{event_type}"


def start_random_session() -> None:
    seed = secrets.randbelow(900_000_000) + 100_000_000
    learned = st.session_state.get("model_selector") == "学习 CPT"
    session = LiveSimulation.create(
        generate_random_scenario(seed, mode="learned" if learned else "fixed"),
        seed=seed,
        model_name="learned_cpt" if learned else "expert_cpt",
    )
    _save_session(session)
    st.session_state["event_notice"] = f"地图模型已导入 · 种子 {seed}"


def advance_phase() -> None:
    session = _load_session()
    if session is None or session.status != "running":
        return
    session.step(network=_network_for(session))
    _save_session(session)


def advance_execution(*, to_transition: bool = False) -> None:
    session = _load_session()
    if session is None or session.phase != "execute" or session.status != "running":
        return
    session.step(
        network=_network_for(session),
        execution_minutes=None if to_transition else 1.0,
        to_next_transition=to_transition,
    )
    _save_session(session)


def inject_event(event_type: str) -> None:
    session = _load_session()
    if session is None or not session.initial_plan or session.status != "running":
        return
    targets = session.available_event_targets(event_type)
    selected = st.session_state.get(_event_target_key(event_type))
    target_id = selected if selected in targets else None
    event = session.inject_event(event_type, target_id=target_id)
    _save_session(session)
    st.session_state["event_notice"] = event["description"]


def move_history(delta: int) -> None:
    session = _load_session()
    if session is None or not session.calculation_history:
        return
    current = int(st.session_state.get("history_index", len(session.calculation_history) - 1))
    st.session_state["history_index"] = max(
        0, min(len(session.calculation_history) - 1, current + delta)
    )


def result_markdown(result: dict[str, Any]) -> str:
    completed = ", ".join(_display_zone_id(zone) for zone in result["completed_zones"]) or "无"
    incomplete = ", ".join(_display_zone_id(zone) for zone in result["incomplete_zones"]) or "无"
    rows = [
        "# AI Emergency Commander 仿真结果",
        "",
        f"- 场景：`{result['scenario_id']}`",
        f"- 随机种子：`{result['seed']}`",
        f"- 结束原因：`{result['end_reason']}`",
        f"- 仿真时间：`{result['simulation_clock']:.1f}` 分钟",
        f"- 已完成区域：{completed}",
        f"- 未完成区域：{incomplete}",
        f"- 估计救援人数：`{result['rescued_people']}`",
        "",
        "| 单位 | 最终状态 | 完成任务 | 救援人数 | 行驶分钟 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    rows.extend(
        f"| {_display_unit_id(unit['unit_id'])} | {STATUS_LABELS.get(unit['final_status'], unit['final_status'])} | "
        f"{unit['completed_missions']} | {unit['rescued_people']} | "
        f"{unit['travel_minutes']:.1f} |"
        for unit in result["units"]
    )
    return "\n".join(rows)


def _current_snapshot(session: LiveSimulation) -> dict[str, Any]:
    if session.timeline:
        snapshot = dict(session.timeline[-1])
        snapshot["plan"] = session.current_plan or snapshot.get("plan", {})
        snapshot["unit_states"] = session.unit_states
        snapshot["scenario_state"] = session.scenario
        return snapshot
    return {
        "clock_minutes": session.clock_minutes,
        "event": None,
        "phase": session.phase,
        "plan": session.current_plan
        or {
            "zone_assessment": session.assessments,
            "assignments": [],
            "routes": [],
            "utility_matrix": session.utility_matrix,
        },
        "unit_states": session.unit_states,
        "scenario_state": session.scenario,
    }


def _selected_record(session: LiveSimulation) -> dict[str, Any] | None:
    if not session.calculation_history:
        return None
    index = int(st.session_state.get("history_index", len(session.calculation_history) - 1))
    index = max(0, min(len(session.calculation_history) - 1, index))
    st.session_state["history_index"] = index
    return session.calculation_history[index]


def _compact_frame(rows: list[dict[str, Any]], *, height: int = 190) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", height=height, hide_index=True)


def _render_validation(record: dict[str, Any]) -> None:
    counts = record["outputs"].get("normalized_counts", {})
    st.markdown(
        "".join(
            f"<span class='count-chip'><b>{value}</b>{html.escape(key)}</span>"
            for key, value in counts.items()
        ),
        unsafe_allow_html=True,
    )
    for check in record["operations"].get("checks", []):
        label = VALIDATION_CHECK_LABELS.get(check, check)
        st.markdown(f"<div class='check-line'>通过&nbsp;&nbsp;{html.escape(label)}</div>", unsafe_allow_html=True)


def _render_inference(record: dict[str, Any]) -> None:
    zones = record["outputs"].get("zones", [])
    if not zones:
        return
    comparison = record["outputs"].get("model_comparison", {})
    if comparison:
        active_model = comparison.get("active_model", "unknown")
        baseline_model = comparison.get("baseline_model", "expert_cpt")
        max_priority_delta = comparison.get("max_abs_priority_delta", 0.0)
        model_note = (
            "学习 CPT 正在替换贝叶斯条件概率表；优先级/效用权重保持相同，"
            "因此差异来自后验概率。"
            if active_model == "learned_cpt"
            else "当前使用固定专家 CPT；下表作为专家基线自检，delta 应接近 0。"
        )
        st.markdown(
            f"<div class='model-compare'><b>{html.escape(active_model.upper())}</b>"
            f"<span>vs {html.escape(baseline_model.upper())}</span>"
            f"<em>最大优先级变化 = {max_priority_delta:.3f}</em>"
            f"<small>{html.escape(model_note)}</small></div>",
            unsafe_allow_html=True,
        )
    zone_options = {zone["zone_id"]: _display_zone_id(zone["zone_id"]) for zone in zones}
    selected = st.selectbox(
        "查看区域",
        [zone["zone_id"] for zone in zones],
        format_func=zone_options.get,
        key=f"inference_zone_{record['index']}",
        label_visibility="collapsed",
    )
    zone = next(item for item in zones if item["zone_id"] == selected)
    trapped = zone["trapped_distribution"]["yes"]
    passable = zone["passability_distribution"]["yes"]
    st.markdown(
        f"<div class='formula-box'><b>P(被困 | 证据) = {trapped:.3f}</b>"
        f"<br><b>P(道路可通 | 证据) = {passable:.3f}</b>"
        "<small>精确枚举 · 删除单项证据后的后验差值用于解释贡献</small></div>",
        unsafe_allow_html=True,
    )
    evidence_rows = [
        {"证据节点": name, "离散状态": state}
        for name, state in zone["evidence"].items()
    ]
    _compact_frame(evidence_rows, height=156)
    contribution_rows = [
        {
            "证据": item["evidence"],
            "状态": item["state"],
            "后验变化": item["delta"],
        }
        for item in zone["trapped_contributions"][:5]
    ]
    _compact_frame(contribution_rows, height=150)
    if comparison:
        _compact_frame(
            [
                {
                    "区域": _display_zone_id(row["zone_id"]),
                    "Δ被困": row["trapped_delta"],
                    "Δ通行": row["passability_delta"],
                    "Δ优先级": row["priority_delta"],
                }
                for row in comparison.get("zones", [])
            ],
            height=165,
        )


def _render_priority(record: dict[str, Any]) -> None:
    ranking = record["outputs"].get("ranking", [])
    rows = [
        {
            "排名": item["rank"],
            "区域": _display_zone_id(item["zone_id"]),
            "被困项": item["priority_terms"]["trapped_prob"],
            "生命项": item["priority_terms"]["life_risk"],
            "紧迫项": item["priority_terms"]["time_urgency"],
            "通行项": item["priority_terms"]["accessibility"],
            "总分": item["priority_score"],
        }
        for item in ranking
    ]
    st.markdown(
        "<div class='formula-box'>优先级 = 0.40×P(被困) + 0.30×生命风险 + "
        "0.20×紧迫度 + 0.10×可通行率</div>",
        unsafe_allow_html=True,
    )
    _compact_frame(rows, height=285)


def _render_route(record: dict[str, Any]) -> None:
    candidates = [
        item for item in record["outputs"].get("candidates", []) if item.get("route")
    ]
    if not candidates:
        st.warning("当前没有可行路线。")
        return
    labels = [
        f"{_display_unit_id(item['unit_id'])} → {_display_zone_id(item['target_zone'])}"
        for item in candidates
    ]
    label = st.selectbox(
        "候选路线",
        labels,
        key=f"route_candidate_{record['index']}",
        label_visibility="collapsed",
    )
    candidate = candidates[labels.index(label)]
    route = candidate["route"]
    st.markdown(
        f"<div class='formula-box'><b>{_display_route_path(route['path'])}</b>"
        f"<br>预计 {route['eta']:.2f} 分钟 · 路径风险 {route['path_risk']:.3f} · "
        f"总代价 {route['total_cost']:.3f}<small>A* 评价函数：f(n) = g(n) + h(n)</small></div>",
        unsafe_allow_html=True,
    )
    trace_rows = [
        {
            "节点": _display_node_id(item["node"]),
            "g": item["g"],
            "h": item["h"],
            "f": item["f"],
            "前沿": item["frontier_size"],
            "松弛": len(item["relaxations"]),
        }
        for item in route.get("search_trace", [])
    ]
    _compact_frame(trace_rows, height=310)


def _render_utility(record: dict[str, Any]) -> None:
    candidates = [
        item
        for item in record["outputs"].get("candidates", [])
        if item.get("expected_utility") is not None
    ]
    if not candidates:
        st.warning("当前没有可计算效用的候选。")
        return
    labels = [
        f"{_display_unit_id(item['unit_id'])} → {_display_zone_id(item['target_zone'])}"
        for item in candidates
    ]
    label = st.selectbox(
        "效用候选",
        labels,
        key=f"utility_candidate_{record['index']}",
        label_visibility="collapsed",
    )
    candidate = candidates[labels.index(label)]
    rows = [
        {"贡献项": name, "加权值": value}
        for name, value in (candidate.get("breakdown") or {}).items()
    ]
    st.markdown(
        f"<div class='formula-box'><b>EU = {candidate['expected_utility']:.4f}</b>"
        "<small>正值为救援收益，负值为时间、风险与资源成本</small></div>",
        unsafe_allow_html=True,
    )
    _compact_frame(rows, height=280)


def _render_allocation(record: dict[str, Any]) -> None:
    trace = record["operations"]
    st.markdown(
        f"<div class='formula-box'><b>{trace.get('considered', 0):,}</b> 个组合被检查 · "
        f"<b>{trace.get('duplicate_zone_rejections', 0):,}</b> 个重复区域组合被剔除"
        f"<small>最优总效用 = {trace.get('winning_total')}</small></div>",
        unsafe_allow_html=True,
    )
    assignments = [
        {
            "单位": _display_unit_id(item["unit_id"]),
            "区域": _display_zone_id(item["target_zone"]),
            "任务": MISSION_LABELS.get(item["mission_type"], item["mission_type"]),
            "效用": item["expected_utility"],
            "预计时间": item["route"]["eta"],
        }
        for item in record["outputs"].get("assignments", [])
    ]
    _compact_frame(assignments, height=230)
    ranked = trace.get("ranked_combinations", [])[:5]
    _compact_frame(
        [
            {
                "候选组合": " / ".join(
                    f"{_display_unit_id(item['unit_id'])}→{_display_zone_id(item['target_zone'])}"
                    for item in row["assignments"]
                ),
                "总效用": row["total"],
            }
            for row in ranked
        ],
        height=180,
    )


def _render_execution(record: dict[str, Any]) -> None:
    st.markdown(
        f"<div class='formula-box'><b>时间步长 = {record['inputs']['elapsed_minutes']:.3f} 分钟</b>"
        f"<small>{record['inputs']['mode']} · 单次状态机推进</small></div>",
        unsafe_allow_html=True,
    )
    transitions = record["operations"].get("transitions", [])
    if transitions:
        _compact_frame(
            [
                {
                    "单位": _display_unit_id(row["unit_id"]),
                    "原状态": STATUS_LABELS.get(row["from"], row["from"]),
                    "新状态": STATUS_LABELS.get(row["to"], row["to"]),
                }
                for row in transitions
            ],
            height=180,
        )
    states = record["operations"].get("after", {})
    _compact_frame(
        [
            {
                "单位": _display_unit_id(unit_id),
                "状态": STATUS_LABELS.get(state["status"], state["status"]),
                "剩余行程": state["remaining_travel"],
                "剩余服务": state["remaining_service"],
                "载员": state["onboard"],
            }
            for unit_id, state in states.items()
        ],
        height=280,
    )


def _render_replan(record: dict[str, Any]) -> None:
    event = record["inputs"].get("event") or record["inputs"].get("trigger_event")
    if event:
        st.markdown(
            f"<div class='alert-box'><b>{html.escape(event['description'])}</b>"
            f"<small>目标 {html.escape(_display_node_id(event['target_id']))}</small></div>",
            unsafe_allow_html=True,
        )
        _compact_frame(
            [
                {"改变字段": key, "新值": json.dumps(value, ensure_ascii=False)}
                for key, value in event["changes"].items()
            ],
            height=220,
        )
    else:
        st.markdown("<div class='formula-box'>从当前单位位置重新开始推理与规划。</div>", unsafe_allow_html=True)


def _render_calculation_inspector(session: LiveSimulation) -> None:
    record = _selected_record(session)
    st.markdown(_render_step_stack(session), unsafe_allow_html=True)
    st.markdown(_render_algorithm_hint(session, record), unsafe_allow_html=True)
    output_label = PHASE_LABELS.get(record["phase"], ("--", "当前输出", ""))[1] if record else "等待输入校验"
    st.markdown(
        f"<div class='output-title'><b>当前输出 / Current Output</b>"
        f"<span>{html.escape(output_label)}</span></div>",
        unsafe_allow_html=True,
    )
    if record is None:
        st.markdown(
            "<div class='empty-output'><b>等待输入校验</b>"
            "<span>点击“执行下一算法步骤”后，将在这里显示 JSON 校验结果。</span></div>",
            unsafe_allow_html=True,
        )
        return

    index = int(st.session_state.get("history_index", len(session.calculation_history) - 1))
    number, label, algorithm = PHASE_LABELS.get(record["phase"], ("--", record["phase"], ""))
    st.markdown(
        f"<div class='inspector-head'><span>{number}</span><div><small>计算过程 "
        f"{index + 1}/{len(session.calculation_history)}</small><b>{html.escape(record['title'])}</b>"
        f"<em>{html.escape(algorithm)}</em></div></div>",
        unsafe_allow_html=True,
    )
    st.caption(record["summary"])

    renderer = {
        "validate": _render_validation,
        "infer": _render_inference,
        "prioritize": _render_priority,
        "route": _render_route,
        "utility": _render_utility,
        "allocate": _render_allocation,
        "execute": _render_execution,
        "replan": _render_replan,
    }.get(record["phase"])
    if renderer:
        renderer(record)

    if session.status == "completed":
        result = session.build_result()
        st.success(
            f"任务结束：{len(result['completed_zones'])}/{len(session.scenario['zones'])} 个区域完成"
        )
        d1, d2 = st.columns(2)
        d1.download_button(
            "结果 JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"live_result_{session.seed}.json",
            mime="application/json",
            width="stretch",
        )
        d2.download_button(
            "结果 Markdown",
            result_markdown(result),
            file_name=f"live_result_{session.seed}.md",
            mime="text/markdown",
            width="stretch",
        )


def _render_event_dock(session: LiveSimulation | None) -> None:
    st.markdown("<div class='dock-label'>现场事件</div>", unsafe_allow_html=True)
    enabled = bool(
        session
        and session.initial_plan
        and session.status == "running"
        and session.available_event_targets("fire_spread")
    )
    for event_type, label, hint in EVENTS:
        targets = session.available_event_targets(event_type) if enabled and session else []
        default_target = session.select_event_target(event_type) if targets and session else EVENT_LOCKED_LABEL
        status_label, status_kind = _event_status(session, event_type, enabled)
        with st.container(border=True):
            st.markdown(
                f"<div class='event-card-head'><b>{html.escape(label)}</b>"
                f"<span class='event-status {status_kind}'>{html.escape(status_label)}</span></div>",
                unsafe_allow_html=True,
            )
            selected = st.selectbox(
                f"{label}目标",
                targets or [EVENT_LOCKED_LABEL],
                index=(targets.index(default_target) if targets and default_target in targets else 0),
                key=_event_target_key(event_type),
                disabled=not targets,
                label_visibility="collapsed",
            )
            st.button(
                label,
                key=f"event_{event_type}",
                disabled=not targets,
                width="stretch",
                on_click=inject_event,
                args=(event_type,),
            )
            st.markdown(
                f"<div class='event-meta'><span>说明：{html.escape(hint)}</span>"
                f"<b>目标：{html.escape(_display_node_id(selected))}</b>"
                "<em>预计影响：触发后重新计算风险、路线和任务。</em></div>",
                unsafe_allow_html=True,
            )
    with st.container(border=True):
        if session and session.current_plan:
            status_label, status_kind = "已处理", "ready"
            target = f"{len(session.current_plan.get('assignments', []))} 个任务"
        elif session:
            status_label, status_kind = "等待规划", "locked"
            target = "等待任务分配"
        else:
            status_label, status_kind = "待解锁", "locked"
            target = "等待导入地图模型"
        st.markdown(
            f"<div class='event-card-head'><b>资源调度</b>"
            f"<span class='event-status {status_kind}'>{status_label}</span></div>",
            unsafe_allow_html=True,
        )
        st.button("随算法自动更新", key="event_resource_dispatch", disabled=True, width="stretch")
        st.markdown(
            f"<div class='event-meta'><span>说明：任务分配完成后自动刷新救援单位状态</span>"
            f"<b>当前：{html.escape(target)}</b>"
            "<em>预计影响：更新底部资源卡与执行进度。</em></div>",
            unsafe_allow_html=True,
        )


def _render_footer(session: LiveSimulation) -> None:
    states = session.unit_states
    if not states:
        states = {
            unit["unit_id"]: {
                "status": "ready",
                "remaining_travel": 0.0,
                "onboard": 0,
                "capacity": unit.get("capacity", 0),
                "current_task": None,
            }
            for unit in session.scenario["units"]
        }
    cards = []
    for unit_id, state in states.items():
        task = state.get("current_task") or {}
        target = task.get("target_zone") or task.get("origin_zone") or "--"
        status = str(state["status"])
        status_label = STATUS_LABELS.get(status, status)
        progress = _unit_progress(state)
        payload_label = "载荷" if state.get("type") == "drone" else "载员"
        cards.append(
            "<div class='unit-card'>"
            f"<i class='state-{html.escape(status)}'></i>"
            f"<b>{html.escape(_display_unit_id(unit_id))}</b><span>{html.escape(status_label)}</span>"
            f"<small>目标：{html.escape(_display_zone_id(target))}<br>"
            f"剩余：{state.get('remaining_travel', 0):.1f} 分钟<br>"
            f"{payload_label}：{state.get('onboard', 0)}/{state.get('capacity', 0)}</small>"
            f"<div class='unit-progress'><i style='--progress:{progress}%'></i></div></div>"
        )
    st.markdown(
        "<div class='footer-strip'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="AI Emergency Commander",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root { --shell:#0e1720; --shell2:#111d28; --panel:#172330; --panel2:#1d2b38;
  --panel3:#243442; --line:#314250; --soft-line:#243541; --paper:#f2ead8;
  --ink:#101820; --muted:#96a7b5; --orange:#ff7043; --amber:#f5b84b;
  --green:#49d18a; --red:#ee5148; --cyan:#36bed2; --blue:#7ea8ff; }
html, body, [data-testid="stAppViewContainer"], .stApp {
  min-height:100vh; overflow-x:hidden; overflow-y:auto;
}
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(54,190,210,.10), transparent 31%),
    linear-gradient(180deg,var(--shell2),var(--shell));
  color:#edf4f7;
}
.block-container { max-width:none; min-height:100vh; padding:.82rem 1.05rem .85rem; }
h1,h2,h3,h4 { font-family:"DIN Condensed","Avenir Next Condensed",sans-serif !important;
  text-transform:uppercase; letter-spacing:.04em; }
p,div,button,input,small { font-family:"Inter","IBM Plex Mono","Menlo",sans-serif; }
h1 { font-size:1.65rem !important; margin:0 !important; line-height:1.05 !important; color:#fff6e7; }
[data-testid="stVerticalBlock"] { gap:.55rem; }
[data-testid="stHorizontalBlock"] { gap:1rem; align-items:stretch; }
[data-testid="stSelectbox"] label { display:none; }
[data-baseweb="select"] > div {
  min-height:34px; background:#21313d; border:1px solid #3b5060; color:#eef6f7;
  border-radius:8px; box-shadow:inset 0 1px 0 rgba(255,255,255,.03);
}
.command-kicker { color:var(--amber); font-size:.68rem; letter-spacing:.16em; margin-bottom:5px; }
.command-sub { color:#9fb1bd; font-size:.76rem; margin-top:6px; line-height:1.4; }
.phase-chip {
  border:1px solid var(--line); border-left:4px solid var(--orange);
  background:linear-gradient(180deg,#1a2834,#14212b); padding:9px 12px;
  min-height:58px; color:#fff4df; border-radius:8px;
  box-shadow:0 10px 28px rgba(0,0,0,.20);
}
.phase-chip small { color:#97a8b4; display:block; font-size:.66rem; letter-spacing:.08em; }
.phase-chip b { color:#fff2d9; font-size:.96rem; }
.phase-chip span { color:#c5d1d7; font-size:.74rem; margin-left:8px; }
.progress-rail {
  height:70px; display:grid; grid-template-columns:repeat(6,1fr); gap:8px;
  padding:8px; border:1px solid var(--line); border-radius:8px;
  background:linear-gradient(180deg,rgba(31,45,57,.98),rgba(20,31,42,.98));
  box-shadow:0 10px 28px rgba(0,0,0,.20);
}
.progress-step {
  position:relative; min-width:0; padding:8px 7px 7px 30px; border-radius:7px;
  background:#172632; color:#7f91a0; border:1px solid #263947;
}
.progress-step::after {
  content:""; position:absolute; left:12px; right:-13px; top:20px; height:2px;
  background:#324657; z-index:0;
}
.progress-step:last-child::after { display:none; }
.progress-step span {
  position:absolute; left:8px; top:10px; width:16px; height:16px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:.58rem; z-index:1;
  background:#0f1b24; border:1px solid #3c5161;
}
.progress-step b { display:block; font-size:.64rem; letter-spacing:.08em; }
.progress-step small { display:block; margin-top:5px; font-size:.72rem; color:inherit; white-space:nowrap; }
.progress-done { color:#b9d6c6; border-color:rgba(73,209,138,.35); }
.progress-done span { background:rgba(73,209,138,.18); color:var(--green); border-color:rgba(73,209,138,.55); }
.progress-active { color:#fff1da; border-color:rgba(255,112,67,.65); box-shadow:0 0 0 1px rgba(255,112,67,.16); }
.progress-active span { background:rgba(255,112,67,.20); color:var(--orange); border-color:var(--orange); }
.status-capsules { display:flex; flex-wrap:wrap; gap:6px; justify-content:flex-end; margin-bottom:8px; }
.status-pill {
  display:inline-flex; align-items:center; height:25px; padding:0 9px; border-radius:999px;
  background:#1b2a36; border:1px solid #334757; color:#cbd9df; font-size:.68rem;
}
.status-pill::before { content:""; width:6px; height:6px; border-radius:50%; background:#8998a3; margin-right:6px; }
.status-ok::before { background:var(--green); box-shadow:0 0 10px rgba(73,209,138,.6); }
.status-standby::before { background:var(--amber); }
.stButton > button, .stDownloadButton > button {
  min-height:40px; border:1px solid #496071; border-radius:8px;
  background:linear-gradient(180deg,#263846,#1d2b36); color:#eef6f7;
  font-weight:800; font-size:.8rem; box-shadow:0 8px 18px rgba(0,0,0,.18);
}
.stButton > button:hover { border-color:var(--orange); color:#fff2e7; background:#2b3d4a; }
.stButton > button[kind="primary"] {
  background:linear-gradient(180deg,#ff7a4f,#f05a2f); color:#1d2022;
  border-color:#ff8c68; box-shadow:0 10px 24px rgba(255,112,67,.24);
}
.stButton > button:disabled {
  opacity:.72; color:#9aaab5; background:#1a2a36; border-color:#334958; box-shadow:none;
}
[data-testid="stPlotlyChart"] {
  border:1px solid #334754; border-radius:10px; overflow:hidden;
  box-shadow:0 14px 34px rgba(0,0,0,.24);
}
.panel-title {
  display:flex; justify-content:space-between; align-items:flex-end; gap:10px;
  margin:0 0 12px; padding-bottom:9px; border-bottom:1px solid var(--soft-line);
}
.panel-title b { color:#fff4df; font-size:1.05rem; letter-spacing:.04em; }
.panel-title span { color:#94a7b4; font-size:.76rem; }
.map-caption { display:flex; justify-content:space-between; align-items:center; color:#94a7b4;
  font-size:.72rem; letter-spacing:.04em; margin:0 2px 10px; }
.map-caption b { color:#f7ecd2; }
.map-legend {
  display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:5px 12px;
  padding:7px 10px; margin-top:8px; border:1px solid #2c4050; border-radius:8px;
  background:rgba(20,31,41,.92);
}
.map-legend span { display:flex; align-items:center; gap:7px; color:#bed0d9; font-size:.7rem; white-space:nowrap; }
.map-legend i { display:inline-block; width:15px; height:10px; border-radius:3px; border:1px solid rgba(255,255,255,.25); }
.map-empty {
  height:500px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  border:1px solid #2d4352; border-radius:8px;
  background:
    linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),
    linear-gradient(0deg,rgba(255,255,255,.035) 1px,transparent 1px),
    radial-gradient(circle at 50% 45%,#22323f,#131f29);
  background-size:28px 28px,28px 28px,auto;
  color:#8ea2b0; text-align:center; padding:20px;
}
.map-empty b { color:#f3e6cc; font-size:1rem; margin-bottom:8px; }
.map-empty span { max-width:520px; line-height:1.55; font-size:.78rem; }
.dock-label { color:#fff1d4; padding:0 0 10px; font-size:1rem; font-weight:900; }
.event-card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.event-card-head b { color:#f4ead8; font-size:.9rem; }
.event-status { border-radius:999px; padding:3px 8px; font-size:.62rem; border:1px solid #3a4d5b; color:#9fb0bb; }
.event-status.ready { color:#ffdbaa; border-color:rgba(245,184,75,.45); background:rgba(245,184,75,.08); }
.event-status.danger { color:#ffd0c9; border-color:rgba(238,81,72,.52); background:rgba(238,81,72,.10); }
.event-status.locked { opacity:.72; }
.event-meta { color:#8fa2af; font-size:.72rem; line-height:1.48; margin:7px 1px 2px; }
.event-meta span,.event-meta b,.event-meta em { display:block; }
.event-meta b { color:#d3e0e6; font-size:.7rem; margin-top:4px; }
.event-meta em { color:#7f929e; font-size:.66rem; font-style:normal; margin-top:3px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#39444b !important; border-radius:2px !important;
  background:linear-gradient(180deg,#1a2834,#14212b) !important;
  box-shadow:0 12px 30px rgba(0,0,0,.20); border-radius:10px !important; }
.inspector-head { display:grid; grid-template-columns:54px 1fr; gap:10px; align-items:center;
  border-bottom:1px solid #3d474e; padding-bottom:9px; }
.inspector-head > span { font-family:"DIN Condensed",sans-serif; font-size:2.2rem; line-height:1;
  color:var(--orange); border-right:1px solid #465159; }
.inspector-head small,.inspector-head em { display:block; color:#7f8b92; font-size:.57rem;
  letter-spacing:.13em; font-style:normal; }
.inspector-head b { display:block; color:#fff0d4; font-size:.92rem; margin:2px 0; }
.step-stack { display:flex; flex-direction:column; gap:7px; margin-bottom:10px; }
.step-card {
  display:grid; grid-template-columns:26px 1fr; gap:8px; align-items:center;
  padding:7px 8px; border:1px solid #2c3f4d; border-radius:8px; background:#15232e;
}
.step-card > span {
  width:20px; height:20px; display:flex; align-items:center; justify-content:center;
  border-radius:50%; background:#0f1b24; color:#718694; border:1px solid #304656; font-size:.7rem;
}
.step-card b { display:block; color:#cddbe2; font-size:.72rem; }
.step-card small { display:block; color:#7f93a0; font-size:.63rem; margin-top:2px; }
.step-done > span { color:var(--green); border-color:rgba(73,209,138,.55); background:rgba(73,209,138,.11); }
.step-active { border-color:rgba(255,112,67,.70); box-shadow:0 0 0 1px rgba(255,112,67,.12); }
.step-active > span { color:var(--orange); border-color:var(--orange); background:rgba(255,112,67,.13); }
.step-active b { color:#fff2dd; }
.algorithm-hint {
  border:1px solid #2e4250; border-radius:8px; padding:11px 12px; margin:8px 0 12px;
  background:rgba(14,25,34,.78);
}
.algorithm-hint b { display:block; color:#f4c66c; font-size:.7rem; margin-top:6px; }
.algorithm-hint b:first-child { margin-top:0; }
.algorithm-hint p { margin:3px 0 0; color:#b9c8cf; font-size:.68rem; line-height:1.45; }
.output-title {
  display:flex; justify-content:space-between; align-items:center; gap:10px;
  margin:10px 0 8px; padding:9px 10px; border-radius:8px;
  border:1px solid #334958; background:rgba(19,31,42,.90);
}
.output-title b { color:#fff3dc; font-size:.82rem; }
.output-title span { color:#f5bf68; font-size:.68rem; }
.empty-output {
  border:1px dashed #405466; border-radius:8px; padding:16px 14px;
  background:rgba(12,22,30,.62); color:#a8bbc6;
}
.empty-output b { display:block; color:#f2e5cf; font-size:.9rem; margin-bottom:6px; }
.empty-output span { display:block; font-size:.74rem; line-height:1.5; }
.formula-box,.alert-box { background:#0f1418; border:1px solid #3b464d; border-left:4px solid var(--cyan);
  color:#dce4e5; padding:11px 13px; font-size:.72rem; margin:5px 0 9px; line-height:1.55; border-radius:8px; }
.formula-box b { color:var(--cyan); }
.formula-box small,.alert-box small { display:block; color:#7f8b92; margin-top:4px; }
.alert-box { border-left-color:var(--orange); }
.alert-box b { color:var(--orange); }
.model-compare { display:grid; grid-template-columns:auto 1fr; gap:4px 10px; align-items:center;
  background:#18232a; border:1px solid #3d4d55; border-left:4px solid var(--amber);
  padding:9px 11px; margin:5px 0 8px; }
.model-compare b { color:var(--amber); font-size:.82rem; }
.model-compare span { color:#c8d1d4; font-size:.62rem; }
.model-compare em { color:var(--cyan); font-style:normal; font-size:.65rem; }
.model-compare small { grid-column:1 / -1; color:#89959b; font-size:.56rem; line-height:1.45; }
.learned-advantage { margin-top:5px; border:1px solid #51442c; border-left:4px solid var(--amber);
  background:#1b211d; padding:7px 8px; color:#f5ead4; line-height:1.35; }
.learned-advantage b { color:var(--amber); display:block; font-size:.66rem; }
.learned-advantage span { color:#dce4e5; display:block; font-size:.55rem; margin-top:2px; }
.learned-advantage small { color:#9aa69c; display:block; font-size:.49rem; margin-top:2px; }
.count-chip { display:inline-flex; flex-direction:column; min-width:66px; border:1px solid #3c474e;
  padding:7px 9px; margin:3px 4px 7px 0; color:#8f9aa0; font-size:.55rem; text-transform:uppercase; }
.count-chip b { color:var(--cyan); font-size:1rem; }
.check-line { border-bottom:1px dotted #38434a; padding:8px 4px; color:#bdc7ca; font-size:.65rem; }
.check-line::first-letter { color:var(--green); }
.empty-inspector { border-top:5px solid var(--orange); padding-top:8px; }
[data-testid="stDataFrame"] { border:1px solid #354047; border-radius:8px; overflow:hidden; }
.resource-title {
  display:flex; justify-content:space-between; align-items:center; margin:10px 0 6px;
  color:#fff2dc; font-size:1rem; font-weight:900;
}
.resource-title span { color:#94a7b4; font-size:.72rem; font-weight:500; }
.footer-strip { min-height:104px; display:grid; grid-template-columns:repeat(5,1fr); gap:10px;
  padding-top:4px; }
.unit-card {
  position:relative; background:linear-gradient(180deg,#1a2a36,#14222d);
  border:1px solid #334859; border-radius:8px; padding:12px 12px 11px 32px;
  min-width:0; box-shadow:0 10px 24px rgba(0,0,0,.18);
}
.unit-card i { position:absolute; left:13px; top:16px; width:8px; height:8px; border-radius:50%;
  background:var(--cyan); box-shadow:0 0 0 4px rgba(54,190,210,.12); }
.unit-card i.state-idle,.unit-card i.state-ready { background:var(--green); }
.unit-card i.state-stranded { background:var(--red); }
.unit-card b { color:#f5ead4; font-size:.82rem; display:block; white-space:nowrap; overflow:hidden; }
.unit-card span {
  position:absolute; right:10px; top:10px; color:#ffe1c9; font-size:.62rem;
  border:1px solid rgba(255,112,67,.38); background:rgba(255,112,67,.10);
  padding:2px 7px; border-radius:999px;
}
.unit-card small { color:#a7bac5; font-size:.67rem; line-height:1.55; display:block; margin-top:7px; }
.unit-progress { height:5px; border-radius:999px; background:#263b49; margin-top:9px; overflow:hidden; }
.unit-progress i { position:static; display:block; width:var(--progress); height:100%; border-radius:999px;
  background:linear-gradient(90deg,var(--green),var(--amber)); box-shadow:none; }
[data-testid="stToast"] { background:#252e34; color:#fff0d4; }
@media (max-width:1200px) {
  .progress-step small { font-size:.62rem; }
  .map-legend { grid-template-columns:repeat(4,minmax(0,1fr)); }
  .footer-strip { grid-template-columns:repeat(3,1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

session = _load_session()
record = _selected_record(session) if session else None

title_col, progress_col, control_col = st.columns([1.35, 2.65, 1.55])
with title_col:
    st.markdown("<div class='command-kicker'>离线救援决策实验台</div>", unsafe_allow_html=True)
    st.title("AI Emergency Commander｜灾后救援指挥台")
    st.markdown("<div class='command-sub'>灾后救援指挥台 / Disaster Response Console</div>", unsafe_allow_html=True)
with progress_col:
    st.markdown(_render_progress_bar(session), unsafe_allow_html=True)
with control_col:
    selected_model = st.selectbox(
        "概率模型",
        ["固定专家 CPT", "学习 CPT"],
        key="model_selector",
        label_visibility="collapsed",
    )
    st.markdown(_render_status_capsules(session, selected_model), unsafe_allow_html=True)
    if selected_model == "学习 CPT":
        _render_learned_advantage_panel()
    action_a, action_b = st.columns(2)
    with action_a:
        st.button(
            "导入地图模型",
            key="generate_map",
            on_click=start_random_session,
            type="primary",
            width="stretch",
        )
    with action_b:
        st.button(
            "执行下一算法步骤",
            key="advance_phase",
            on_click=advance_phase,
            disabled=session is None or session.status != "running" or session.phase == "execute",
            width="stretch",
        )
    next_transition = session.next_transition_minutes() if session and session.phase == "execute" else None
    action_c, action_d = st.columns(2)
    with action_c:
        st.button(
            "推进 1 分钟",
            key="advance_minute",
            on_click=advance_execution,
            disabled=session is None or session.status != "running" or session.phase != "execute",
            width="stretch",
        )
    with action_d:
        st.button(
            "推进到状态",
            key="advance_transition",
            on_click=advance_execution,
            kwargs={"to_transition": True},
            disabled=next_transition is None or (session is not None and session.status != "running"),
            width="stretch",
        )

notice = st.session_state.pop("event_notice", None)
if notice:
    st.toast(notice)

if session is None:
    map_column, event_column, inspector_column = st.columns([6.2, 1.8, 2.4], gap="large")
    with map_column:
        with st.container(border=True):
            st.markdown(
                "<div class='panel-title'><b>灾区态势地图 / Situation Map</b>"
                "<span>等待生成场景</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='map-empty'><b>等待导入地图模型</b>"
                "<span>点击“导入地图模型”，系统将载入 4-7 个灾区、随机道路骨架、受损路段和 5 个异构救援单位。</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(_render_map_legend(), unsafe_allow_html=True)
    with event_column:
        with st.container(height=620, border=True):
            _render_event_dock(None)
    with inspector_column:
        with st.container(height=620, border=True):
            st.markdown("<div class='dock-label'>算法步骤</div>", unsafe_allow_html=True)
            st.markdown(_render_step_stack(None), unsafe_allow_html=True)
            st.markdown(_render_algorithm_hint(None, None), unsafe_allow_html=True)
            st.info("等待生成地图后开始输入校验。")
else:
    snapshot = _current_snapshot(session)
    focus = record.get("focus", {}) if record else {}
    map_column, event_column, inspector_column = st.columns([6.2, 1.8, 2.4], gap="large")
    with map_column:
        with st.container(border=True):
            st.markdown(
                "<div class='panel-title'><b>灾区态势地图 / Situation Map</b>"
                "<span>当前计算对象高亮显示</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='map-caption'><b>战术路网：{len(session.scenario['nodes'])} 个节点 · "
                f"{len(session.scenario['roads'])} 条道路</b><span>滚轮可缩放，悬停可查看地块含义</span></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                build_map_figure(session.scenario, snapshot, focus=focus),
                width="stretch",
                key=f"command_map_{session.step_count}_{len(session.event_log)}_{st.session_state.get('history_index', -1)}",
                config={"displayModeBar": False, "scrollZoom": True},
            )
            st.markdown(_render_map_legend(), unsafe_allow_html=True)
    with event_column:
        with st.container(height=620, border=True):
            _render_event_dock(session)
    with inspector_column:
        with st.container(height=620, border=True):
            st.markdown("<div class='dock-label'>算法步骤</div>", unsafe_allow_html=True)
            nav_prev, nav_label, nav_next = st.columns([1, 1.4, 1])
            with nav_prev:
                st.button(
                    "上一条",
                    key="history_previous",
                    on_click=move_history,
                    args=(-1,),
                    disabled=not session.calculation_history
                    or st.session_state.get("history_index", 0) <= 0,
                    width="stretch",
                )
            with nav_label:
                st.markdown(
                    f"<div style='text-align:center;color:#91a4b1;font-size:.66rem;padding-top:10px'>"
                    f"步骤 {session.step_count} · 事件 {len(session.event_log)}</div>",
                    unsafe_allow_html=True,
                )
            with nav_next:
                st.button(
                    "下一条",
                    key="history_next",
                    on_click=move_history,
                    args=(1,),
                    disabled=not session.calculation_history
                    or st.session_state.get("history_index", -1)
                    >= len(session.calculation_history) - 1,
                    width="stretch",
            )
            _render_calculation_inspector(session)
    st.markdown(
        "<div class='resource-title'>救援资源 / Rescue Assets"
        "<span>车辆、无人机与当前任务状态</span></div>",
        unsafe_allow_html=True,
    )
    _render_footer(session)
