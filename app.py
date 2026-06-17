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
    st.session_state["event_notice"] = f"复杂救援地图已生成 · 种子 {seed}"


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
    if record is None:
        st.markdown("<div class='empty-inspector'>", unsafe_allow_html=True)
        st.markdown("#### 等待第一步计算")
        st.caption("地图已生成。点击“执行下一算法步骤”，从输入校验开始展示完整证据链。")
        st.markdown(
            """
            1. JSON 结构校验与归一化
            2. 贝叶斯精确枚举
            3. 加权风险排序
            4. 风险感知 A*
            5. 期望效用分解
            6. 全局组合分配
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
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
            f"<div class='event-meta'>{html.escape(hint)}<b>目标 · {html.escape(_display_node_id(selected))}</b></div>",
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
        cards.append(
            "<div class='unit-card'>"
            f"<i class='state-{html.escape(status)}'></i>"
            f"<b>{html.escape(_display_unit_id(unit_id))}</b><span>{html.escape(status_label)}</span>"
            f"<small>目标 {html.escape(_display_zone_id(target))} · 剩余 {state.get('remaining_travel', 0):.1f} 分钟 · "
            f"载员 {state.get('onboard', 0)}/{state.get('capacity', 0)}</small></div>"
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
:root { --shell:#11161b; --panel:#1b2228; --panel2:#242c32; --line:#39434a;
  --paper:#f3ead5; --ink:#151a1d; --muted:#8c989f; --orange:#ff6332;
  --cyan:#25c4d8; --amber:#f0b33a; --green:#50c98b; --red:#e94f37; }
html, body, [data-testid="stAppViewContainer"], .stApp { height:100vh; overflow:hidden; }
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
.stApp { background:var(--shell); color:#f7eedc; }
.block-container { max-width:none; height:100vh; overflow:hidden; padding:.55rem .8rem .4rem; }
h1,h2,h3,h4 { font-family:"DIN Condensed","Avenir Next Condensed",sans-serif !important;
  text-transform:uppercase; letter-spacing:.055em; }
p,div,button,input,small { font-family:"IBM Plex Mono","Menlo",monospace; }
h1 { font-size:1.42rem !important; margin:0 !important; line-height:1 !important; color:#fff4df; }
[data-testid="stVerticalBlock"] { gap:.42rem; }
[data-testid="stHorizontalBlock"] { gap:.55rem; align-items:center; }
[data-testid="stSelectbox"] label { display:none; }
[data-baseweb="select"] > div { background:#20282e; border-color:#45515a; color:#fff4df; border-radius:2px; }
.command-kicker { color:var(--orange); font-size:.66rem; letter-spacing:.18em; margin-bottom:3px; }
.command-sub { color:#87939a; font-size:.67rem; margin-top:4px; }
.phase-chip { border-left:4px solid var(--orange); background:#1b2228; padding:7px 10px;
  min-height:44px; color:#fff4df; }
.phase-chip small { color:#839198; display:block; font-size:.58rem; letter-spacing:.12em; }
.phase-chip b { color:var(--orange); font-size:.82rem; }
.phase-chip span { color:#c9d1d5; font-size:.67rem; margin-left:8px; }
.stButton > button, .stDownloadButton > button { min-height:36px; border:1px solid #53616a;
  border-radius:2px; background:#20282e; color:#f8eedb; font-weight:800; font-size:.72rem; }
.stButton > button:hover { border-color:var(--orange); color:var(--orange); }
.stButton > button[kind="primary"] { background:var(--orange); color:#151a1d; border-color:var(--orange); }
.stButton > button:disabled { opacity:.72; color:#6f7b82; background:#151b20; border-color:#2e383e; }
[data-testid="stPlotlyChart"] { border:1px solid #3a454c; box-shadow:0 0 0 1px #0b0e10; }
.map-caption { display:flex; justify-content:space-between; align-items:center; color:#9ba5aa;
  font-size:.62rem; letter-spacing:.08em; margin:0 2px -2px; }
.map-caption b { color:#f4e6ca; }
.dock-label { color:var(--orange); border-bottom:2px solid var(--orange); padding:0 0 7px;
  font-size:.65rem; letter-spacing:.18em; font-weight:900; }
.event-meta { color:#7f8b92; font-size:.55rem; line-height:1.35; margin:-3px 1px 9px; }
.event-meta b { display:block; color:#bac4c8; font-size:.52rem; margin-top:2px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#39444b !important; border-radius:2px !important;
  background:linear-gradient(180deg,#1d252b,#171d22); }
.inspector-head { display:grid; grid-template-columns:54px 1fr; gap:10px; align-items:center;
  border-bottom:1px solid #3d474e; padding-bottom:9px; }
.inspector-head > span { font-family:"DIN Condensed",sans-serif; font-size:2.2rem; line-height:1;
  color:var(--orange); border-right:1px solid #465159; }
.inspector-head small,.inspector-head em { display:block; color:#7f8b92; font-size:.57rem;
  letter-spacing:.13em; font-style:normal; }
.inspector-head b { display:block; color:#fff0d4; font-size:.92rem; margin:2px 0; }
.formula-box,.alert-box { background:#0f1418; border:1px solid #3b464d; border-left:4px solid var(--cyan);
  color:#dce4e5; padding:10px 12px; font-size:.68rem; margin:5px 0 8px; line-height:1.55; }
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
[data-testid="stDataFrame"] { border:1px solid #354047; }
.footer-strip { height:66px; display:grid; grid-template-columns:repeat(5,1fr); gap:7px;
  border-top:1px solid #3d484f; padding-top:7px; }
.unit-card { position:relative; background:#192026; border:1px solid #333e45; padding:6px 8px 5px 24px;
  min-width:0; }
.unit-card i { position:absolute; left:9px; top:10px; width:7px; height:7px; border-radius:50%;
  background:var(--cyan); box-shadow:0 0 0 3px rgba(37,196,216,.12); }
.unit-card i.state-idle,.unit-card i.state-ready { background:var(--green); }
.unit-card i.state-stranded { background:var(--red); }
.unit-card b { color:#f5ead4; font-size:.62rem; display:block; white-space:nowrap; overflow:hidden; }
.unit-card span { position:absolute; right:7px; top:6px; color:var(--orange); font-size:.52rem; }
.unit-card small { color:#7f8b92; font-size:.51rem; white-space:nowrap; }
[data-testid="stToast"] { background:#252e34; color:#fff0d4; }
@media (max-width:1100px) { .footer-strip { grid-template-columns:repeat(3,1fr); } }
</style>
""",
    unsafe_allow_html=True,
)

session = _load_session()
record = _selected_record(session) if session else None

header_title, header_meta, model_col, generate_col, next_col, minute_col, transition_col = st.columns(
    [2.25, 1.55, 1.05, 1.05, 1.22, 1.08, 1.3]
)
with header_title:
    st.markdown("<div class='command-kicker'>离线救援决策实验台</div>", unsafe_allow_html=True)
    st.title("AI Emergency Commander｜灾后救援指挥台")
    st.markdown("<div class='command-sub'>复杂路网 · 可解释算法 · 手动应急推演</div>", unsafe_allow_html=True)
with header_meta:
    if session:
        number, label, algorithm = PHASE_LABELS[session.phase]
        st.markdown(
            f"<div class='phase-chip'><small>种子 {session.seed} · 时间 +{session.clock_minutes:.1f} 分钟</small>"
            f"<b>{number} {label}</b><span>{algorithm}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='phase-chip'><small>等待场景</small><b>00 待命</b>"
            "<span>生成地图后开始</span></div>",
            unsafe_allow_html=True,
        )
with model_col:
    selected_model = st.selectbox(
        "概率模型",
        ["固定专家 CPT", "学习 CPT"],
        key="model_selector",
        label_visibility="collapsed",
    )
    if selected_model == "学习 CPT":
        _render_learned_advantage_panel()
with generate_col:
    st.button(
        "生成复杂地图",
        key="generate_map",
        on_click=start_random_session,
        type="primary",
        width="stretch",
    )
with next_col:
    st.button(
        "执行下一算法步骤",
        key="advance_phase",
        on_click=advance_phase,
        disabled=session is None or session.status != "running" or session.phase == "execute",
        width="stretch",
    )
with minute_col:
    st.button(
        "推进 1 分钟",
        key="advance_minute",
        on_click=advance_execution,
        disabled=session is None or session.status != "running" or session.phase != "execute",
        width="stretch",
    )
with transition_col:
    next_transition = session.next_transition_minutes() if session and session.phase == "execute" else None
    st.button(
        "推进到下一状态",
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
    st.markdown(
        "<div class='phase-chip' style='margin-top:8px'><small>指挥台就绪</small>"
        "<b>等待生成复杂救援地图</b><span>系统将创建 4-7 个灾区、随机道路骨架、受损路段和 5 个异构救援单位。</span></div>",
        unsafe_allow_html=True,
    )
    empty_left, empty_events, empty_detail = st.columns([5.7, 1.15, 3.15])
    with empty_left:
        st.markdown(
            "<div style='height:610px;border:1px solid #39444b;background:radial-gradient(circle at 50% 45%,#283139,#171d22);"
            "display:flex;align-items:center;justify-content:center;color:#647078;font-size:.75rem;letter-spacing:.12em'>"
            "地图阵列待生成</div>",
            unsafe_allow_html=True,
        )
    with empty_events:
        _render_event_dock(None)
    with empty_detail:
        with st.container(height=610, border=True):
            st.markdown("#### 计算证据链")
            st.caption("生成地图后，每一步由演示者手动触发。")
else:
    snapshot = _current_snapshot(session)
    focus = record.get("focus", {}) if record else {}
    map_column, event_column, inspector_column = st.columns([5.7, 1.15, 3.15])
    with map_column:
        st.markdown(
            f"<div class='map-caption'><b>战术路网 / {len(session.scenario['nodes'])} 个节点 · "
            f"{len(session.scenario['roads'])} 条道路</b><span>当前计算对象高亮显示</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_map_figure(session.scenario, snapshot, focus=focus),
            width="stretch",
            key=f"command_map_{session.step_count}_{len(session.event_log)}_{st.session_state.get('history_index', -1)}",
            config={"displayModeBar": False, "scrollZoom": True},
        )
    with event_column:
        _render_event_dock(session)
    with inspector_column:
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
                f"<div style='text-align:center;color:#7f8b92;font-size:.58rem;padding-top:10px'>"
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
        with st.container(height=560, border=True):
            _render_calculation_inspector(session)
    _render_footer(session)
