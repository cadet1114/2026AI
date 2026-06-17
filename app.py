from __future__ import annotations

import json
import os
import copy
import random
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.demo_engine import (
    ROUTE_COLORS,
    apply_road_collapse,
    apply_update_json_to_scenario,
    assign_tasks,
    calculate_route_cost,
    compute_zone_scores,
    generate_random_scenario,
    generate_report,
    load_scenario,
    plan_routes,
    summarize_update_json,
)
from core.a_engine_adapter import AEngineUnavailable, run_a_engine_on_grid
from core.qwen_client import (
    QwenApiError,
    generate_qwen_report,
    merge_recognition_into_scenario,
    parse_disaster_update_with_qwen,
    recognize_disaster_image,
)


BASE_DIR = Path(__file__).resolve().parent
SCENARIO_PATH = BASE_DIR / "data" / "scenario.json"
DEFAULT_IMAGE_PATH = BASE_DIR / "assets" / "disaster_grid_input.png"
EXPORT_DIR = BASE_DIR / "exports"
CURRENT_SCENARIO_EXPORT_PATH = EXPORT_DIR / "current_scenario.json"
MAP_RENDER_SCALE = 3
COST_MODEL_VERSION = "terrain-cost-v5-air-grid"
SCENARIO_VERSION = "scenario-v4-expanded-buildings"
ENGINE_DEMO = "B 演示引擎"
ENGINE_A_ADAPTER = "A 同学算法适配"
PIXEL_TILE_COLORS = [
    "#c8d7a0",
    "#b9cc91",
    "#d8e0b0",
    "#6f7880",
    "#808991",
    "#59616a",
    "#9aa4ad",
    "#7d8791",
    "#b2bac1",
    "#f2d16b",
    "#e5bf55",
    "#f28a2e",
    "#f05f22",
    "#ffd166",
    "#202124",
    "#3a3a3a",
    "#7b3140",
    "#a1444e",
    "#5ca8c8",
    "#75bdd8",
    "#7dae68",
    "#91bd76",
]
TILE_VARIANTS = {
    0: (0, 1, 2),
    1: (3, 4, 5),
    2: (6, 7, 8),
    3: (9, 10),
    4: (11, 12, 13),
    5: (14, 15),
    6: (16, 17),
    7: (18, 19),
    8: (20, 21),
}
TILE_CATEGORY_META = {
    0: {
        "name": "地面/空地",
        "description": "救援车可通行，单格代价 1.8；无人机按空中格网飞越。",
        "color": "#c8d7a0",
    },
    1: {
        "name": "道路",
        "description": "救援车优先通行，单格代价 1.0。",
        "color": "#6f7880",
    },
    2: {
        "name": "建筑",
        "description": "救援车不可通行；无人机可飞越，空中障碍代价 +0.3。",
        "color": "#9aa4ad",
    },
    3: {
        "name": "拥堵",
        "description": "救援车额外代价 +3.5；无人机空中代价 +0.35。",
        "color": "#f2d16b",
    },
    4: {
        "name": "火灾风险",
        "description": "救援车额外代价 +5.0；无人机空中代价 +2.4。",
        "color": "#f28a2e",
    },
    5: {
        "name": "断路",
        "description": "不可通行道路，救援车不能穿过。",
        "color": "#202124",
    },
    6: {
        "name": "塌方风险",
        "description": "救援车额外代价 +4.0；无人机空中代价 +1.1；模拟塌方后转为断路。",
        "color": "#7b3140",
    },
    7: {
        "name": "水域",
        "description": "救援车不可通行；无人机可飞越，空中障碍代价 +0.3。",
        "color": "#5ca8c8",
    },
    8: {
        "name": "公园/绿地",
        "description": "救援车可绕行，单格代价 1.8；无人机按空中格网飞越。",
        "color": "#7dae68",
    },
}


st.set_page_config(
    page_title="AI Emergency Commander",
    page_icon="AEC",
    layout="wide",
)


def main() -> None:
    _inject_styles()
    _init_session_state()
    _ensure_scenario_current()
    _ensure_cost_model_current()

    st.title("AI Emergency Commander：灾后救援智能指挥系统")
    st.caption("B 分工控制台：24×24 俯视像素灾区地图、自然语言灾情更新、可解释推理、任务分配与动态重规划")

    with st.sidebar:
        st.header("演示控制台")
        _render_api_controls()

        st.divider()
        _render_algorithm_controls()

        st.divider()
        st.subheader("场景初始化")
        if st.button("加载初始灾区场景", use_container_width=True):
            _load_initial_scene()
        st.text_input(
            "随机种子（可选）",
            key="random_seed_text",
            placeholder="留空则每次随机；填写后可复现同一张地图",
        )
        if st.button("随机生成灾区场景", use_container_width=True):
            _load_random_scene()

        st.divider()
        st.subheader("自然语言灾情更新")
        st.text_area(
            "灾情变化",
            key="disaster_update_text",
            height=150,
            placeholder="例如：无人机发现 C 区北侧道路可以通行，但 C 区火势扩大，SOS 信号增强。",
        )
        if st.button("应用灾情更新并重新规划", use_container_width=True):
            _apply_natural_language_update()

        st.divider()
        st.subheader("演示测试")
        if st.button("一键演示完整流程", use_container_width=True):
            _run_full_demo()
        if st.button("模拟道路塌方", use_container_width=True):
            _simulate_collapse()

        st.divider()
        st.subheader("报告输出")
        if st.button("生成救援报告", use_container_width=True):
            _generate_report()

        st.divider()
        _render_scenario_export_panel()

        st.divider()
        _render_optional_image_panel()

        st.divider()
        st.subheader("图例")
        st.markdown(
            """
            - 红线：RescueCar-1
            - 蓝线：RescueCar-2
            - 绿线：Drone-1 空中路径
            - 灰色路面：救援车优先通行道路
            - 深灰建筑/水域：救援车不可直接穿过
            - 黑色格：断裂道路，救援车不可通行
            - 深红格：塌方风险或塌方路段
            - 橙色格：火灾风险
            - 黄色格：拥堵道路
            """
        )

    if st.session_state.status_message:
        st.info(st.session_state.status_message)
    _render_last_update_summary()

    if _routes_cross_forbidden_cells(st.session_state.scenario, st.session_state.routes):
        _replan_current_scenario("检测到旧路线经过不可通行格，系统已自动重新规划。")

    _render_summary_metrics()

    scenario = st.session_state.scenario
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.subheader("灾区地图与路线")
        st.plotly_chart(
            _build_map_figure(
                scenario,
                st.session_state.routes,
                st.session_state.assignments,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        _render_map_tile_legend()
        _render_route_summary(st.session_state.assignments, st.session_state.routes)

    with right:
        st.subheader("概率推理与优先级")
        _render_scores(st.session_state.zone_scores)
        st.subheader("任务分配")
        _render_assignments(
            st.session_state.assignments,
            st.session_state.routes,
            st.session_state.scenario,
            st.session_state.route_details,
        )
        _render_previous_plan_snapshot()

    st.subheader("救援报告")
    st.caption(f"报告生成方式：{st.session_state.report_source}")
    st.text_area(
        "报告内容",
        value=st.session_state.report_text,
        height=230,
        label_visibility="collapsed",
        disabled=True,
    )
    _render_current_scenario_preview()

    _render_debug_outputs()


def _init_session_state() -> None:
    if "scenario" not in st.session_state:
        st.session_state.scenario = load_scenario(SCENARIO_PATH)
    st.session_state.setdefault("zone_scores", {})
    st.session_state.setdefault("assignments", {})
    st.session_state.setdefault("routes", {})
    st.session_state.setdefault("previous_scenario", {})
    st.session_state.setdefault("previous_zone_scores", {})
    st.session_state.setdefault("previous_assignments", {})
    st.session_state.setdefault("previous_routes", {})
    st.session_state.setdefault("previous_snapshot_label", "")
    st.session_state.setdefault("report_text", "点击左侧“加载初始灾区场景”开始课堂演示。")
    st.session_state.setdefault("report_source", "尚未生成")
    st.session_state.setdefault("status_message", "")
    st.session_state.setdefault("scenario_source", "预设 scenario.json")
    st.session_state.setdefault("scenario_seed", "-")
    st.session_state.setdefault("algorithm_engine", ENGINE_DEMO)
    st.session_state.setdefault("last_algorithm_engine", st.session_state.algorithm_engine)
    st.session_state.setdefault("engine_status", "当前使用 B 演示引擎。")
    st.session_state.setdefault("engine_summary", {})
    st.session_state.setdefault("route_details", {})
    st.session_state.setdefault("previous_route_details", {})
    st.session_state.setdefault("last_update_summary", "")
    st.session_state.setdefault("qwen_api_key", "")
    st.session_state.setdefault("qwen_report_enabled", True)
    st.session_state.setdefault("qwen_text_backend", "千问 API")
    st.session_state.setdefault("qwen_text_model", "qwen-max")
    st.session_state.setdefault("local_qwen_endpoint", "http://127.0.0.1:8000/v1/chat/completions")
    st.session_state.setdefault("local_qwen_model", "qwen2.5-7b-instruct")
    st.session_state.setdefault("local_qwen_api_key", "")
    st.session_state.setdefault("qwen_vl_model", "qwen-vl-max")
    st.session_state.setdefault("qwen_image_mode", "标准网格图识别")
    st.session_state.setdefault("qwen_raw_json", {})
    st.session_state.setdefault("qwen_raw_text", "")
    st.session_state.setdefault("qwen_update_json", {})
    st.session_state.setdefault("qwen_update_raw_text", "")
    st.session_state.setdefault("disaster_update_text", "")
    st.session_state.setdefault("random_seed_text", "")
    st.session_state.setdefault("uploaded_image_bytes", None)
    st.session_state.setdefault("uploaded_image_mime", "image/png")
    st.session_state.setdefault("uploaded_image_name", "")
    st.session_state.setdefault("scenario_export_path", "")
    st.session_state.setdefault("scenario_export_error", "")


def _ensure_cost_model_current() -> None:
    if st.session_state.get("cost_model_version") == COST_MODEL_VERSION:
        return
    st.session_state.cost_model_version = COST_MODEL_VERSION
    if st.session_state.routes:
        _replan_current_scenario("路径代价模型已更新：拥堵惩罚提高，路线已重新规划。")


def _ensure_scenario_current() -> None:
    if st.session_state.get("scenario_version") == SCENARIO_VERSION:
        return
    st.session_state.scenario_version = SCENARIO_VERSION
    st.session_state.scenario = load_scenario(SCENARIO_PATH)
    st.session_state.scenario_source = "预设 scenario.json"
    st.session_state.scenario_seed = "-"
    _clear_previous_plan_snapshot()
    st.session_state.qwen_update_json = {}
    st.session_state.qwen_update_raw_text = ""
    st.session_state.last_update_summary = "预设场景已更新：建筑占地调大"
    if st.session_state.zone_scores or st.session_state.routes:
        _replan_current_scenario("预设场景已更新：建筑占地调大，系统已重新规划路线。")
    else:
        st.session_state.report_text = "预设场景已更新：建筑占地调大。点击“加载初始灾区场景”开始演示。"


def _render_api_controls() -> None:
    with st.expander("模型接入设置", expanded=True):
        st.radio(
            "文本模型来源",
            options=["千问 API", "本地 Qwen 7B"],
            key="qwen_text_backend",
            help="自然语言灾情解析和救援报告使用这里选择的后端。",
        )

        st.session_state.qwen_report_enabled = st.checkbox(
            "生成报告时调用所选文本模型",
            value=st.session_state.qwen_report_enabled,
        )

        if st.session_state.qwen_text_backend == "千问 API":
            env_key_ready = bool(os.getenv("DASHSCOPE_API_KEY"))
            api_key_input = st.text_input(
                "DASHSCOPE_API_KEY",
                type="password",
                value="",
                placeholder="已从环境变量读取" if env_key_ready else "仅保存在本次页面会话",
            )
            st.session_state.qwen_api_key = api_key_input.strip() or os.getenv(
                "DASHSCOPE_API_KEY", ""
            )
            st.session_state.qwen_text_model = st.selectbox(
                "API 文本模型",
                options=["qwen-max", "qwen-plus", "qwen-turbo"],
                index=_option_index(
                    ["qwen-max", "qwen-plus", "qwen-turbo"],
                    st.session_state.qwen_text_model,
                ),
                help="用于自然语言灾情解析和救援报告生成。",
            )
        else:
            st.text_input(
                "本地 Qwen 7B 地址",
                key="local_qwen_endpoint",
                help="填写 OpenAI-compatible 接口，例如 http://127.0.0.1:8000/v1/chat/completions。",
            )
            st.text_input(
                "本地模型名",
                key="local_qwen_model",
                help="例如 qwen2.5-7b-instruct，具体名称以组员本地服务为准。",
            )
            local_key_input = st.text_input(
                "本地 API Key（可选）",
                type="password",
                value="",
                placeholder="多数本地服务可留空",
            )
            if local_key_input.strip():
                st.session_state.local_qwen_api_key = local_key_input.strip()
            env_key_ready = bool(os.getenv("DASHSCOPE_API_KEY"))
            vl_api_key_input = st.text_input(
                "DASHSCOPE_API_KEY（图片识别用，可选）",
                type="password",
                value="",
                placeholder="已从环境变量读取" if env_key_ready else "仅 Qwen-VL 图片识别需要",
            )
            st.session_state.qwen_api_key = vl_api_key_input.strip() or os.getenv(
                "DASHSCOPE_API_KEY", ""
            )
            st.caption("本地 Qwen 7B 只处理文字：自然语言灾情更新、救援报告。图片识别仍需要 Qwen-VL。")

        st.session_state.qwen_vl_model = st.selectbox(
            "视觉模型（图片识别用）",
            options=["qwen-vl-max", "qwen-vl-plus"],
            index=_option_index(
                ["qwen-vl-max", "qwen-vl-plus"],
                st.session_state.qwen_vl_model,
            ),
            help="仅用于可选图片识别功能。普通本地 Qwen 7B 不能看图。",
        )


def _render_algorithm_controls() -> None:
    with st.expander("算法引擎设置", expanded=True):
        previous_engine = st.session_state.get("last_algorithm_engine", ENGINE_DEMO)
        st.radio(
            "推理 / 分配 / 路径算法",
            options=[ENGINE_DEMO, ENGINE_A_ADAPTER],
            key="algorithm_engine",
            help=(
                "B 演示引擎保证稳定闭环；A 同学算法适配会把当前 24x24 区块地图转换成图结构，"
                "再调用 A 的贝叶斯推理、效用分配和风险 A*。"
            ),
        )
        if st.session_state.algorithm_engine == ENGINE_A_ADAPTER:
            st.caption("区块、坍塌、火灾、拥堵等图层仍按当前 B 地图展示；算法结果来自 A 引擎适配层。")
        else:
            st.caption("使用 B 端内置临时逻辑，适合稳定课堂演示。")

        if (
            previous_engine != st.session_state.algorithm_engine
            and (st.session_state.zone_scores or st.session_state.routes)
        ):
            st.session_state.last_algorithm_engine = st.session_state.algorithm_engine
            _replan_current_scenario(
                f"算法引擎已切换为 {st.session_state.algorithm_engine}，已重新推理、分配并规划路线。"
            )
        else:
            st.session_state.last_algorithm_engine = st.session_state.algorithm_engine


def _render_optional_image_panel() -> None:
    with st.expander("可选功能：图片识别生成 24×24 场景", expanded=False):
        st.caption("实验项：用于展示 Qwen-VL 可把图片转成场景 JSON，主流程不依赖它。")
        st.session_state.qwen_image_mode = st.radio(
            "图片类型",
            options=["标准网格图识别", "实验功能：真实图片抽象为24×24网格"],
            index=_image_mode_index(st.session_state.qwen_image_mode),
        )

        uploaded = st.file_uploader(
            "上传灾区图片",
            type=["png", "jpg", "jpeg"],
            help="不上传时默认使用 assets/disaster_grid_input.png。",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            st.session_state.uploaded_image_bytes = image_bytes
            st.session_state.uploaded_image_mime = uploaded.type or "image/png"
            st.session_state.uploaded_image_name = uploaded.name
            st.image(image_bytes, caption=uploaded.name, use_container_width=True)
        elif DEFAULT_IMAGE_PATH.exists():
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_mime = "image/png"
            st.session_state.uploaded_image_name = ""
            st.image(str(DEFAULT_IMAGE_PATH), caption="内置灾区网格图", use_container_width=True)

        if st.button("从图片加载 24×24 场景", use_container_width=True):
            _load_scene_from_image()


def _render_scenario_export_panel() -> None:
    st.subheader("场景 JSON 导出")
    st.caption("临时导出格式与 data/scenario.json 保持一致，等组员给格式后再做转换。")
    _sync_current_scenario_export()

    json_text = _current_scenario_json_text()
    st.download_button(
        "下载当前场景 JSON",
        data=json_text.encode("utf-8"),
        file_name=_current_scenario_export_name(),
        mime="application/json",
        use_container_width=True,
    )

    export_path = st.session_state.get("scenario_export_path", "")
    if export_path:
        st.caption(f"本地自动保存：{export_path}")
    export_error = st.session_state.get("scenario_export_error", "")
    if export_error:
        st.warning(export_error)


def _load_initial_scene() -> None:
    st.session_state.scenario = load_scenario(SCENARIO_PATH)
    st.session_state.scenario_source = "预设 scenario.json"
    st.session_state.scenario_seed = "-"
    _clear_previous_plan_snapshot()
    st.session_state.qwen_raw_json = {}
    st.session_state.qwen_raw_text = ""
    st.session_state.qwen_update_json = {}
    st.session_state.qwen_update_raw_text = ""
    st.session_state.last_update_summary = ""
    _replan_current_scenario("已加载初始灾区场景，并完成推理、分配和路线规划。")


def _load_random_scene() -> None:
    seed = _parse_random_seed(st.session_state.get("random_seed_text", ""))
    if seed is None:
        seed = random.SystemRandom().randint(1, 2_147_483_647)

    st.session_state.scenario = generate_random_scenario(seed)
    st.session_state.scenario_source = "随机生成"
    st.session_state.scenario_seed = str(seed)
    _clear_previous_plan_snapshot()
    st.session_state.qwen_raw_json = {}
    st.session_state.qwen_raw_text = ""
    st.session_state.qwen_update_json = {}
    st.session_state.qwen_update_raw_text = ""
    st.session_state.last_update_summary = f"随机生成灾区场景，seed={seed}"
    _replan_current_scenario(
        f"已随机生成灾区场景并完成推理与路线规划。当前 seed = {seed}。"
    )


def _apply_natural_language_update() -> None:
    update_text = st.session_state.disaster_update_text.strip()
    if not update_text:
        st.session_state.status_message = "请输入灾情变化描述后再应用更新。"
        return

    _capture_previous_plan_snapshot("自然语言灾情更新前")
    text_model_config = _current_text_model_config()
    parser_source = f"{text_model_config['backend']} update_json"
    try:
        if not _text_model_ready(text_model_config):
            if text_model_config["backend"] == "本地 Qwen 7B":
                raise QwenApiError("未配置本地 Qwen 7B 地址")
            raise QwenApiError("未检测到 DASHSCOPE_API_KEY")
        with st.spinner(f"正在调用{text_model_config['backend']}解析灾情更新..."):
            update_json, raw_text = parse_disaster_update_with_qwen(
                update_text,
                st.session_state.scenario,
                str(text_model_config.get("api_key") or ""),
                model=str(text_model_config.get("model") or "qwen-max"),
                endpoint=text_model_config.get("endpoint"),
            )
    except QwenApiError as exc:
        update_json = _fallback_update_json_from_text(update_text, st.session_state.scenario)
        raw_text = f"本地关键词 fallback：{exc}"
        parser_source = "本地关键词 fallback"

    update_json, changed_by_local_repair = _repair_update_json_with_local_hints(
        update_json,
        update_text,
        st.session_state.scenario,
    )
    if changed_by_local_repair and parser_source.endswith("update_json"):
        parser_source = f"{parser_source} + 本地规则校正"

    st.session_state.qwen_update_json = update_json
    st.session_state.qwen_update_raw_text = raw_text
    st.session_state.scenario = apply_update_json_to_scenario(
        st.session_state.scenario,
        update_json,
    )
    st.session_state.last_update_summary = summarize_update_json(update_json)
    _replan_current_scenario(
        f"{parser_source} 已应用：{st.session_state.last_update_summary}；系统已自动重新规划路线。"
    )


def _load_scene_from_image() -> None:
    api_key = _current_api_key()
    if not api_key:
        st.session_state.status_message = "图片识别需要 DASHSCOPE_API_KEY；主流程可直接使用预设场景。"
        return

    try:
        image_bytes, mime_type, source_name = _get_image_payload()
    except FileNotFoundError as exc:
        st.session_state.status_message = str(exc)
        return

    try:
        with st.spinner("正在调用 Qwen-VL 生成 24×24 场景..."):
            recognition, raw_text = recognize_disaster_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                api_key=api_key,
                model=st.session_state.qwen_vl_model,
                image_mode=_qwen_image_mode_value(),
            )
            scenario = merge_recognition_into_scenario(
                load_scenario(SCENARIO_PATH),
                recognition,
                update_map=True,
            )
    except QwenApiError as exc:
        st.session_state.status_message = f"Qwen-VL 图片识别失败：{exc}"
        return

    st.session_state.scenario = scenario
    st.session_state.scenario_source = f"Qwen-VL 图片识别：{source_name}"
    st.session_state.scenario_seed = "-"
    _clear_previous_plan_snapshot()
    st.session_state.qwen_raw_json = recognition
    st.session_state.qwen_raw_text = raw_text
    st.session_state.last_update_summary = "图片识别生成 24×24 场景"
    _replan_current_scenario("已从图片生成 24×24 场景，并完成推理、分配和路线规划。")


def _simulate_collapse() -> None:
    _capture_previous_plan_snapshot("模拟道路塌方前")
    st.session_state.scenario = apply_road_collapse(st.session_state.scenario)
    st.session_state.last_update_summary = "塌方风险格已转为不可通行道路"
    _replan_current_scenario("已触发动态重规划：塌方路段变为不可通行，路线已刷新。")


def _run_full_demo() -> None:
    st.session_state.scenario = load_scenario(SCENARIO_PATH)
    st.session_state.scenario_source = "预设一键演示"
    st.session_state.scenario_seed = "-"
    st.session_state.qwen_raw_json = {}
    st.session_state.qwen_raw_text = ""
    st.session_state.qwen_update_json = {}
    st.session_state.qwen_update_raw_text = ""

    _replan_current_scenario("一键演示：已加载预设场景并生成初始救援方案。")
    _capture_previous_plan_snapshot("一键演示：道路塌方前")

    st.session_state.scenario = apply_road_collapse(st.session_state.scenario)
    st.session_state.last_update_summary = (
        "加载预设场景 -> 概率推理 -> 任务分配 -> 路线规划 -> 模拟塌方 -> 动态重规划"
    )
    _replan_current_scenario(
        "一键演示完整流程已完成：已加载场景、完成推理分配、规划路线、模拟道路塌方并刷新报告。"
    )


def _generate_report() -> None:
    _ensure_full_plan()
    template_report = generate_report(
        st.session_state.scenario,
        st.session_state.zone_scores,
        st.session_state.assignments,
        st.session_state.routes,
        st.session_state.route_details,
    )

    text_model_config = _current_text_model_config()
    if st.session_state.qwen_report_enabled and _text_model_ready(text_model_config):
        try:
            with st.spinner(f"正在调用{text_model_config['backend']}生成救援报告..."):
                st.session_state.report_text = generate_qwen_report(
                    st.session_state.scenario,
                    st.session_state.zone_scores,
                    st.session_state.assignments,
                    st.session_state.routes,
                    str(text_model_config.get("api_key") or ""),
                    model=str(text_model_config.get("model") or "qwen-max"),
                    endpoint=text_model_config.get("endpoint"),
                    route_details=st.session_state.route_details,
                )
            st.session_state.report_source = str(text_model_config["backend"])
            st.session_state.status_message = f"{text_model_config['backend']}救援报告已生成。"
            return
        except QwenApiError as exc:
            st.session_state.report_text = template_report
            st.session_state.report_source = "模板 fallback"
            st.session_state.status_message = f"{text_model_config['backend']}报告生成失败，已回退模板报告：{exc}"
            return

    st.session_state.report_text = template_report
    st.session_state.report_source = "模板 fallback"
    if st.session_state.qwen_report_enabled:
        st.session_state.status_message = "文本模型未配置完整，已使用模板报告。"
    else:
        st.session_state.status_message = "救援报告已生成。"


def _replan_current_scenario(status_message: str) -> None:
    plan = _compute_plan_with_selected_engine(st.session_state.scenario)
    st.session_state.zone_scores = plan["zone_scores"]
    st.session_state.assignments = plan["assignments"]
    st.session_state.routes = plan["routes"]
    st.session_state.route_details = plan["route_details"]
    st.session_state.engine_summary = plan["engine_summary"]
    st.session_state.engine_status = plan["engine_status"]
    st.session_state.report_text = generate_report(
        st.session_state.scenario,
        st.session_state.zone_scores,
        st.session_state.assignments,
        st.session_state.routes,
        st.session_state.route_details,
    )
    st.session_state.report_source = "模板 fallback"
    st.session_state.status_message = _status_with_engine(status_message)
    _sync_current_scenario_export()


def _ensure_full_plan() -> None:
    if (
        not st.session_state.zone_scores
        or not st.session_state.assignments
        or not st.session_state.routes
    ):
        _replan_current_scenario("已补全当前救援方案。")


def _compute_plan_with_selected_engine(scenario: dict[str, Any]) -> dict[str, Any]:
    if st.session_state.get("algorithm_engine") == ENGINE_A_ADAPTER:
        try:
            result = run_a_engine_on_grid(scenario)
            return {
                "zone_scores": result["zone_scores"],
                "assignments": result["assignments"],
                "routes": result["routes"],
                "route_details": result["route_details"],
                "engine_summary": result["engine_summary"],
                "engine_status": (
                    "A 同学算法适配已启用：B 端区块地图已转换为图结构，"
                    "并调用 A 的贝叶斯推理、期望效用分配和风险感知 A*。"
                ),
            }
        except AEngineUnavailable as exc:
            fallback = _compute_demo_plan(scenario)
            fallback["engine_status"] = (
                f"A 同学算法适配失败，已回退 B 演示引擎：{exc}"
            )
            fallback["engine_summary"] = {
                "engine": ENGINE_DEMO,
                "warning": str(exc),
            }
            return fallback

    return _compute_demo_plan(scenario)


def _compute_demo_plan(scenario: dict[str, Any]) -> dict[str, Any]:
    zone_scores = compute_zone_scores(scenario)
    assignments = assign_tasks(scenario, zone_scores)
    routes = plan_routes(scenario, assignments)
    route_details = {
        unit: {
            "engine": ENGINE_DEMO,
            "total_cost": calculate_route_cost(scenario, unit, route),
            "route_layer": "ground" if scenario["units"][unit]["type"] == "car" else "air",
        }
        for unit, route in routes.items()
    }
    return {
        "zone_scores": zone_scores,
        "assignments": assignments,
        "routes": routes,
        "route_details": route_details,
        "engine_summary": {
            "engine": ENGINE_DEMO,
            "note": "B 端内置临时逻辑，用于稳定演示闭环。",
        },
        "engine_status": "当前使用 B 演示引擎。",
    }


def _status_with_engine(status_message: str) -> str:
    engine_status = st.session_state.get("engine_status", "")
    if not engine_status:
        return status_message
    return f"{status_message}（{engine_status}）"


def _capture_previous_plan_snapshot(label: str) -> None:
    _ensure_full_plan()
    st.session_state.previous_scenario = copy.deepcopy(st.session_state.scenario)
    st.session_state.previous_zone_scores = copy.deepcopy(st.session_state.zone_scores)
    st.session_state.previous_assignments = copy.deepcopy(st.session_state.assignments)
    st.session_state.previous_routes = copy.deepcopy(st.session_state.routes)
    st.session_state.previous_route_details = copy.deepcopy(st.session_state.route_details)
    st.session_state.previous_snapshot_label = label


def _clear_previous_plan_snapshot() -> None:
    st.session_state.previous_scenario = {}
    st.session_state.previous_zone_scores = {}
    st.session_state.previous_assignments = {}
    st.session_state.previous_routes = {}
    st.session_state.previous_route_details = {}
    st.session_state.previous_snapshot_label = ""


def _current_api_key() -> str:
    return st.session_state.get("qwen_api_key", "") or os.getenv("DASHSCOPE_API_KEY", "")


def _current_text_model_config() -> dict[str, str | None]:
    if st.session_state.get("qwen_text_backend") == "本地 Qwen 7B":
        return {
            "backend": "本地 Qwen 7B",
            "api_key": st.session_state.get("local_qwen_api_key", ""),
            "model": st.session_state.get("local_qwen_model", "").strip()
            or "qwen2.5-7b-instruct",
            "endpoint": st.session_state.get("local_qwen_endpoint", "").strip(),
        }
    return {
        "backend": "千问 API",
        "api_key": _current_api_key(),
        "model": st.session_state.get("qwen_text_model", "qwen-max"),
        "endpoint": None,
    }


def _text_model_ready(config: dict[str, str | None]) -> bool:
    if config["backend"] == "本地 Qwen 7B":
        return bool(config.get("endpoint"))
    return bool(config.get("api_key"))


def _qwen_image_mode_value() -> str:
    if st.session_state.qwen_image_mode == "实验功能：真实图片抽象为24×24网格":
        return "photo_to_grid"
    return "schematic"


def _image_mode_index(mode: str) -> int:
    options = ["标准网格图识别", "实验功能：真实图片抽象为24×24网格"]
    return options.index(mode) if mode in options else 0


def _option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _parse_random_seed(value: Any) -> int | str | None:
    seed_text = str(value).strip()
    if not seed_text:
        return None
    try:
        return int(seed_text)
    except ValueError:
        return seed_text


def _get_image_payload() -> tuple[bytes, str, str]:
    if st.session_state.uploaded_image_bytes is not None:
        return (
            st.session_state.uploaded_image_bytes,
            st.session_state.uploaded_image_mime,
            st.session_state.uploaded_image_name or "上传图片",
        )
    if not DEFAULT_IMAGE_PATH.exists():
        raise FileNotFoundError("没有上传图片，也找不到内置灾区网格图。")
    return DEFAULT_IMAGE_PATH.read_bytes(), "image/png", "内置灾区网格图"


def _current_scenario_json_text() -> str:
    return json.dumps(st.session_state.scenario, ensure_ascii=False, indent=2)


def _current_scenario_export_name() -> str:
    source = _safe_filename_part(st.session_state.get("scenario_source", "scenario"))
    seed = str(st.session_state.get("scenario_seed", "")).strip()
    seed_part = f"_seed_{_safe_filename_part(seed)}" if seed and seed != "-" else ""
    return f"scenario_{source}{seed_part}.json"


def _safe_filename_part(value: str) -> str:
    cleaned = "".join(
        char if char.isascii() and (char.isalnum() or char in ("-", "_")) else "_"
        for char in value.strip()
    ).strip("_")
    return cleaned or "current"


def _sync_current_scenario_export() -> None:
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_SCENARIO_EXPORT_PATH.write_text(
            _current_scenario_json_text(),
            encoding="utf-8",
        )
    except OSError as exc:
        st.session_state.scenario_export_error = f"自动保存 JSON 失败：{exc}"
        return

    st.session_state.scenario_export_path = str(CURRENT_SCENARIO_EXPORT_PATH)
    st.session_state.scenario_export_error = ""


def _render_current_scenario_preview() -> None:
    with st.expander("当前场景 JSON 预览（临时算法输入）", expanded=False):
        st.caption("当前预览保持 data/scenario.json 的结构；后续可按组员要求改成新的算法输入 schema。")
        st.code(_current_scenario_json_text(), language="json")


def _render_last_update_summary() -> None:
    summary = st.session_state.get("last_update_summary", "")
    if not summary:
        return
    st.success(f"本次变更：{summary}")


def _render_summary_metrics() -> None:
    map_data = st.session_state.scenario["map"]
    leader = "-"
    if st.session_state.zone_scores:
        leader_name, leader_scores = max(
            st.session_state.zone_scores.items(),
            key=lambda item: item[1]["priority"],
        )
        leader = f"{leader_name}区 / {leader_scores['priority']:.1f}"

    cols = st.columns(7)
    cols[0].metric("场景来源", st.session_state.scenario_source)
    cols[1].metric("seed", st.session_state.scenario_seed)
    cols[2].metric("算法引擎", st.session_state.get("algorithm_engine", ENGINE_DEMO))
    cols[3].metric("最高优先级", leader)
    cols[4].metric("断路格", len(map_data.get("blocked", [])))
    cols[5].metric("火灾格", len(map_data.get("fire", [])))
    cols[6].metric("塌方风险格", len(map_data.get("collapse_cells", [])))


def _render_scores(zone_scores: dict[str, dict[str, float]]) -> None:
    if not zone_scores:
        st.warning("尚未执行智能推理。")
        return

    rows = []
    for zone, scores in zone_scores.items():
        rows.append(
            {
                "区域": f"{zone}区",
                "被困概率": scores["trapped_probability"],
                "道路可通行概率": scores["road_accessibility"],
                "生命风险": scores["life_risk"],
                "紧迫度": scores["urgency"],
                "优先级": scores["priority"],
            }
        )
    df = pd.DataFrame(rows).sort_values("优先级", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_assignments(
    assignments: dict[str, str],
    routes: dict[str, list[list[int]]],
    scenario: dict[str, Any] | None = None,
    route_details: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not assignments:
        st.warning("尚未规划救援路线。")
        return

    rows = []
    for unit, zone in assignments.items():
        route = routes.get(unit, [])
        row = {
            "救援单位": unit,
            "任务": _display_assignment_task(unit, zone),
            "路线长度": max(len(route) - 1, 0),
        }
        if scenario:
            row["路线代价"] = _route_cost_for_display(
                scenario,
                unit,
                route,
                route_details,
            )
            detail = (route_details or {}).get(unit, {})
            if detail.get("path_risk") is not None:
                row["路径风险"] = detail["path_risk"]
            if detail.get("expected_utility") is not None:
                row["期望效用"] = detail["expected_utility"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_previous_plan_snapshot() -> None:
    st.subheader("重规划前快照")
    if not st.session_state.previous_zone_scores:
        st.caption("暂无重规划前数据。执行灾情更新或模拟塌方后，这里会保留更新前的优先级和任务分配。")
        return

    label = st.session_state.previous_snapshot_label or "重规划前"
    st.markdown("**重规划前后对比**")
    _render_replan_comparison()
    with st.expander(label, expanded=True):
        st.markdown("**重规划前优先级**")
        _render_scores(st.session_state.previous_zone_scores)
        st.markdown("**重规划前任务分配**")
        _render_assignments(
            st.session_state.previous_assignments,
            st.session_state.previous_routes,
            st.session_state.previous_scenario,
            st.session_state.previous_route_details,
        )


def _render_replan_comparison() -> None:
    previous_scores = st.session_state.previous_zone_scores
    current_scores = st.session_state.zone_scores
    if not previous_scores or not current_scores:
        return

    score_rows = []
    for zone in sorted(set(previous_scores) | set(current_scores)):
        before = previous_scores.get(zone, {})
        after = current_scores.get(zone, {})
        before_priority = before.get("priority", 0.0)
        after_priority = after.get("priority", 0.0)
        score_rows.append(
            {
                "区域": f"{zone}区",
                "优先级(前)": before_priority,
                "优先级(后)": after_priority,
                "变化": round(after_priority - before_priority, 1),
                "生命风险(前)": before.get("life_risk", 0.0),
                "生命风险(后)": after.get("life_risk", 0.0),
            }
        )
    st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

    route_rows = []
    units = sorted(
        set(st.session_state.previous_assignments)
        | set(st.session_state.assignments)
    )
    for unit in units:
        before_route = st.session_state.previous_routes.get(unit, [])
        after_route = st.session_state.routes.get(unit, [])
        route_rows.append(
            {
                "单位": unit,
                "任务(前)": _display_assignment_task(
                    unit,
                    st.session_state.previous_assignments.get(unit, "-"),
                ),
                "任务(后)": _display_assignment_task(
                    unit,
                    st.session_state.assignments.get(unit, "-"),
                ),
                "长度(前)": max(len(before_route) - 1, 0),
                "长度(后)": max(len(after_route) - 1, 0),
                "代价(前)": _route_cost_for_display(
                    st.session_state.previous_scenario,
                    unit,
                    before_route,
                    st.session_state.previous_route_details,
                )
                if before_route
                else 0.0,
                "代价(后)": _route_cost_for_display(
                    st.session_state.scenario,
                    unit,
                    after_route,
                    st.session_state.route_details,
                )
                if after_route
                else 0.0,
            }
        )
    st.dataframe(pd.DataFrame(route_rows), hide_index=True, use_container_width=True)


def _render_route_summary(
    assignments: dict[str, str],
    routes: dict[str, list[list[int]]],
) -> None:
    st.subheader("路线摘要")
    if not routes:
        st.warning("尚未生成路线。")
        return

    rows = []
    for unit, route in routes.items():
        target = assignments.get(unit, "-")
        rows.append(
            {
                "单位": unit,
                "任务": _display_assignment_task(unit, target),
                "起点": _format_point(route[0]) if route else "-",
                "终点": _format_point(route[-1]) if route else "-",
                "长度": max(len(route) - 1, 0),
                "路线代价": _route_cost_for_display(
                    st.session_state.scenario,
                    unit,
                    route,
                    st.session_state.route_details,
                ),
                "路径风险": st.session_state.route_details.get(unit, {}).get("path_risk", "-"),
                "ETA": st.session_state.route_details.get(unit, {}).get("eta", "-"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    _render_algorithm_engine_panel()


def _route_cost_for_display(
    scenario: dict[str, Any],
    unit: str,
    route: list[list[int]],
    route_details: dict[str, dict[str, Any]] | None = None,
) -> float:
    detail = (route_details or {}).get(unit, {})
    if isinstance(detail.get("total_cost"), (int, float)):
        return round(float(detail["total_cost"]), 2)
    return calculate_route_cost(scenario, unit, route)


def _render_algorithm_engine_panel() -> None:
    summary = st.session_state.get("engine_summary", {})
    route_details = st.session_state.get("route_details", {})
    with st.expander("算法引擎与适配说明", expanded=False):
        st.caption(st.session_state.get("engine_status", ""))
        if summary:
            summary_rows = [
                {"项目": key, "值": value}
                for key, value in summary.items()
                if key != "note"
            ]
            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=220,
                )
            if summary.get("note"):
                st.info(str(summary["note"]))

        if route_details:
            rows = []
            for unit, detail in route_details.items():
                rows.append(
                    {
                        "单位": unit,
                        "算法": detail.get("engine", "-"),
                        "图层": detail.get("route_layer", "-"),
                        "总代价": detail.get("total_cost", "-"),
                        "路径风险": detail.get("path_risk", "-"),
                        "ETA": detail.get("eta", "-"),
                        "A*扩展节点": detail.get("expanded_nodes", "-"),
                        "期望效用": detail.get("expected_utility", "-"),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                height=190,
            )


def _render_map_tile_legend() -> None:
    st.markdown(
        """
        <div class="map-legend-panel">
          <div class="legend-title">地图区块说明</div>
          <div class="tile-legend-grid">
        """
        + "\n".join(
            _tile_legend_item_html(category)
            for category in (0, 1, 2, 7, 8, 3, 4, 5, 6)
        )
        + """
          </div>
          <div class="cost-rule">
            <b>地面代价：</b>救援车道路=1.0，草地/空地/绿地=1.8；拥堵 +3.5，火灾 +5.0，塌方风险 +4.0；建筑、水域、断路对救援车不可通行。<br>
            <b>空中代价：</b>无人机使用 8 邻接空中格网，直飞单格=1.0、斜飞约=1.41；可飞越建筑/水域/断路，障碍 +0.3，拥堵 +0.35，塌方 +1.1，火灾 +2.4。
          </div>
          <div class="legend-title legend-title-spaced">标记与路线</div>
          <div class="marker-legend-grid">
            <span><b class="marker-dot" style="background:#2f4858">S</b> 救援中心/出发点</span>
            <span><b class="marker-dot" style="background:#6a4c93">H</b> 医院/安全点</span>
            <span><b class="marker-dot" style="background:#c1121f">A</b> A/B/C 灾情目标区</span>
            <span><b class="line-sample" style="background:#d62728"></b> RescueCar-1 路线</span>
            <span><b class="line-sample" style="background:#1f77b4"></b> RescueCar-2 路线</span>
            <span><b class="line-sample" style="background:#2ca02c"></b> Drone-1 空中侦查路线</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tile_legend_item_html(category: int) -> str:
    meta = TILE_CATEGORY_META[category]
    return (
        '<div class="tile-legend-item">'
        f'<span class="tile-swatch" style="background:{meta["color"]}"></span>'
        f'<span class="tile-copy"><b>{meta["name"]}</b><small>{meta["description"]}</small></span>'
        "</div>"
    )


def _render_debug_outputs() -> None:
    if st.session_state.qwen_update_json:
        with st.expander("Qwen 自然语言更新 JSON", expanded=False):
            st.code(
                json.dumps(st.session_state.qwen_update_json, ensure_ascii=False, indent=2),
                language="json",
            )
            if st.session_state.qwen_update_raw_text:
                st.text_area(
                    "原始解析输出",
                    value=st.session_state.qwen_update_raw_text,
                    height=160,
                    disabled=True,
                )

    if st.session_state.qwen_raw_json:
        with st.expander("Qwen-VL 图片识别 JSON", expanded=False):
            st.code(
                json.dumps(st.session_state.qwen_raw_json, ensure_ascii=False, indent=2),
                language="json",
            )
            if st.session_state.qwen_raw_text:
                st.text_area(
                    "原始模型输出",
                    value=st.session_state.qwen_raw_text,
                    height=180,
                    disabled=True,
                )


def _build_map_figure(
    scenario: dict[str, Any],
    routes: dict[str, list[list[int]]],
    assignments: dict[str, str],
) -> go.Figure:
    map_data = scenario["map"]
    width = map_data["width"]
    height = map_data["height"]
    grid = [[0 for _ in range(width)] for _ in range(height)]

    for x, y in map_data.get("park", []):
        grid[y][x] = 8
    for x, y in map_data.get("water", []):
        grid[y][x] = 7
    for x, y in map_data.get("buildings", []):
        grid[y][x] = 2
    for x, y in map_data.get("roads", []):
        grid[y][x] = 1
    for x, y in map_data.get("congestion", []):
        grid[y][x] = 3
    for x, y in map_data.get("fire", []):
        grid[y][x] = 4
    for x, y in map_data.get("blocked", []):
        grid[y][x] = 5
    for x, y in map_data.get("collapse_cells", []):
        grid[y][x] = 6

    visual_grid, visual_x, visual_y, hover_cells = _build_visual_grid(grid, MAP_RENDER_SCALE)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=visual_grid,
            x=visual_x,
            y=visual_y,
            customdata=hover_cells,
            colorscale=_discrete_colorscale(PIXEL_TILE_COLORS),
            zmin=0,
            zmax=len(PIXEL_TILE_COLORS) - 1,
            showscale=False,
            xgap=0,
            ygap=0,
            hovertemplate=(
                "坐标=(%{customdata[0]}, %{customdata[1]})"
                "<br>区块=%{customdata[2]}"
                "<br>%{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

    for unit, path in routes.items():
        if not path:
            continue
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        route_length = max(len(path) - 1, 0)
        route_cost = _route_cost_for_display(
            scenario,
            unit,
            path,
            st.session_state.route_details,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=_display_route_label(unit, assignments.get(unit, "-")),
                customdata=[[route_length, route_cost] for _ in path],
                line={
                    "color": ROUTE_COLORS.get(unit, "#555555"),
                    "width": 2.8,
                },
                marker={
                    "size": 5.5,
                    "color": ROUTE_COLORS.get(unit, "#555555"),
                    "line": {"width": 1.2, "color": "white"},
                },
                hovertemplate=(
                    f"{_display_route_label(unit, assignments.get(unit, '-'))}"
                    "<br>路线节点=(%{x}, %{y})"
                    "<br>路线长度=%{customdata[0]} 格"
                    "<br>路线代价=%{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    _add_marker(fig, map_data["base"], "救援中心", "#2f4858", "S")
    _add_marker(fig, map_data["hospital"], "医院", "#6a4c93", "H")
    for zone, target in map_data["targets"].items():
        _add_marker(fig, target, f"{zone}区", "#c1121f", zone)

    unit_offsets = {
        "RescueCar-1": (-0.18, -0.18),
        "RescueCar-2": (0.18, -0.18),
        "Drone-1": (0.18, 0.18),
    }
    unit_labels = {
        "RescueCar-1": "R1",
        "RescueCar-2": "R2",
        "Drone-1": "D",
    }
    for unit, detail in scenario["units"].items():
        point = _offset_point(detail["start"], unit_offsets.get(unit, (0.0, 0.0)))
        _add_marker(
            fig,
            point,
            unit,
            ROUTE_COLORS.get(unit, "#333333"),
            unit_labels.get(unit, unit[:2]),
            size=26,
        )

    fig.update_layout(
        height=820,
        margin={"l": 12, "r": 12, "t": 16, "b": 12},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0,
            "font": {"size": 12},
        },
        xaxis={
            "range": [-0.5, width - 0.5],
            "dtick": 1,
            "showgrid": True,
            "gridcolor": "rgba(50, 60, 70, 0.28)",
            "gridwidth": 1,
            "zeroline": False,
            "title": "",
            "showline": True,
            "linecolor": "#b9c3d0",
            "mirror": True,
        },
        yaxis={
            "range": [-0.5, height - 0.5],
            "dtick": 1,
            "showgrid": True,
            "gridcolor": "rgba(50, 60, 70, 0.28)",
            "gridwidth": 1,
            "zeroline": False,
            "title": "",
            "scaleanchor": "x",
            "showline": True,
            "linecolor": "#b9c3d0",
            "mirror": True,
        },
    )
    return fig


def _build_visual_grid(
    grid: list[list[int]], scale: int
) -> tuple[list[list[int]], list[float], list[float], list[list[list[Any]]]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    visual_grid: list[list[int]] = []
    hover_cells: list[list[list[Any]]] = []

    for y, row in enumerate(grid):
        for sy in range(scale):
            visual_row: list[int] = []
            hover_row: list[list[Any]] = []
            for x, value in enumerate(row):
                meta = TILE_CATEGORY_META.get(value, TILE_CATEGORY_META[0])
                for sx in range(scale):
                    visual_row.append(_textured_tile_code(value, x, y, sx, sy))
                    hover_row.append([x, y, meta["name"], meta["description"]])
            visual_grid.append(visual_row)
            hover_cells.append(hover_row)

    visual_x = [-0.5 + (index + 0.5) / scale for index in range(width * scale)]
    visual_y = [-0.5 + (index + 0.5) / scale for index in range(height * scale)]
    return visual_grid, visual_x, visual_y, hover_cells


def _routes_cross_forbidden_cells(
    scenario: dict[str, Any],
    routes: dict[str, list[list[int]]],
) -> bool:
    if not routes:
        return False
    map_data = scenario.get("map", {})
    forbidden = {
        tuple(cell)
        for key in ("blocked", "buildings", "water")
        for cell in map_data.get(key, [])
    }
    for unit, path in routes.items():
        if unit == "Drone-1":
            continue
        if any(tuple(point) in forbidden for point in path):
            return True
    return False


def _textured_tile_code(category: int, x: int, y: int, sx: int, sy: int) -> int:
    variants = TILE_VARIANTS.get(category, TILE_VARIANTS[0])
    index = (x * 17 + y * 31 + sx * 7 + sy * 11) % len(variants)
    return variants[index]


def _discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    max_index = len(colors) - 1
    if max_index <= 0:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    scale: list[list[float | str]] = []
    for index, color in enumerate(colors):
        left = max(0.0, (index - 0.5) / max_index)
        right = min(1.0, (index + 0.5) / max_index)
        scale.append([left, color])
        scale.append([right, color])
    return scale


def _add_marker(
    fig: go.Figure,
    point: list[float],
    name: str,
    color: str,
    label: str,
    size: int = 30,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[point[0]],
            y=[point[1]],
            mode="markers+text",
            name=name,
            text=[label],
            textposition="middle center",
            marker={
                "size": size,
                "color": color,
                "line": {"width": 2.5, "color": "white"},
            },
            textfont={"color": "white", "size": 14},
            hovertemplate=f"{name}<br>x=%{{x}}, y=%{{y}}<extra></extra>",
        )
    )


def _fallback_update_json_from_text(
    text: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    targets = _targets_from_text(text)
    target_updates: list[dict[str, Any]] = []
    cell_updates: list[dict[str, Any]] = []

    for target in targets:
        fields: dict[str, float] = {}

        if _zone_has_fire_update(text, target):
            fire_level = 0.98 if _zone_has_intensified_fire(text, target) else 0.9
            fields.update({"fire": fire_level, "smoke": 0.9, "urgency": 0.9})
            cell_updates.append(
                {
                    "type": "add_fire_cells",
                    "cells": _nearby_cells(scenario, target, count=2),
                }
            )

        if any(word in text for word in ("SOS", "求救", "被困", "生命体征")):
            fields.update({"sos_signal": 0.95, "human_activity": 0.75, "urgency": 0.9})

        if any(word in text for word in ("拥堵", "堵塞", "车流")):
            fields.update({"congestion": 0.85})
            cell_updates.append(
                {
                    "type": "add_congestion_cells",
                    "cells": _nearby_cells(scenario, target, count=2),
                }
            )

        if any(word in text for word in ("道路可以通行", "道路恢复", "可通行", "打通")):
            fields.update({"road_damage": 0.25})
            blocked_cell = _nearest_existing_cell(scenario, "blocked", target)
            collapse_cell = _nearest_existing_cell(scenario, "collapse_cells", target)
            if blocked_cell:
                cell_updates.append({"type": "remove_blocked_cells", "cells": [blocked_cell]})
            if collapse_cell:
                cell_updates.append({"type": "remove_collapse_cells", "cells": [collapse_cell]})

        if any(word in text for word in ("塌方", "断裂", "断路", "新增障碍")):
            fields.update({"road_damage": 0.85, "building_collapse": 0.85})
            cell_updates.append(
                {
                    "type": "add_collapse_cells",
                    "cells": _nearby_cells(scenario, target, count=2),
                }
            )

        if fields:
            target_updates.append({"type": "target_update", "target": target, "fields": fields})

    return {"updates": target_updates + cell_updates}


def _repair_update_json_with_local_hints(
    update_json: dict[str, Any],
    text: str,
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    local_json = _fallback_update_json_from_text(text, scenario)
    if not local_json.get("updates"):
        return update_json, False

    before = json.dumps(update_json, ensure_ascii=False, sort_keys=True)
    repaired = json.loads(json.dumps(update_json, ensure_ascii=False))
    repaired.setdefault("updates", [])

    existing_target_updates: dict[str, dict[str, Any]] = {}
    for update in repaired["updates"]:
        if isinstance(update, dict) and update.get("type") == "target_update":
            target = update.get("target")
            if target in ("A", "B", "C"):
                update.setdefault("fields", {})
                existing_target_updates[target] = update

    existing_cell_updates = {
        (
            update.get("type"),
            tuple(tuple(cell) for cell in update.get("cells", []))
            if isinstance(update.get("cells"), list)
            else (),
        )
        for update in repaired["updates"]
        if isinstance(update, dict) and update.get("type") != "target_update"
    }

    for local_update in local_json.get("updates", []):
        if not isinstance(local_update, dict):
            continue
        if local_update.get("type") == "target_update":
            target = local_update.get("target")
            fields = local_update.get("fields", {})
            if target not in ("A", "B", "C") or not isinstance(fields, dict):
                continue
            target_update = existing_target_updates.get(target)
            if target_update is None:
                target_update = {"type": "target_update", "target": target, "fields": {}}
                repaired["updates"].append(target_update)
                existing_target_updates[target] = target_update
            for key, value in fields.items():
                if isinstance(value, (int, float)):
                    current = target_update["fields"].get(key)
                    if not isinstance(current, (int, float)) or value > current:
                        target_update["fields"][key] = value
        else:
            cells = local_update.get("cells", [])
            signature = (
                local_update.get("type"),
                tuple(tuple(cell) for cell in cells) if isinstance(cells, list) else (),
            )
            if signature not in existing_cell_updates:
                repaired["updates"].append(local_update)
                existing_cell_updates.add(signature)

    after = json.dumps(repaired, ensure_ascii=False, sort_keys=True)
    return repaired, before != after


def _targets_from_text(text: str) -> list[str]:
    targets = [
        zone
        for zone in ("A", "B", "C")
        if f"{zone}区" in text or f"{zone} 区" in text
    ]
    if targets:
        return targets
    if any(word in text for word in ("火势", "火灾", "浓烟")):
        return ["C"]
    return ["A"]


def _zone_has_fire_update(text: str, target: str) -> bool:
    if not any(word in text for word in ("火势", "起火", "火灾", "烟雾", "浓烟")):
        return False
    if f"{target}区" in text or f"{target} 区" in text:
        return True
    return len(_targets_from_text(text)) == 1


def _zone_has_intensified_fire(text: str, target: str) -> bool:
    zone_forms = (f"{target}区", f"{target} 区")
    intensity_words = ("扩大", "加剧", "蔓延", "严重")
    for zone_form in zone_forms:
        zone_index = text.find(zone_form)
        if zone_index == -1:
            continue
        segment_end = len(text)
        for other_zone in ("A", "B", "C"):
            if other_zone == target:
                continue
            for other_form in (f"{other_zone}区", f"{other_zone} 区"):
                other_index = text.find(other_form, zone_index + len(zone_form))
                if other_index != -1:
                    segment_end = min(segment_end, other_index)
        segment = text[zone_index:segment_end]
        if any(word in segment for word in intensity_words):
            return True
        for word in intensity_words:
            word_index = text.find(word, zone_index + len(zone_form))
            if word_index == -1:
                continue
            between = text[zone_index:word_index]
            if not any(mark in between for mark in "，,。；;！!？?\n"):
                return True
    return False


def _nearby_cells(scenario: dict[str, Any], target: str, count: int) -> list[list[int]]:
    map_data = scenario["map"]
    width = int(map_data.get("width", 24))
    height = int(map_data.get("height", 24))
    x, y = map_data.get("targets", {}).get(target, [5, 5])
    reserved = {
        tuple(map_data.get("base", [0, 0])),
        tuple(map_data.get("hospital", [9, 9])),
        *(tuple(point) for point in map_data.get("targets", {}).values()),
    }
    candidates = [
        (x, y + 1),
        (x + 1, y),
        (x - 1, y),
        (x, y - 1),
        (x + 1, y + 1),
        (x - 1, y + 1),
    ]
    cells: list[list[int]] = []
    for cx, cy in candidates:
        if 0 <= cx < width and 0 <= cy < height and (cx, cy) not in reserved:
            cells.append([cx, cy])
        if len(cells) >= count:
            break
    return cells


def _nearest_existing_cell(
    scenario: dict[str, Any],
    field: str,
    target: str,
) -> list[int]:
    map_data = scenario["map"]
    cells = map_data.get(field, [])
    if not cells:
        return []
    target_point = map_data.get("targets", {}).get(target, [5, 5])
    nearest = min(
        cells,
        key=lambda cell: abs(cell[0] - target_point[0]) + abs(cell[1] - target_point[1]),
    )
    return [int(nearest[0]), int(nearest[1])]


def _offset_point(point: list[int], offset: tuple[float, float]) -> list[float]:
    return [point[0] + offset[0], point[1] + offset[1]]


def _format_point(point: list[int]) -> str:
    return f"({point[0]}, {point[1]})"


def _display_assignment_task(unit: str, zone: str) -> str:
    if unit == "Drone-1":
        return "无人机侦查"
    return f"{zone}区救援"


def _display_route_label(unit: str, zone: str) -> str:
    if unit == "Drone-1":
        return "Drone-1 -> 无人机侦查"
    return f"{unit} -> {zone}区"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        textarea {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif !important;
            line-height: 1.7 !important;
        }
        .map-legend-panel {
            border: 1px solid #d8dee8;
            background: #ffffff;
            border-radius: 8px;
            padding: 12px 14px 14px;
            margin: -0.4rem 0 1rem;
        }
        .legend-title {
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .legend-title-spaced {
            margin-top: 12px;
        }
        .cost-rule {
            margin-top: 10px;
            padding: 8px 10px;
            border-radius: 6px;
            background: #f8fafc;
            color: #344054;
            border: 1px solid #e4e7ec;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .tile-legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 8px 12px;
        }
        .tile-legend-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            min-width: 0;
        }
        .tile-swatch {
            width: 22px;
            height: 22px;
            border-radius: 4px;
            border: 1px solid rgba(31, 41, 55, 0.24);
            flex: 0 0 auto;
            margin-top: 1px;
        }
        .tile-copy {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .tile-copy b {
            font-size: 0.9rem;
            color: #1f2937;
            line-height: 1.15;
        }
        .tile-copy small {
            color: #667085;
            line-height: 1.25;
            margin-top: 2px;
        }
        .marker-legend-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 14px;
            color: #344054;
            font-size: 0.9rem;
        }
        .marker-dot {
            display: inline-flex;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 0.75rem;
            margin-right: 5px;
            border: 1px solid #fff;
        }
        .line-sample {
            display: inline-block;
            width: 28px;
            height: 4px;
            border-radius: 999px;
            margin: 0 6px 3px 0;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
