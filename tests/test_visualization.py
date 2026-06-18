import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from emergency_commander.visualization import (
    build_calibration_figure,
    build_map_figure,
    build_metrics_frame,
    build_probability_frame,
    build_utility_contribution_figure,
    build_utility_frame,
)
from emergency_commander.pipeline import run_pipeline
from emergency_commander.random_scenario import generate_random_scenario
from tests.test_pipeline_replanning import scenario_with_collapse


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_visualization_builders_expose_map_metrics_and_calibration_layers():
    scenario = load_json("examples/scenario_input.json")
    output = run_pipeline(scenario)
    metrics = load_json("artifacts/full_bayesian_experiment/experiment_metrics.json")
    snapshot = output["timeline"][1]

    map_figure = build_map_figure(snapshot["scenario_state"], snapshot)
    probability_frame = build_probability_frame(snapshot["plan"])
    metrics_frame = build_metrics_frame(metrics)
    calibration = build_calibration_figure(metrics, "trapped_people")
    initial_matrix = output["timeline"][0]["plan"]["utility_matrix"]
    utility_frame = build_utility_frame(initial_matrix)
    feasible = next(item for item in initial_matrix if item["feasible"])
    utility_figure = build_utility_contribution_figure(feasible)

    trace_names = {trace.name for trace in map_figure.data}
    assert {"无人机航线", "救援单位", "灾区优先级", "阻断道路"} <= trace_names
    assert any(name.endswith("风险道路") for name in trace_names)
    assert set(probability_frame["区域"]) == {"A", "B", "C"}
    assert {"Expert CPT", "Learned CPT"} == set(metrics_frame["模型"])
    assert {trace.name for trace in calibration.data} >= {"Perfect calibration", "Expert CPT", "Learned CPT"}
    assert {"单位", "区域", "可行", "总效用", "资源成本", "原因"} <= set(utility_frame.columns)
    assert {trace.name for trace in utility_figure.data} == {"效用贡献"}


def test_map_renders_complex_scenario_before_inference_and_highlights_focus():
    scenario = generate_random_scenario(20260616)
    snapshot = {
        "plan": {
            "zone_assessment": [],
            "assignments": [],
            "routes": [],
            "utility_matrix": [],
        },
        "unit_states": {},
        "scenario_state": scenario,
    }
    focus = {
        "roads": [scenario["roads"][0]["road_id"]],
        "zones": ["A"],
        "units": ["RescueCar-1"],
    }

    figure = build_map_figure(scenario, snapshot, focus=focus)

    trace_names = [trace.name for trace in figure.data]
    assert "地形网格" in trace_names
    assert "道路骨架" in trace_names
    assert "火势范围" in trace_names
    assert "灾区优先级" in trace_names
    assert "救援单位" in trace_names
    assert "当前计算道路" in trace_names
    assert "当前计算灾区" in trace_names
    assert "当前计算单位" in trace_names
    disaster_trace = next(trace for trace in figure.data if trace.name == "灾区优先级")
    assert len(disaster_trace.x) == len(scenario["zones"])
    unit_trace = next(trace for trace in figure.data if trace.name == "救援单位")
    assert len(unit_trace.x) == 5
    unit_labels = [
        annotation.text
        for annotation in figure.layout.annotations
        if "救援车" in annotation.text or "无人机" in annotation.text
    ]
    assert len(unit_labels) == 1
    assert "救援车1 / 救援车2" in unit_labels[0]
    unit_label = next(
        annotation
        for annotation in figure.layout.annotations
        if "救援车1 / 救援车2" in annotation.text
    )
    unit_start = scenario["nodes"][scenario["units"][0]["start_node"]]
    assert unit_label.x == unit_start["x"]
    assert unit_label.y == unit_start["y"]
    assert unit_label.xshift or unit_label.yshift
    state_grid = next(trace for trace in figure.data if trace.name == "地形网格")
    assert state_grid.type == "heatmap"
    assert len(state_grid.x) * len(state_grid.y) >= 2000
    assert state_grid.x[1] - state_grid.x[0] <= 0.75
    grid_values = {value for row in state_grid.z for value in row}
    assert {3, 4} <= grid_values
    grid_colors = [entry[1] for entry in state_grid.colorscale]
    assert "#df553f" in grid_colors
    assert "#39444c" in grid_colors
    assert len(figure.layout.shapes) == 0
    road_trace = next(trace for trace in figure.data if trace.name == "道路骨架")
    assert road_trace.line.color == "rgba(35,43,49,.50)"
    fire_trace = next(trace for trace in figure.data if trace.name == "火势范围")
    assert "#e3342f" in list(fire_trace.marker.color)
    assert figure.layout.plot_bgcolor == "#cfc3a3"


def test_map_hides_synthetic_position_connectors_from_base_road_layers():
    output = run_pipeline(scenario_with_collapse(), process_events=True)
    snapshot = output["timeline"][1]

    figure = build_map_figure(snapshot["scenario_state"], snapshot)

    base_traces = [
        trace
        for trace in figure.data
        if trace.name == "道路骨架" or str(trace.name).endswith("风险道路")
    ]
    assert base_traces
    assert not any(
        "__unit_" in str(label)
        for trace in base_traces
        for label in (trace.text or [])
        if label
    )


def test_streamlit_demo_starts_without_runtime_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("AI Emergency Commander" in title.value for title in app.title)
    labels = {button.label for button in app.button}
    assert "导入地图模型" in labels
    assert {"道路坍塌", "火势蔓延", "新增求救"} <= labels
    assert "无人机情报" not in labels
    assert any("等待导入地图模型" in item.value for item in app.markdown)


def test_learned_cpt_mode_shows_advantage_metrics_in_console():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    app.radio(key="model_selector").set_value("学习 CPT").run()
    next(
        button for button in app.button if button.label == "导入地图模型"
    ).click().run()

    markdown = "\n".join(item.value for item in app.markdown)
    assert "学习 CPT 优势" in markdown
    assert "被困 F1" in markdown
    assert "道路 ROC-AUC" in markdown


def test_cpt_selector_syncs_existing_session_model():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    next(
        button for button in app.button if button.label == "导入地图模型"
    ).click().run()
    assert app.session_state["live_simulation"]["model_name"] == "expert_cpt"

    app.radio(key="model_selector").set_value("学习 CPT").run()

    assert app.session_state["live_simulation"]["model_name"] == "learned_cpt"
    assert app.session_state["live_simulation"]["scenario"]["run_mode"] == "learned"
    markdown = "\n".join(item.value for item in app.markdown)
    assert "已切换为 学习 CPT" in markdown


def test_generation_creates_a_manual_session_that_advances_one_phase_per_click():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    next(
        button for button in app.button if button.label == "导入地图模型"
    ).click().run()

    assert not app.exception
    session = app.session_state["live_simulation"]
    assert session["status"] == "running"
    assert session["seed"]
    assert session["scenario"]["nodes"]
    assert 4 <= len(session["scenario"]["zones"]) <= 7
    assert session["phase"] == "validate"
    assert session["calculation_history"] == []

    next(
        button for button in app.button if button.label == "执行下一算法步骤"
    ).click().run()

    session = app.session_state["live_simulation"]
    assert session["phase"] == "infer"
    assert len(session["calculation_history"]) == 1
    assert session["calculation_history"][0]["phase"] == "validate"


def test_event_dock_uses_operator_selected_target():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    next(
        button for button in app.button if button.label == "导入地图模型"
    ).click().run()
    for _ in range(6):
        next(
            button for button in app.button if button.label == "执行下一算法步骤"
        ).click().run()

    session = app.session_state["live_simulation"]
    targets = [
        road["road_id"]
        for road in session["scenario"]["roads"]
        if road["status"] == "open"
    ]
    assert len(targets) >= 2

    app.selectbox(key="event_target_road_collapse").select(targets[-1]).run()
    next(button for button in app.button if button.label == "道路坍塌").click().run()

    event = app.session_state["live_simulation"]["event_log"][-1]
    assert event["event_type"] == "road_collapse"
    assert event["target_id"] == targets[-1]
