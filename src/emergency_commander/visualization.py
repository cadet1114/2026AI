from __future__ import annotations

import html
import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from emergency_commander.routing import road_risk


UNIT_COLORS = {
    "RescueCar-1": "#f05a28",
    "RescueCar-2": "#e2b13c",
    "RescueCar-3": "#e84a5f",
    "Drone-1": "#20b8cd",
    "Drone-2": "#5b8ff9",
}

FIRE_STATUS_COLORS = {
    "low": "#f4d35e",
    "medium": "#ff9f1c",
    "high": "#e3342f",
}
FIRE_STATUS_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}
SANDBOX_GRID_CELL_SIZE = 0.42
SANDBOX_STATE_COLORS = {
    0: "#cddbb3",
    1: "#d4c198",
    2: "#e9bd55",
    3: "#df553f",
    4: "#39444c",
    5: "#3a171b",
}
SANDBOX_STATE_LABELS = {
    0: "稳定地面",
    1: "烟雾或轻微风险",
    2: "受损或拥堵区域",
    3: "活跃火点",
    4: "可通行道路",
    5: "阻断道路",
}
NODE_DISPLAY_LABELS = {
    "HQ": "指挥中心",
    "HOSPITAL": "医院",
    "AIR_RELAY": "空中中继",
}
MAP_LABEL_FONT = "Microsoft YaHei, Noto Sans CJK SC, Inter, Arial, sans-serif"
MAP_LABEL_STYLES = {
    "zone": {
        "bgcolor": "rgba(246,240,220,.88)",
        "bordercolor": "rgba(22,30,35,.46)",
        "font_color": "#132029",
    },
    "facility": {
        "bgcolor": "rgba(16,27,36,.84)",
        "bordercolor": "rgba(255,244,218,.50)",
        "font_color": "#fff3d4",
    },
    "unit": {
        "bgcolor": "rgba(15,25,34,.84)",
        "bordercolor": "rgba(255,248,232,.46)",
        "font_color": "#f7f1e4",
    },
}


def _zone_label(zone_id: str) -> str:
    return f"{zone_id}区"


def _node_label(node_id: str) -> str:
    if node_id.startswith("ZONE_"):
        return _zone_label(node_id.removeprefix("ZONE_"))
    return NODE_DISPLAY_LABELS.get(node_id, node_id)


def _unit_label(unit_id: str) -> str:
    return (
        unit_id.replace("RescueCar-", "救援车")
        .replace("Drone-", "无人机")
    )


def _add_map_label(
    figure: go.Figure,
    *,
    x: float,
    y: float,
    text: str,
    kind: str = "zone",
    xshift: int = 0,
    yshift: int = 0,
    size: int = 12,
) -> None:
    style = MAP_LABEL_STYLES[kind]
    escaped_text = "<br>".join(
        f"<b>{html.escape(line)}</b>" for line in text.splitlines()
    )
    figure.add_annotation(
        x=x,
        y=y,
        text=escaped_text,
        showarrow=False,
        xshift=xshift,
        yshift=yshift,
        align="center",
        bgcolor=style["bgcolor"],
        bordercolor=style["bordercolor"],
        borderwidth=1,
        borderpad=3,
        opacity=0.96,
        font={
            "family": MAP_LABEL_FONT,
            "size": size,
            "color": style["font_color"],
        },
    )


def _label_collision_radius(text: str) -> float:
    longest_line = max((len(line) for line in text.splitlines()), default=len(text))
    line_count = max(1, len(text.splitlines()))
    return min(3.2, max(1.0, longest_line * 0.16 + line_count * 0.24))


def _point_segment_distance(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    segment_dx = end_x - start_x
    segment_dy = end_y - start_y
    segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy
    if segment_length_sq == 0:
        return math.hypot(point_x - start_x, point_y - start_y)
    projection = (
        (point_x - start_x) * segment_dx + (point_y - start_y) * segment_dy
    ) / segment_length_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = start_x + projection * segment_dx
    closest_y = start_y + projection * segment_dy
    return math.hypot(point_x - closest_x, point_y - closest_y)


def _label_avoidance_segments(
    scenario: dict[str, Any],
    nodes: dict[str, dict[str, float]],
    plan: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    segments: list[tuple[float, float, float, float]] = []
    for edge in scenario.get("roads", []):
        if _is_synthetic_connector(edge):
            continue
        if edge["from"] not in nodes or edge["to"] not in nodes:
            continue
        start = nodes[edge["from"]]
        end = nodes[edge["to"]]
        segments.append((start["x"], start["y"], end["x"], end["y"]))
    for route in plan.get("routes", []):
        path = route.get("path") or []
        for start_id, end_id in zip(path, path[1:]):
            if start_id not in nodes or end_id not in nodes:
                continue
            start = nodes[start_id]
            end = nodes[end_id]
            segments.append((start["x"], start["y"], end["x"], end["y"]))
    return segments


def _choose_label_position(
    anchor_x: float,
    anchor_y: float,
    avoidance_segments: list[tuple[float, float, float, float]],
    occupied_labels: list[tuple[float, float, float]],
    *,
    label_radius: float = 1.1,
    prefer_above: bool = True,
) -> tuple[float, float]:
    candidate_offsets = [
        (0.0, 1.18),
        (1.08, 0.86),
        (-1.08, 0.86),
        (1.24, -0.70),
        (-1.24, -0.70),
        (0.0, -1.18),
        (1.70, 0.08),
        (-1.70, 0.08),
        (0.0, 1.72),
        (1.48, 1.30),
        (-1.48, 1.30),
        (0.0, -1.72),
    ]
    if not prefer_above:
        candidate_offsets = [
            (x_offset, -y_offset) for x_offset, y_offset in candidate_offsets
        ]

    best_position = (anchor_x, anchor_y + 1.18)
    best_score = float("-inf")
    for x_offset, y_offset in candidate_offsets:
        candidate_x = anchor_x + x_offset
        candidate_y = anchor_y + y_offset
        road_distance = min(
            (
                _point_segment_distance(
                    candidate_x,
                    candidate_y,
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                )
                for start_x, start_y, end_x, end_y in avoidance_segments
            ),
            default=9.0,
        )
        label_clearance = min(
            (
                math.hypot(candidate_x - label_x, candidate_y - label_y)
                - label_radius
                - occupied_radius
                for label_x, label_y, occupied_radius in occupied_labels
            ),
            default=9.0,
        )
        score = min(road_distance, 1.8) * 5.0 + min(label_clearance, 2.2) * 1.4
        score -= math.hypot(x_offset, y_offset) * 0.25
        if road_distance < 0.74:
            score -= (0.74 - road_distance) * 18.0
        if label_clearance < 0.0:
            score -= abs(label_clearance) * 16.0
        if y_offset > 0:
            score += 0.2
        if score > best_score:
            best_score = score
            best_position = (candidate_x, candidate_y)
    occupied_labels.append((best_position[0], best_position[1], label_radius))
    return best_position


def _group_unit_states_by_position(
    states: dict[str, dict[str, Any]],
) -> list[tuple[float, float, list[tuple[str, dict[str, Any]]]]]:
    grouped: dict[tuple[float, float], list[tuple[str, dict[str, Any]]]] = {}
    anchors: dict[tuple[float, float], tuple[float, float]] = {}
    for unit_id, state in states.items():
        x = float(state["position"]["x"])
        y = float(state["position"]["y"])
        key = (round(x, 1), round(y, 1))
        grouped.setdefault(key, []).append((unit_id, state))
        anchors.setdefault(key, (x, y))
    return [(anchors[key][0], anchors[key][1], grouped[key]) for key in grouped]


def _unit_group_label(units: list[tuple[str, dict[str, Any]]]) -> str:
    labels = [_unit_label(unit_id) for unit_id, _state in units]
    if len(labels) == 1:
        return labels[0]
    lines = [" / ".join(labels[index : index + 2]) for index in range(0, len(labels), 2)]
    return "\n".join(lines)


def _is_synthetic_connector(edge: dict[str, Any]) -> bool:
    return bool(edge.get("labels", {}).get("unit_anchor"))


def _sandbox_bounds(nodes: dict[str, dict[str, float]]) -> tuple[float, float, float, float]:
    x_values = [node["x"] for node in nodes.values()]
    y_values = [node["y"] for node in nodes.values()]
    return (
        min(x_values) - 2.5,
        max(x_values) + 2.5,
        min(y_values) - 2.5,
        max(y_values) + 2.5,
    )


def _discrete_colorscale(colors: dict[int, str]) -> list[list[float | str]]:
    max_value = max(colors)
    scale: list[list[float | str]] = []
    for value, color in sorted(colors.items()):
        start = max(0.0, (value - 0.5) / max_value)
        end = min(1.0, (value + 0.5) / max_value)
        scale.append([start, color])
        scale.append([end, color])
    return scale


def _distance_to_segment(
    point_x: float,
    point_y: float,
    start: dict[str, float],
    end: dict[str, float],
) -> float:
    segment_x = end["x"] - start["x"]
    segment_y = end["y"] - start["y"]
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0:
        return math.hypot(point_x - start["x"], point_y - start["y"])
    projection = (
        ((point_x - start["x"]) * segment_x + (point_y - start["y"]) * segment_y)
        / segment_length_squared
    )
    projection = max(0.0, min(1.0, projection))
    nearest_x = start["x"] + projection * segment_x
    nearest_y = start["y"] + projection * segment_y
    return math.hypot(point_x - nearest_x, point_y - nearest_y)


def _zone_state_for_cell(
    point_x: float,
    point_y: float,
    scenario: dict[str, Any],
) -> int:
    nodes = scenario["nodes"]
    state = 0
    for zone in scenario["zones"]:
        node = nodes[zone["node_id"]]
        distance = math.hypot(point_x - node["x"], point_y - node["y"])
        if distance > 3.4:
            continue
        observations = zone["observations"]
        if observations["fire"] >= 0.50 and distance <= 2.6:
            state = max(state, 3)
        elif (
            observations["road_damage"] >= 0.38
            or observations["congestion"] >= 0.55
            or observations["smoke"] >= 0.55
            or (observations["fire"] >= 0.35 and distance <= 3.0)
        ):
            state = max(state, 2)
        else:
            state = max(state, 1)
    return state


def _road_state_for_cell(
    point_x: float,
    point_y: float,
    scenario: dict[str, Any],
) -> int | None:
    nodes = scenario["nodes"]
    road_half_width = SANDBOX_GRID_CELL_SIZE * 0.52
    for road in scenario["roads"]:
        if _is_synthetic_connector(road):
            continue
        if road["from"] not in nodes or road["to"] not in nodes:
            continue
        distance = _distance_to_segment(
            point_x,
            point_y,
            nodes[road["from"]],
            nodes[road["to"]],
        )
        if distance <= road_half_width:
            return 5 if road.get("status") == "blocked" else 4
    return None


def _add_sandbox_state_grid(figure: go.Figure, scenario: dict[str, Any]) -> None:
    nodes = scenario["nodes"]
    min_x, max_x, min_y, max_y = _sandbox_bounds(nodes)
    x_values = [
        round(min_x + SANDBOX_GRID_CELL_SIZE / 2 + index * SANDBOX_GRID_CELL_SIZE, 3)
        for index in range(math.ceil((max_x - min_x) / SANDBOX_GRID_CELL_SIZE))
    ]
    y_values = [
        round(min_y + SANDBOX_GRID_CELL_SIZE / 2 + index * SANDBOX_GRID_CELL_SIZE, 3)
        for index in range(math.ceil((max_y - min_y) / SANDBOX_GRID_CELL_SIZE))
    ]
    z_values: list[list[int]] = []
    hover_text: list[list[str]] = []
    for y in y_values:
        z_row: list[int] = []
        hover_row: list[str] = []
        for x in x_values:
            state = _zone_state_for_cell(x, y, scenario)
            road_state = _road_state_for_cell(x, y, scenario)
            if road_state is not None:
                state = road_state
            z_row.append(state)
            hover_row.append(
                f"坐标 ({x:.2f}, {y:.2f})<br>{SANDBOX_STATE_LABELS[state]}"
            )
        z_values.append(z_row)
        hover_text.append(hover_row)
    figure.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            name="地形网格",
            zmin=0,
            zmax=max(SANDBOX_STATE_COLORS),
            colorscale=_discrete_colorscale(SANDBOX_STATE_COLORS),
            showscale=True,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={
                "title": {"text": "地块状态", "font": {"color": "#f4ead5"}},
                "tickmode": "array",
                "tickvals": list(SANDBOX_STATE_LABELS),
                "ticktext": [
                    "地面",
                    "烟雾",
                    "受损",
                    "火点",
                    "道路",
                    "阻断",
                ],
                "tickfont": {"color": "#dce8ef", "size": 8},
                "thickness": 7,
                "len": 0.58,
                "y": 0.48,
            },
        )
    )


def _fire_status(observations: dict[str, float]) -> str:
    fire = observations["fire"]
    if fire >= 0.50:
        return "high"
    if fire >= 0.30:
        return "medium"
    return "low"


def _edge_trace(
    edges: list[dict[str, Any]],
    nodes: dict[str, dict[str, float]],
    *,
    name: str,
    color: str,
    dash: str | None = None,
    width: float = 1.8,
) -> go.Scatter:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    hover = []
    for edge in edges:
        if edge["from"] not in nodes or edge["to"] not in nodes:
            continue
        x_values.extend([nodes[edge["from"]]["x"], nodes[edge["to"]]["x"], None])
        y_values.extend([nodes[edge["from"]]["y"], nodes[edge["to"]]["y"], None])
        risk_label = edge.get("display_risk")
        status_label = "阻断" if edge.get("status") == "blocked" else "通行"
        label = (
            f"{edge['road_id']}<br>{_node_label(edge['from'])} → "
            f"{_node_label(edge['to'])}<br>状态：{status_label}"
        )
        if risk_label is not None:
            label += f"<br>风险：{risk_label:.2f}"
        hover.extend([label, label, None])
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        text=hover,
        hoverinfo="text",
        line={"color": color, "width": width, "dash": dash or "solid"},
    )


def _ground_risk_groups(scenario: dict[str, Any]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    weights = scenario["config"]["weights"]["astar_risk"]
    groups: dict[str, list[dict[str, Any]]] = {
        "低风险": [],
        "中风险": [],
        "高风险": [],
        "阻断": [],
    }
    for road in scenario["roads"]:
        if _is_synthetic_connector(road):
            continue
        decorated = dict(road)
        decorated["display_risk"] = road_risk(road, weights)
        if road.get("status", "open") == "blocked":
            groups["阻断"].append(decorated)
        elif decorated["display_risk"] < 0.20:
            groups["低风险"].append(decorated)
        elif decorated["display_risk"] < 0.45:
            groups["中风险"].append(decorated)
        else:
            groups["高风险"].append(decorated)
    return [
        ("低风险", "rgba(24,116,75,.78)", groups["低风险"]),
        ("中风险", "rgba(191,130,31,.82)", groups["中风险"]),
        ("高风险", "rgba(205,64,48,.84)", groups["高风险"]),
        ("阻断", "rgba(22,18,16,.92)", groups["阻断"]),
    ]


def build_map_figure(
    scenario: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    focus: dict[str, Any] | None = None,
) -> go.Figure:
    nodes = scenario["nodes"]
    plan = snapshot.get("plan") or {
        "zone_assessment": [],
        "assignments": [],
        "routes": [],
        "utility_matrix": [],
    }
    focus = focus or {}
    figure = go.Figure()
    avoidance_segments = _label_avoidance_segments(scenario, nodes, plan)
    occupied_labels: list[tuple[float, float, float]] = []
    _add_sandbox_state_grid(figure, scenario)
    open_roads = [
        road
        for road in scenario["roads"]
        if road.get("status", "open") == "open"
        and not _is_synthetic_connector(road)
    ]
    if open_roads:
        figure.add_trace(
            _edge_trace(
                open_roads,
                nodes,
                name="道路骨架",
                color="rgba(35,43,49,.50)",
                width=1.25,
            )
        )
        figure.data[-1].showlegend = False
    for label, color, roads in _ground_risk_groups(scenario):
        if not roads:
            continue
        figure.add_trace(
            _edge_trace(
                roads,
                nodes,
                name=f"{label}道路",
                color=color,
                dash="dash" if label == "阻断" else None,
                width=1.55 if label != "阻断" else 2.0,
            )
        )
    figure.add_trace(
        _edge_trace(
            [
                route
                for route in scenario.get("air_routes", [])
                if not _is_synthetic_connector(route)
            ],
            nodes,
            name="无人机航线",
            color="rgba(11,168,200,.58)",
            dash="dot",
            width=1.45,
        )
    )
    figure.data[-1].opacity = 0.58

    blocked = [
        road
        for road in scenario["roads"]
        if road.get("status") == "blocked"
        and not _is_synthetic_connector(road)
    ]
    if blocked:
        figure.add_trace(
            go.Scatter(
                x=[(nodes[road["from"]]["x"] + nodes[road["to"]]["x"]) / 2 for road in blocked],
                y=[(nodes[road["from"]]["y"] + nodes[road["to"]]["y"]) / 2 for road in blocked],
                mode="markers",
                name="断裂路段",
                marker={"symbol": "x", "size": 14, "color": "#171717", "line": {"width": 3}},
                text=[road["road_id"] for road in blocked],
                hovertemplate="%{text}<br>状态：阻断<extra></extra>",
            )
        )

    candidate_routes = []
    candidates_by_unit: dict[str, list[dict[str, Any]]] = {}
    for candidate in plan.get("utility_matrix", []):
        if candidate.get("feasible") and candidate.get("route"):
            candidates_by_unit.setdefault(candidate["unit_id"], []).append(candidate)
    for candidates in candidates_by_unit.values():
        candidate_routes.extend(
            sorted(
                candidates,
                key=lambda item: item.get("expected_utility") or -999.0,
                reverse=True,
            )[:2]
        )
    if candidate_routes:
        candidate_x: list[float | None] = []
        candidate_y: list[float | None] = []
        candidate_hover: list[str | None] = []
        for candidate in candidate_routes:
            route = candidate["route"]
            label = (
                f"{_unit_label(candidate['unit_id'])} → {_zone_label(candidate['target_zone'])}"
                f"<br>预计 {route['eta']:.1f} 分钟 · 风险 {route['path_risk']:.2f}"
            )
            for node in route["path"]:
                candidate_x.append(nodes[node]["x"])
                candidate_y.append(nodes[node]["y"])
                candidate_hover.append(label)
            candidate_x.append(None)
            candidate_y.append(None)
            candidate_hover.append(None)
        figure.add_trace(
            go.Scatter(
                x=candidate_x,
                y=candidate_y,
                mode="lines",
                name="候选路线",
                text=candidate_hover,
                hoverinfo="text",
                line={"color": "rgba(20,35,45,.34)", "width": 2},
            )
        )
        figure.data[-1].showlegend = False

    for route in plan.get("routes", []):
        path = route["path"]
        if len(path) < 2 or any(node not in nodes for node in path):
            continue
        unit_id = route["unit_id"]
        figure.add_trace(
            go.Scatter(
                x=[nodes[node]["x"] for node in path],
                y=[nodes[node]["y"] for node in path],
                mode="lines+markers",
                name=f"执行路线 · {_unit_label(unit_id)}",
                line={"color": UNIT_COLORS.get(unit_id, "#f05a28"), "width": 5},
                marker={"size": 6, "line": {"color": "#fff8e8", "width": 1}},
                hovertemplate=(
                    f"{_unit_label(unit_id)}<br>剩余 {route['remaining_eta']:.1f} 分钟"
                    f"<br>路径风险 {route['path_risk']:.2f}<extra></extra>"
                ),
            )
        )

    assessments = {item["zone_id"]: item for item in plan.get("zone_assessment", [])}
    zones = scenario["zones"]
    display_assessments = {}
    for zone in zones:
        zone_id = zone["zone_id"]
        if zone_id in assessments:
            display_assessments[zone_id] = assessments[zone_id]
            continue
        observations = zone["observations"]
        trapped = min(
            1.0,
            0.45 * observations["sos_signal"]
            + 0.35 * observations["building_collapse"]
            + 0.20 * observations["human_activity"],
        )
        passability = max(
            0.0,
            1.0
            - 0.50 * observations["road_damage"]
            - 0.30 * observations["fire"]
            - 0.20 * observations["congestion"],
        )
        life_risk = min(
            1.0,
            0.40 * observations["fire"]
            + 0.35 * trapped
            + 0.25 * observations["time_urgency"],
        )
        display_assessments[zone_id] = {
            "zone_id": zone_id,
            "trapped_prob": trapped,
            "passability_prob": passability,
            "life_risk": life_risk,
            "priority_score": min(
                1.0,
                0.40 * trapped
                + 0.30 * life_risk
                + 0.20 * observations["time_urgency"]
                + 0.10 * passability,
            ),
        }
    ranked_zone_ids = [
        item[0]
        for item in sorted(
            display_assessments.items(),
            key=lambda pair: pair[1]["priority_score"],
            reverse=True,
        )
    ]
    rank_by_zone = {zone_id: index + 1 for index, zone_id in enumerate(ranked_zone_ids)}
    figure.add_trace(
        go.Scatter(
            x=[nodes[zone["node_id"]]["x"] for zone in zones],
            y=[nodes[zone["node_id"]]["y"] for zone in zones],
            mode="markers",
            name="火势范围",
            text=[
                _zone_label(zone["zone_id"])
                for zone in zones
            ],
            marker={
                "size": [18 + 18 * zone["observations"]["fire"] for zone in zones],
                "symbol": "square",
                "color": [
                    FIRE_STATUS_COLORS[_fire_status(zone["observations"])]
                    for zone in zones
                ],
                "opacity": 0.66,
                "line": {"color": "rgba(20,20,20,.62)", "width": 1},
            },
            customdata=[
                [
                    zone["observations"]["fire"],
                    zone["observations"]["smoke"],
                    FIRE_STATUS_LABELS[_fire_status(zone["observations"])],
                ]
                for zone in zones
            ],
            hovertemplate=(
                "%{text}<br>火势 %{customdata[0]:.2f}"
                "<br>烟雾 %{customdata[1]:.2f}<br>等级 %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[nodes[zone["node_id"]]["x"] for zone in zones],
            y=[nodes[zone["node_id"]]["y"] for zone in zones],
            mode="markers",
            name="生命风险范围",
            hoverinfo="skip",
            marker={
                "size": [42 + 34 * display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "color": "rgba(236,82,45,.10)",
                "line": {"color": "rgba(236,82,45,.16)", "width": 1},
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[nodes[zone["node_id"]]["x"] for zone in zones],
            y=[nodes[zone["node_id"]]["y"] for zone in zones],
            mode="markers",
            name="灾区优先级",
            text=[f"#{rank_by_zone[zone['zone_id']]} {_zone_label(zone['zone_id'])}" for zone in zones],
            marker={
                "size": [21 + 16 * display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "color": [display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "colorscale": [[0, "#f5d58d"], [0.55, "#f2943d"], [1, "#dc3b2a"]],
                "cmin": 0,
                "cmax": 1,
                "showscale": False,
                "line": {"color": "#fff6df", "width": 2},
            },
            customdata=[
                [
                    display_assessments[zone["zone_id"]]["trapped_prob"],
                    display_assessments[zone["zone_id"]]["passability_prob"],
                    display_assessments[zone["zone_id"]]["priority_score"],
                ]
                for zone in zones
            ],
            hovertemplate=(
                "%{text}<br>被困概率 %{customdata[0]:.2f}"
                "<br>通行概率 %{customdata[1]:.2f}"
                "<br>优先级 %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    for zone in zones:
        node = nodes[zone["node_id"]]
        label = f"#{rank_by_zone[zone['zone_id']]} {_zone_label(zone['zone_id'])}"
        label_x, label_y = _choose_label_position(
            node["x"],
            node["y"],
            avoidance_segments,
            occupied_labels,
            label_radius=_label_collision_radius(label),
        )
        _add_map_label(
            figure,
            x=label_x,
            y=label_y,
            text=label,
            kind="zone",
            size=12,
        )

    junction_ids = sorted(node_id for node_id in nodes if node_id.startswith("J"))
    if junction_ids:
        figure.add_trace(
            go.Scatter(
                x=[nodes[node]["x"] for node in junction_ids],
                y=[nodes[node]["y"] for node in junction_ids],
                mode="markers",
                name="路网节点",
                text=junction_ids,
                marker={
                    "size": 4,
                    "color": "#6a5738",
                    "line": {"color": "#fff3d0", "width": 0.8},
                },
                hovertemplate="路网节点 %{text}<extra></extra>",
                showlegend=False,
            )
        )

    infrastructure_ids = [
        node_id for node_id in ("HQ", "HOSPITAL", "AIR_RELAY") if node_id in nodes
    ]
    figure.add_trace(
        go.Scatter(
            x=[nodes[node]["x"] for node in infrastructure_ids],
            y=[nodes[node]["y"] for node in infrastructure_ids],
            mode="markers",
            name="关键设施",
            text=[_node_label(node) for node in infrastructure_ids],
            marker={
                "size": 16,
                "symbol": "diamond",
                "color": ["#2d2a26", "#f6f0e1", "#2d2a26"][: len(infrastructure_ids)],
                "line": {"color": "#1b1712", "width": 1.5},
            },
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for node_id in infrastructure_ids:
        node = nodes[node_id]
        label = _node_label(node_id)
        label_x, label_y = _choose_label_position(
            node["x"],
            node["y"],
            avoidance_segments,
            occupied_labels,
            label_radius=_label_collision_radius(label),
            prefer_above=node_id != "AIR_RELAY",
        )
        _add_map_label(
            figure,
            x=label_x,
            y=label_y,
            text=label,
            kind="facility",
            size=11,
        )

    states = snapshot.get("unit_states") or {
        unit["unit_id"]: {
            "unit_id": unit["unit_id"],
            "type": unit["type"],
            "status": "ready",
            "position": nodes[unit["start_node"]],
            "onboard": 0,
            "capacity": int(unit.get("capacity", 0)),
        }
        for unit in scenario["units"]
    }
    figure.add_trace(
        go.Scatter(
            x=[state["position"]["x"] for state in states.values()],
            y=[state["position"]["y"] for state in states.values()],
            mode="markers",
            name="救援单位",
            text=[_unit_label(unit_id) for unit_id in states],
            marker={
                "size": 17,
                "symbol": ["triangle-up" if state["type"] == "drone" else "square" for state in states.values()],
                "color": [UNIT_COLORS.get(unit_id, "#f05a28") for unit_id in states],
                "line": {"color": "#fff8e8", "width": 2},
            },
            customdata=[
                [
                    unit_id,
                    state["status"],
                    state["onboard"],
                    state["capacity"],
                ]
                for unit_id, state in states.items()
            ],
            hovertemplate=(
                "%{text}<br>编号 %{customdata[0]}<br>状态 %{customdata[1]}"
                "<br>载员 %{customdata[2]}/%{customdata[3]}<extra></extra>"
            ),
        )
    )
    for anchor_x, anchor_y, units_at_position in _group_unit_states_by_position(states):
        label = _unit_group_label(units_at_position)
        label_x, label_y = _choose_label_position(
            anchor_x,
            anchor_y,
            avoidance_segments,
            occupied_labels,
            label_radius=_label_collision_radius(label),
            prefer_above=False,
        )
        _add_map_label(
            figure,
            x=label_x,
            y=label_y,
            text=label,
            kind="unit",
            size=11,
        )

    focused_roads = [
        road for road in [*scenario["roads"], *scenario.get("air_routes", [])]
        if road["road_id"] in set(focus.get("roads", []))
    ]
    if focused_roads:
        figure.add_trace(
            _edge_trace(
                focused_roads,
                nodes,
                name="当前计算道路",
                color="#ff6b35",
                width=5.6,
            )
        )

    focused_zones = [zone for zone in zones if zone["zone_id"] in set(focus.get("zones", []))]
    if focused_zones:
        figure.add_trace(
            go.Scatter(
                x=[nodes[zone["node_id"]]["x"] for zone in focused_zones],
                y=[nodes[zone["node_id"]]["y"] for zone in focused_zones],
                mode="markers",
                name="当前计算灾区",
                marker={
                    "size": 54,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#fff4cf", "width": 4},
                },
                hoverinfo="skip",
            )
        )

    focused_unit_ids = [unit_id for unit_id in states if unit_id in set(focus.get("units", []))]
    if focused_unit_ids:
        figure.add_trace(
            go.Scatter(
                x=[states[unit_id]["position"]["x"] for unit_id in focused_unit_ids],
                y=[states[unit_id]["position"]["y"] for unit_id in focused_unit_ids],
                mode="markers",
                name="当前计算单位",
                marker={
                    "size": 28,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#ff6b35", "width": 4},
                },
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        height=510,
        margin={"l": 4, "r": 4, "t": 22, "b": 4},
        paper_bgcolor="#14212b",
        plot_bgcolor="#cfc3a3",
        font={"family": "Avenir Next Condensed, sans-serif", "color": "#1b1712"},
        showlegend=False,
        legend={
            "orientation": "h",
            "y": 1.025,
            "x": 0,
            "font": {"color": "#f4ead5", "size": 9},
            "bgcolor": "rgba(23,27,32,.66)",
            "bordercolor": "rgba(255,255,255,.12)",
            "borderwidth": 1,
            "itemsizing": "constant",
        },
        xaxis={
            "visible": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={"visible": False},
        hoverlabel={"bgcolor": "#2d2a26", "font_color": "#fff8e8"},
        hovermode="closest",
        dragmode="pan",
    )
    return figure


def build_probability_frame(plan: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "区域": item["zone_id"],
                "被困概率": item["trapped_prob"],
                "道路可通": item["passability_prob"],
                "生命风险": item["life_risk"],
                "优先级": item["priority_score"],
            }
            for item in plan["zone_assessment"]
        ]
    )


def build_utility_frame(utility_matrix: list[dict[str, Any]]) -> pd.DataFrame:
    reason_labels = {
        "feasible": "可执行",
        "passability_below_vehicle_minimum": "道路通行概率不足",
        "fire_risk_above_vehicle_maximum": "火灾风险超限",
        "direct_air_route": "空中直飞",
        "feasible_with_risk_override": "高风险绕行",
        "no_air_route": "无可达空中航线",
        "no_ground_route": "无可达地面路线",
    }
    rows = []
    for item in utility_matrix:
        route = item.get("route") or {}
        rows.append(
            {
                "单位": item["unit_id"],
                "区域": item["target_zone"],
                "任务": "侦察" if item["mission_type"] == "reconnaissance" else "救援",
                "可行": "是" if item["feasible"] else "否",
                "总效用": item.get("expected_utility"),
                "ETA": route.get("eta"),
                "路径风险": route.get("path_risk"),
                "资源成本": item.get("resource_cost"),
                "原因": reason_labels.get(item.get("reason", ""), item.get("reason", "")),
            }
        )
    return pd.DataFrame(rows)


def build_utility_contribution_figure(candidate: dict[str, Any]) -> go.Figure:
    breakdown = candidate.get("utility_breakdown") or {}
    labels = {
        "trapped_benefit": "被困收益",
        "life_risk_benefit": "生命风险收益",
        "accessibility_benefit": "任务适配收益",
        "arrival_time_cost": "到达时间成本",
        "path_risk_cost": "路径风险成本",
        "resource_cost": "资源消耗成本",
    }
    names = [name for name in labels if name in breakdown]
    values = [breakdown[name] for name in names]
    total = candidate.get("expected_utility")
    figure = go.Figure(
        go.Waterfall(
            name="效用贡献",
            orientation="v",
            measure=["relative"] * len(names) + ["total"],
            x=[labels[name] for name in names] + ["总期望效用"],
            y=values + [0],
            text=[f"{value:+.3f}" for value in values] + [f"{total:.3f}" if total is not None else "-"],
            textposition="outside",
            connector={"line": {"color": "#746f65", "width": 1}},
            increasing={"marker": {"color": "#20b8cd"}},
            decreasing={"marker": {"color": "#f05a28"}},
            totals={"marker": {"color": "#2d2a26"}},
        )
    )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 25, "b": 20},
        paper_bgcolor="#f3ead7",
        plot_bgcolor="#fff8e8",
        yaxis_title="加权效用贡献",
        showlegend=False,
        font={"family": "Avenir Next Condensed, sans-serif", "color": "#2d2a26"},
    )
    return figure


def build_metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key, model_label in (
        ("expert_cpt", "Expert CPT"),
        ("learned_cpt", "Learned CPT"),
    ):
        for target, target_label in (
            ("trapped_people", "Trapped people"),
            ("road_passable", "Road passable"),
        ):
            values = metrics["aggregate"][model_key][target]
            rows.append(
                {
                    "模型": model_label,
                    "目标": target_label,
                    "Brier": values["brier"],
                    "Accuracy": values["accuracy"],
                    "F1": values["f1"],
                    "ROC-AUC": values["roc_auc"],
                }
            )
    return pd.DataFrame(rows)


def build_calibration_figure(metrics: dict[str, Any], target: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#746f65", "dash": "dash"},
        )
    )
    for model_key, label, color in (
        ("expert_cpt", "Expert CPT", "#e2b13c"),
        ("learned_cpt", "Learned CPT", "#20b8cd"),
    ):
        bins = metrics["aggregate"][model_key][target]["calibration_bins"]
        figure.add_trace(
            go.Scatter(
                x=[item["mean_predicted"] for item in bins],
                y=[item["fraction_positive"] for item in bins],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3},
            )
        )
    figure.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="#f3ead7",
        plot_bgcolor="#fff8e8",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
        font={"family": "Avenir Next Condensed, sans-serif", "color": "#2d2a26"},
    )
    return figure
