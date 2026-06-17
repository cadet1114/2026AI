from __future__ import annotations

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
SANDBOX_GRID_CELL_SIZE = 0.45
SANDBOX_STATE_COLORS = {
    0: "#e7d3a8",
    1: "#d8c28f",
    2: "#ffb13b",
    3: "#e3342f",
    4: "#111111",
    5: "#5b1515",
}
SANDBOX_STATE_LABELS = {
    0: "stable terrain",
    1: "smoke or light hazard",
    2: "damaged / congested area",
    3: "active fire",
    4: "open road",
    5: "blocked road",
}


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
    road_half_width = SANDBOX_GRID_CELL_SIZE * 0.72
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
                f"x {x:.2f} / y {y:.2f}<br>{SANDBOX_STATE_LABELS[state]}"
            )
        z_values.append(z_row)
        hover_text.append(hover_row)
    figure.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            name="Sandbox state grid",
            zmin=0,
            zmax=max(SANDBOX_STATE_COLORS),
            colorscale=_discrete_colorscale(SANDBOX_STATE_COLORS),
            showscale=True,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={
                "title": {"text": "Cell state", "font": {"color": "#f4ead5"}},
                "tickmode": "array",
                "tickvals": list(SANDBOX_STATE_LABELS),
                "ticktext": [
                    "clear",
                    "smoke",
                    "damage",
                    "fire",
                    "road",
                    "block",
                ],
                "tickfont": {"color": "#f4ead5", "size": 9},
                "thickness": 9,
                "len": 0.74,
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
        label = f"{edge['road_id']} · {edge.get('status', 'open')}"
        if risk_label is not None:
            label += f" · risk {risk_label:.2f}"
        hover.extend([label, label, None])
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        text=hover,
        hoverinfo="text",
        line={"color": color, "width": 2, "dash": dash or "solid"},
    )


def _ground_risk_groups(scenario: dict[str, Any]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    weights = scenario["config"]["weights"]["astar_risk"]
    groups: dict[str, list[dict[str, Any]]] = {
        "Low risk": [],
        "Medium risk": [],
        "High risk": [],
        "Blocked": [],
    }
    for road in scenario["roads"]:
        if _is_synthetic_connector(road):
            continue
        decorated = dict(road)
        decorated["display_risk"] = road_risk(road, weights)
        if road.get("status", "open") == "blocked":
            groups["Blocked"].append(decorated)
        elif decorated["display_risk"] < 0.20:
            groups["Low risk"].append(decorated)
        elif decorated["display_risk"] < 0.45:
            groups["Medium risk"].append(decorated)
        else:
            groups["High risk"].append(decorated)
    return [
        ("Low risk", "#12633f", groups["Low risk"]),
        ("Medium risk", "#b97808", groups["Medium risk"]),
        ("High risk", "#b52f23", groups["High risk"]),
        ("Blocked", "#1b1712", groups["Blocked"]),
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
                name="State · Roads",
                color="#111111",
            )
        )
        figure.data[-1].line.width = 3
    for label, color, roads in _ground_risk_groups(scenario):
        if not roads:
            continue
        figure.add_trace(
            _edge_trace(
                roads,
                nodes,
                name=f"Ground · {label}",
                color=color,
                dash="dash" if label == "Blocked" else None,
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
            name="Air corridors",
            color="#008bb0",
            dash="dot",
        )
    )

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
                name="Collapsed road",
                marker={"symbol": "x", "size": 15, "color": "#171717", "line": {"width": 3}},
                text=[road["road_id"] for road in blocked],
                hovertemplate="%{text}<br>BLOCKED<extra></extra>",
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
                f"{candidate['unit_id']} → {candidate['target_zone']}"
                f"<br>ETA {route['eta']:.1f} min · risk {route['path_risk']:.2f}"
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
                name="Candidate routes",
                text=candidate_hover,
                hoverinfo="text",
                line={"color": "rgba(42,55,65,.45)", "width": 3},
            )
        )

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
                name=f"Route · {unit_id}",
                line={"color": UNIT_COLORS.get(unit_id, "#f05a28"), "width": 6},
                marker={"size": 7},
                hovertemplate=(
                    f"{unit_id}<br>ETA {route['remaining_eta']:.1f} min"
                    f"<br>Risk {route['path_risk']:.2f}<extra></extra>"
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
            name="State · Fire zones",
            text=[
                f"{zone['zone_id']} fire {zone['observations']['fire']:.2f}"
                for zone in zones
            ],
            marker={
                "size": [18 + 18 * zone["observations"]["fire"] for zone in zones],
                "symbol": "square",
                "color": [
                    FIRE_STATUS_COLORS[_fire_status(zone["observations"])]
                    for zone in zones
                ],
                "opacity": 0.82,
                "line": {"color": "#111111", "width": 1},
            },
            customdata=[
                [
                    zone["observations"]["fire"],
                    zone["observations"]["smoke"],
                    _fire_status(zone["observations"]),
                ]
                for zone in zones
            ],
            hovertemplate=(
                "ZONE %{text}<br>Fire %{customdata[0]:.2f}"
                "<br>Smoke %{customdata[1]:.2f}<br>Status %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[nodes[zone["node_id"]]["x"] for zone in zones],
            y=[nodes[zone["node_id"]]["y"] for zone in zones],
            mode="markers",
            name="Risk halo",
            hoverinfo="skip",
            marker={
                "size": [42 + 34 * display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "color": "rgba(236,82,45,.16)",
                "line": {"color": "rgba(236,82,45,.28)", "width": 1},
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[nodes[zone["node_id"]]["x"] for zone in zones],
            y=[nodes[zone["node_id"]]["y"] for zone in zones],
            mode="markers+text",
            name="Disaster zones",
            text=[f"#{rank_by_zone[zone['zone_id']]}  ZONE {zone['zone_id']}" for zone in zones],
            textposition="top center",
            textfont={"color": "#1b1712", "size": 12},
            marker={
                "size": [21 + 16 * display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "color": [display_assessments[zone["zone_id"]]["life_risk"] for zone in zones],
                "colorscale": [[0, "#f5d58d"], [0.55, "#f2943d"], [1, "#dc3b2a"]],
                "cmin": 0,
                "cmax": 1,
                "showscale": False,
                "line": {"color": "#1b1712", "width": 2},
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
                "Zone %{text}<br>Trapped P %{customdata[0]:.2f}"
                "<br>Passable P %{customdata[1]:.2f}"
                "<br>Priority %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )

    junction_ids = sorted(node_id for node_id in nodes if node_id.startswith("J"))
    if junction_ids:
        figure.add_trace(
            go.Scatter(
                x=[nodes[node]["x"] for node in junction_ids],
                y=[nodes[node]["y"] for node in junction_ids],
                mode="markers",
                name="District junctions",
                text=junction_ids,
                marker={
                    "size": 5,
                    "color": "#59462d",
                    "line": {"color": "#fff3d0", "width": 1},
                },
                hovertemplate="Junction %{text}<extra></extra>",
            )
        )

    infrastructure_ids = [
        node_id for node_id in ("HQ", "HOSPITAL", "AIR_RELAY") if node_id in nodes
    ]
    figure.add_trace(
        go.Scatter(
            x=[nodes[node]["x"] for node in infrastructure_ids],
            y=[nodes[node]["y"] for node in infrastructure_ids],
            mode="markers+text",
            name="Infrastructure",
            text=infrastructure_ids,
            textposition="bottom center",
            textfont={"color": "#1b1712", "size": 11},
            marker={"size": 15, "symbol": "diamond", "color": "#2d2a26"},
        )
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
            mode="markers+text",
            name="Units",
            text=list(states),
            textposition="middle right",
            textfont={"color": "#1b1712", "size": 11},
            marker={
                "size": 17,
                "symbol": ["triangle-up" if state["type"] == "drone" else "square" for state in states.values()],
                "color": [UNIT_COLORS.get(unit_id, "#f05a28") for unit_id in states],
                "line": {"color": "#fff8e8", "width": 2},
            },
            customdata=[
                [state["status"], state["onboard"], state["capacity"]]
                for state in states.values()
            ],
            hovertemplate=(
                "%{text}<br>Status %{customdata[0]}<br>Onboard %{customdata[1]}/%{customdata[2]}<extra></extra>"
            ),
        )
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
                name="Calculation focus · roads",
                color="#ff6b35",
            )
        )
        figure.data[-1].line.width = 8

    focused_zones = [zone for zone in zones if zone["zone_id"] in set(focus.get("zones", []))]
    if focused_zones:
        figure.add_trace(
            go.Scatter(
                x=[nodes[zone["node_id"]]["x"] for zone in focused_zones],
                y=[nodes[zone["node_id"]]["y"] for zone in focused_zones],
                mode="markers",
                name="Calculation focus · zones",
                marker={
                    "size": 54,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#fff4cf", "width": 5},
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
                name="Calculation focus · units",
                marker={
                    "size": 28,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#ff6b35", "width": 4},
                },
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        height=610,
        margin={"l": 4, "r": 4, "t": 30, "b": 4},
        paper_bgcolor="#11161b",
        plot_bgcolor="#e5d0a6",
        font={"family": "Avenir Next Condensed, sans-serif", "color": "#1b1712"},
        legend={
            "orientation": "h",
            "y": 1.04,
            "x": 0,
            "font": {"color": "#f4ead5", "size": 10},
            "bgcolor": "rgba(23,27,32,.78)",
        },
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        hoverlabel={"bgcolor": "#2d2a26", "font_color": "#fff8e8"},
        hovermode="closest",
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
