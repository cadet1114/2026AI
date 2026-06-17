from __future__ import annotations

import heapq
from collections import defaultdict
from math import hypot, inf
from typing import Any


class NoRouteError(RuntimeError):
    """Raised when no traversable path exists between two nodes."""


def road_risk(road: dict[str, Any], risk_weights: dict[str, float]) -> float:
    risk = road["risk"]
    return max(
        0.0,
        min(
            1.0,
            risk_weights["fire"] * risk["fire"]
            + risk_weights["damage"] * risk["damage"]
            + risk_weights["congestion"] * risk["congestion"]
            + risk_weights["secondary"] * risk["secondary_disaster"],
        ),
    )


def risk_aware_astar(
    roads: list[dict[str, Any]],
    *,
    nodes: dict[str, dict[str, float]] | None = None,
    start: str,
    goal: str,
    speed: float,
    risk_weights: dict[str, float],
    unit_type: str = "rescue_car",
    max_fire_risk: float | None = None,
    route_layer: str = "ground",
    include_trace: bool = False,
) -> dict[str, Any]:
    """Find the minimum time-and-risk route with an admissible time heuristic."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    if start == goal:
        result = {
            "path": [start],
            "road_ids": [],
            "eta": 0.0,
            "path_risk": 0.0,
            "total_cost": 0.0,
            "route_layer": route_layer,
            "heuristic": "euclidean_time_lower_bound" if nodes else "zero_fallback",
            "heuristic_start": 0.0,
            "expanded_nodes": 0,
        }
        if include_trace:
            result["search_trace"] = []
        return result

    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for road in roads:
        if road.get("status", "open") == "blocked":
            continue
        if (
            unit_type != "drone"
            and max_fire_risk is not None
            and road["risk"]["fire"] > max_fire_risk
        ):
            continue
        adjacency[road["from"]].append((road["to"], road))
        if road.get("bidirectional", True):
            adjacency[road["to"]].append((road["from"], road))

    minimum_time_per_distance = 0.0
    if nodes and start in nodes and goal in nodes:
        ratios = []
        for road in roads:
            if road["from"] not in nodes or road["to"] not in nodes:
                continue
            distance = hypot(
                nodes[road["from"]]["x"] - nodes[road["to"]]["x"],
                nodes[road["from"]]["y"] - nodes[road["to"]]["y"],
            )
            if distance > 0:
                ratios.append(float(road["travel_time_base"]) / distance)
        if ratios:
            minimum_time_per_distance = min(ratios) / speed

    def heuristic(node: str) -> float:
        if not nodes or minimum_time_per_distance <= 0 or node not in nodes or goal not in nodes:
            return 0.0
        return hypot(
            nodes[node]["x"] - nodes[goal]["x"],
            nodes[node]["y"] - nodes[goal]["y"],
        ) * minimum_time_per_distance

    heuristic_start = heuristic(start)
    frontier: list[tuple[float, float, str]] = [(heuristic_start, 0.0, start)]
    best_cost = {start: 0.0}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    expanded_nodes = 0
    search_trace: list[dict[str, Any]] = []

    while frontier:
        _, current_cost, node = heapq.heappop(frontier)
        if current_cost > best_cost.get(node, inf):
            continue
        expanded_nodes += 1
        current_h = heuristic(node)
        relaxations: list[dict[str, Any]] = []
        if node == goal:
            if include_trace:
                search_trace.append(
                    {
                        "node": node,
                        "g": round(current_cost, 6),
                        "h": round(current_h, 6),
                        "f": round(current_cost + current_h, 6),
                        "frontier_size": len(frontier),
                        "relaxations": relaxations,
                    }
                )
            break
        for neighbor, road in adjacency.get(node, []):
            base_time = float(road["travel_time_base"]) / speed
            risk = road_risk(road, risk_weights)
            if unit_type == "drone":
                risk *= 0.25
            traversal_cost = base_time * (1.0 + risk)
            candidate_cost = current_cost + traversal_cost
            if candidate_cost < best_cost.get(neighbor, inf):
                best_cost[neighbor] = candidate_cost
                previous[neighbor] = (node, road)
                neighbor_h = heuristic(neighbor)
                heapq.heappush(
                    frontier,
                    (candidate_cost + neighbor_h, candidate_cost, neighbor),
                )
                if include_trace:
                    relaxations.append(
                        {
                            "neighbor": neighbor,
                            "road_id": road["road_id"],
                            "g": round(candidate_cost, 6),
                            "h": round(neighbor_h, 6),
                            "f": round(candidate_cost + neighbor_h, 6),
                            "edge_cost": round(traversal_cost, 6),
                        }
                    )
        if include_trace:
            search_trace.append(
                {
                    "node": node,
                    "g": round(current_cost, 6),
                    "h": round(current_h, 6),
                    "f": round(current_cost + current_h, 6),
                    "frontier_size": len(frontier),
                    "relaxations": relaxations,
                }
            )

    if goal not in best_cost:
        raise NoRouteError(f"no route from {start} to {goal}")

    path_nodes = [goal]
    selected_roads: list[dict[str, Any]] = []
    cursor = goal
    while cursor != start:
        prior_node, road = previous[cursor]
        path_nodes.append(prior_node)
        selected_roads.append(road)
        cursor = prior_node
    path_nodes.reverse()
    selected_roads.reverse()

    eta = sum(float(road["travel_time_base"]) / speed for road in selected_roads)
    if eta:
        weighted_risk = sum(
            road_risk(road, risk_weights) * (float(road["travel_time_base"]) / speed)
            for road in selected_roads
        ) / eta
    else:
        weighted_risk = 0.0
    if unit_type == "drone":
        weighted_risk *= 0.25

    result = {
        "path": path_nodes,
        "road_ids": [road["road_id"] for road in selected_roads],
        "eta": round(eta, 6),
        "path_risk": round(weighted_risk, 6),
        "total_cost": round(best_cost[goal], 6),
        "route_layer": route_layer,
        "heuristic": "euclidean_time_lower_bound" if heuristic_start > 0 else "zero_fallback",
        "heuristic_start": round(heuristic_start, 6),
        "expanded_nodes": expanded_nodes,
    }
    if include_trace:
        result["search_trace"] = search_trace
    return result
