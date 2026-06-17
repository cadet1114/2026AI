from copy import deepcopy
from collections import deque

from emergency_commander.contracts import validate_scenario
from emergency_commander.random_scenario import generate_random_scenario


def _reachable(roads, start, goal):
    graph = {}
    for road in roads:
        if road["status"] != "open":
            continue
        graph.setdefault(road["from"], set()).add(road["to"])
        if road.get("bidirectional", True):
            graph.setdefault(road["to"], set()).add(road["from"])
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for neighbor in graph.get(node, set()) - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def test_random_scenario_is_reproducible_and_contract_valid():
    first = generate_random_scenario(20260616)
    second = generate_random_scenario(20260616)
    other = generate_random_scenario(20260617)

    assert first == second
    assert first != other
    validate_scenario(first)
    assert first["events"] == []
    assert 4 <= len(first["zones"]) <= 7
    assert len(first["nodes"]) >= 25
    assert 34 <= len(first["roads"]) <= 52
    assert len(first["air_routes"]) >= len(first["zones"]) + 2
    assert any(road["status"] == "blocked" for road in first["roads"])
    unit_types = [unit["type"] for unit in first["units"]]
    assert unit_types.count("rescue_car") == 3
    assert unit_types.count("drone") == 2


def test_random_scenario_changes_road_topology_between_seeds():
    first = generate_random_scenario(20260616)
    other = generate_random_scenario(20260617)

    first_roads = {road["road_id"] for road in first["roads"]}
    other_roads = {road["road_id"] for road in other["roads"]}
    first_air = {route["road_id"] for route in first["air_routes"]}
    other_air = {route["road_id"] for route in other["air_routes"]}

    assert first_roads != other_roads
    assert first_air != other_air
    assert len(first_roads ^ other_roads) >= 4
    assert [zone["node_id"] for zone in first["zones"]] != [
        zone["node_id"] for zone in other["zones"]
    ]


def test_random_scenario_includes_diagonal_ground_roads():
    scenario = generate_random_scenario(20260616)
    nodes = scenario["nodes"]

    diagonal_roads = [
        road
        for road in scenario["roads"]
        if abs(nodes[road["from"]]["x"] - nodes[road["to"]]["x"]) >= 1.0
        and abs(nodes[road["from"]]["y"] - nodes[road["to"]]["y"]) >= 1.0
    ]

    assert len(diagonal_roads) >= 2
    assert sum("BYPASS" in road["road_id"] for road in diagonal_roads) >= 2


def test_every_single_ground_road_failure_keeps_zones_reachable():
    for seed in (7, 20260616, 20260617):
        scenario = generate_random_scenario(seed)
        for failed in scenario["roads"]:
            if failed["status"] != "open" or failed["road_id"].startswith("R_HOSPITAL"):
                continue
            roads = deepcopy(scenario["roads"])
            target = next(
                road for road in roads if road["road_id"] == failed["road_id"]
            )
            target["status"] = "blocked"
            for zone in scenario["zones"]:
                assert _reachable(roads, "HQ", zone["node_id"]), (
                    seed,
                    failed["road_id"],
                )


def test_air_routes_reach_every_zone():
    scenario = generate_random_scenario(9)
    for zone in scenario["zones"]:
        assert _reachable(scenario["air_routes"], "HQ", zone["node_id"])
