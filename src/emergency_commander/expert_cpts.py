from __future__ import annotations

from itertools import product

from emergency_commander.bayesian_network import BayesianNode, DiscreteBayesianNetwork


LEVELS = ("low", "medium", "high")
BINARY = ("no", "yes")
REPORT_STATES = ("blocked", "uncertain", "open")
LEVEL_VALUE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def _root(name: str, states: tuple[str, ...], probabilities: tuple[float, ...]) -> BayesianNode:
    return BayesianNode(name, states, (), {(): dict(zip(states, probabilities, strict=True))})


def _single_parent(
    name: str,
    parent: str,
    rows: dict[str, tuple[float, float, float]],
) -> BayesianNode:
    return BayesianNode(
        name,
        LEVELS,
        (parent,),
        {(parent_state,): dict(zip(LEVELS, values, strict=True)) for parent_state, values in rows.items()},
    )


def _trapped_node() -> BayesianNode:
    parents = (
        "building_damage",
        "fire_severity",
        "smoke_level",
        "sos_signal",
        "human_activity",
    )
    cpt = {}
    for values in product(LEVELS, repeat=len(parents)):
        building, fire, smoke, sos, activity = (LEVEL_VALUE[value] for value in values)
        probability = min(
            0.98,
            max(
                0.02,
                0.03
                + 0.25 * building
                + 0.12 * fire
                + 0.08 * smoke
                + 0.32 * sos
                + 0.20 * activity,
            ),
        )
        cpt[values] = {"no": 1.0 - probability, "yes": probability}
    return BayesianNode("trapped_people", BINARY, parents, cpt)


def _road_passable_node() -> BayesianNode:
    parents = ("road_damage", "fire_severity", "congestion", "drone_road_report")
    report_adjustment = {"blocked": -0.25, "uncertain": 0.0, "open": 0.18}
    cpt = {}
    for values in product(LEVELS, LEVELS, LEVELS, REPORT_STATES):
        damage, fire, congestion = (LEVEL_VALUE[value] for value in values[:3])
        probability = min(
            0.98,
            max(
                0.02,
                0.93
                - 0.40 * damage
                - 0.28 * fire
                - 0.15 * congestion
                + report_adjustment[values[3]],
            ),
        )
        cpt[values] = {"no": 1.0 - probability, "yes": probability}
    return BayesianNode("road_passable", BINARY, parents, cpt)


def build_expert_network() -> DiscreteBayesianNetwork:
    """Build the documented expert network used by fixed mode and as a prior."""
    nodes = [
        _root("hazard_intensity", LEVELS, (0.60, 0.30, 0.10)),
        _single_parent(
            "building_damage",
            "hazard_intensity",
            {
                "low": (0.82, 0.16, 0.02),
                "medium": (0.22, 0.58, 0.20),
                "high": (0.05, 0.28, 0.67),
            },
        ),
        _single_parent(
            "fire_severity",
            "hazard_intensity",
            {
                "low": (0.80, 0.17, 0.03),
                "medium": (0.30, 0.52, 0.18),
                "high": (0.10, 0.34, 0.56),
            },
        ),
        _single_parent(
            "smoke_level",
            "fire_severity",
            {
                "low": (0.86, 0.12, 0.02),
                "medium": (0.18, 0.65, 0.17),
                "high": (0.04, 0.26, 0.70),
            },
        ),
        _root("sos_signal", LEVELS, (0.55, 0.30, 0.15)),
        _root("human_activity", LEVELS, (0.48, 0.37, 0.15)),
        _trapped_node(),
        _single_parent(
            "road_damage",
            "hazard_intensity",
            {
                "low": (0.84, 0.14, 0.02),
                "medium": (0.26, 0.56, 0.18),
                "high": (0.06, 0.30, 0.64),
            },
        ),
        _root("congestion", LEVELS, (0.50, 0.35, 0.15)),
        _root("drone_road_report", REPORT_STATES, (0.12, 0.63, 0.25)),
        _road_passable_node(),
    ]
    return DiscreteBayesianNetwork(nodes)

