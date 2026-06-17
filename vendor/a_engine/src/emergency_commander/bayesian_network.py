from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable


class BayesianNetworkError(ValueError):
    """Raised when a Bayesian network or evidence assignment is invalid."""


def _row_key(values: tuple[str, ...]) -> str:
    return "|".join(values)


def _parse_row_key(value: str) -> tuple[str, ...]:
    return tuple(value.split("|")) if value else ()


@dataclass(frozen=True)
class BayesianNode:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    cpt: dict[tuple[str, ...], dict[str, float]]

    def probability(self, state: str, assignment: dict[str, str]) -> float:
        if state not in self.states:
            raise BayesianNetworkError(f"unknown state '{state}' for node '{self.name}'")
        key = tuple(assignment[parent] for parent in self.parents)
        try:
            return self.cpt[key][state]
        except KeyError as exc:
            raise BayesianNetworkError(
                f"missing CPT row for node '{self.name}' and parents {key}"
            ) from exc


class DiscreteBayesianNetwork:
    def __init__(self, nodes: Iterable[BayesianNode]):
        self.nodes = tuple(nodes)
        self._nodes = {node.name: node for node in self.nodes}
        self._validate()

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    def _validate(self) -> None:
        if len(self._nodes) != len(self.nodes):
            raise BayesianNetworkError("node names must be unique")
        seen: set[str] = set()
        for node in self.nodes:
            if not node.states:
                raise BayesianNetworkError(f"node '{node.name}' must define states")
            if len(set(node.states)) != len(node.states):
                raise BayesianNetworkError(f"node '{node.name}' has duplicate states")
            for parent in node.parents:
                if parent not in seen:
                    raise BayesianNetworkError(
                        f"parent '{parent}' must appear before node '{node.name}'"
                    )
            expected_rows = set(
                product(*(self._nodes[parent].states for parent in node.parents))
            )
            if not node.parents:
                expected_rows = {()}
            if set(node.cpt) != expected_rows:
                raise BayesianNetworkError(f"CPT rows for node '{node.name}' are incomplete")
            for key, row in node.cpt.items():
                if set(row) != set(node.states):
                    raise BayesianNetworkError(
                        f"CPT row {key} for node '{node.name}' has invalid states"
                    )
                if any(probability <= 0.0 for probability in row.values()):
                    raise BayesianNetworkError(
                        f"CPT probabilities for node '{node.name}' must be positive"
                    )
                if abs(sum(row.values()) - 1.0) > 1e-8:
                    raise BayesianNetworkError(
                        f"CPT probabilities for node '{node.name}' must sum to 1"
                    )
            seen.add(node.name)

    def _validate_evidence(self, evidence: dict[str, str]) -> None:
        for name, state in evidence.items():
            if name not in self._nodes:
                raise BayesianNetworkError(f"unknown evidence node '{name}'")
            if state not in self._nodes[name].states:
                raise BayesianNetworkError(
                    f"unknown state '{state}' for evidence node '{name}'"
                )

    def _enumerate_all(
        self,
        nodes: tuple[BayesianNode, ...],
        index: int,
        assignment: dict[str, str],
    ) -> float:
        if index == len(nodes):
            return 1.0
        node = nodes[index]
        if node.name in assignment:
            probability = node.probability(assignment[node.name], assignment)
            return probability * self._enumerate_all(nodes, index + 1, assignment)

        total = 0.0
        for state in node.states:
            assignment[node.name] = state
            total += node.probability(state, assignment) * self._enumerate_all(
                nodes, index + 1, assignment
            )
            assignment.pop(node.name)
        return total

    def query(self, query_node: str, evidence: dict[str, str] | None = None) -> dict[str, float]:
        if query_node not in self._nodes:
            raise BayesianNetworkError(f"unknown query node '{query_node}'")
        evidence = dict(evidence or {})
        self._validate_evidence(evidence)
        if query_node in evidence:
            observed = evidence[query_node]
            return {
                state: 1.0 if state == observed else 0.0
                for state in self._nodes[query_node].states
            }

        node = self._nodes[query_node]
        descendants = {query_node}
        changed = True
        while changed:
            changed = False
            for candidate in self.nodes:
                if candidate.name not in descendants and any(
                    parent in descendants for parent in candidate.parents
                ):
                    descendants.add(candidate.name)
                    changed = True
        observed_descendants = (set(evidence) & descendants) - {query_node}
        if all(parent in evidence for parent in node.parents) and not observed_descendants:
            key = tuple(evidence[parent] for parent in node.parents)
            return dict(node.cpt[key])

        distribution: dict[str, float] = {}
        relevant_names = {query_node, *evidence.keys()}
        changed = True
        while changed:
            changed = False
            for name in tuple(relevant_names):
                for parent in self._nodes[name].parents:
                    if parent not in relevant_names:
                        relevant_names.add(parent)
                        changed = True
        relevant_nodes = tuple(node for node in self.nodes if node.name in relevant_names)
        for state in self._nodes[query_node].states:
            assignment = dict(evidence)
            assignment[query_node] = state
            distribution[state] = self._enumerate_all(relevant_nodes, 0, assignment)
        total = sum(distribution.values())
        if total <= 0.0:
            raise BayesianNetworkError("evidence has zero probability")
        return {state: probability / total for state, probability in distribution.items()}

    def explain(
        self, query_node: str, target_state: str, evidence: dict[str, str]
    ) -> dict[str, Any]:
        posterior = self.query(query_node, evidence)[target_state]
        contributions = []
        for evidence_name, evidence_state in evidence.items():
            reduced = {name: state for name, state in evidence.items() if name != evidence_name}
            without = self.query(query_node, reduced)[target_state]
            contributions.append(
                {
                    "evidence": evidence_name,
                    "state": evidence_state,
                    "posterior_without": round(without, 8),
                    "delta": round(posterior - without, 8),
                }
            )
        contributions.sort(key=lambda item: abs(item["delta"]), reverse=True)
        return {
            "query": query_node,
            "target_state": target_state,
            "posterior": posterior,
            "contributions": contributions,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "discrete_bayesian_network_v1",
            "nodes": [
                {
                    "name": node.name,
                    "states": list(node.states),
                    "parents": list(node.parents),
                    "cpt": {
                        _row_key(key): dict(row) for key, row in node.cpt.items()
                    },
                }
                for node in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscreteBayesianNetwork":
        nodes = []
        for raw_node in payload.get("nodes", []):
            nodes.append(
                BayesianNode(
                    name=raw_node["name"],
                    states=tuple(raw_node["states"]),
                    parents=tuple(raw_node.get("parents", [])),
                    cpt={
                        _parse_row_key(key): {
                            state: float(probability)
                            for state, probability in row.items()
                        }
                        for key, row in raw_node["cpt"].items()
                    },
                )
            )
        return cls(nodes)


def fit_cpts(
    template: DiscreteBayesianNetwork,
    records: Iterable[dict[str, Any]],
    *,
    prior_strength: float = 1.0,
    weight_field: str = "sample_weight",
) -> DiscreteBayesianNetwork:
    """Estimate every CPT with an expert-shaped Dirichlet prior."""
    if prior_strength <= 0.0:
        raise BayesianNetworkError("prior_strength must be positive")
    record_list = list(records)
    learned_nodes = []
    for node in template.nodes:
        counts = {
            key: {
                state: prior_strength * probability
                for state, probability in row.items()
            }
            for key, row in node.cpt.items()
        }
        for record in record_list:
            if node.name not in record or any(parent not in record for parent in node.parents):
                continue
            state = record[node.name]
            key = tuple(record[parent] for parent in node.parents)
            if key not in counts or state not in counts[key]:
                continue
            weight = float(record.get(weight_field, 1.0))
            if weight > 0:
                counts[key][state] += weight
        normalized = {}
        for key, row in counts.items():
            total = sum(row.values())
            normalized[key] = {state: value / total for state, value in row.items()}
        learned_nodes.append(
            BayesianNode(node.name, node.states, node.parents, normalized)
        )
    return DiscreteBayesianNetwork(learned_nodes)
