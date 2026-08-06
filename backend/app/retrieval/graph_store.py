"""Clinical knowledge graph over conditions, medications, symptoms and guidelines.

Nodes: condition | medication | symptom | guideline_section
Edges: treats | contraindicated_with | symptom_of | references | interacts_with

**What the graph is for.** Vector search answers "what text looks like this query".
It cannot answer "what else is contraindicated with the drug this guideline
recommends", because the answer lives in a document that shares no vocabulary with
the question. That multi-hop, relationship-shaped question is the only reason this
component exists, and the Retrieval Planner routes to it only when the query has
that shape — otherwise it is pure latency.

**Honest caveat.** Edges come from a free-tier LLM extracting triples from guideline
text. They are noisier than a curated ontology such as UMLS. `scripts/build_graph.py`
holds out a manually-checkable sample and `RESULTS.md` reports measured extraction
precision rather than assuming the edges are clean.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

NODE_TYPES = ("condition", "medication", "symptom", "guideline_section")
EDGE_TYPES = ("treats", "contraindicated_with", "symptom_of", "references", "interacts_with")


@dataclass(slots=True)
class GraphPath:
    """One traversal result, kept with its provenance so it can be cited."""

    nodes: list[str]
    edges: list[tuple[str, str, str]]
    chunk_ids: list[str]
    score: float

    def describe(self) -> str:
        return " → ".join(
            f"{s} —[{rel}]→ {t}" if i == 0 else f"[{rel}]→ {t}"
            for i, (s, rel, t) in enumerate(self.edges)
        )


def normalize_entity(name: str) -> str:
    """Canonical node key. Entity resolution is deliberately shallow here.

    Real systems resolve against UMLS/RxNorm CUIs. We lowercase and strip, and lean
    on the RxNorm cross-reference in ingestion for medication names specifically —
    which is where inconsistent naming actually costs us edges.
    """
    return " ".join(name.lower().strip().split())


class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.g = graph if graph is not None else nx.MultiDiGraph()

    # -- construction ------------------------------------------------------

    def add_entity(self, name: str, node_type: str, **attrs: Any) -> str:
        key = normalize_entity(name)
        if not key:
            return ""
        if self.g.has_node(key):
            self.g.nodes[key].setdefault("chunk_ids", [])
            if "chunk_id" in attrs:
                self.g.nodes[key]["chunk_ids"].append(attrs["chunk_id"])
            return key
        chunk_ids = [attrs["chunk_id"]] if "chunk_id" in attrs else []
        self.g.add_node(
            key,
            label=name.strip(),
            node_type=node_type if node_type in NODE_TYPES else "condition",
            chunk_ids=chunk_ids,
            rxcui=attrs.get("rxcui"),
        )
        return key

    def add_relation(self, source: str, relation: str, target: str, **attrs: Any) -> None:
        s, t = normalize_entity(source), normalize_entity(target)
        if not s or not t or s == t:
            return
        if not self.g.has_node(s) or not self.g.has_node(t):
            return
        rel = relation if relation in EDGE_TYPES else "references"
        # Reinforce an existing edge rather than duplicating it. Repeated extraction
        # of the same triple across documents is corroboration, and weight records it.
        for _, _, data in self.g.edges(s, data=True):
            if data.get("relation") == rel and data.get("_target") == t:
                data["weight"] = data.get("weight", 1) + 1
                return
        self.g.add_edge(
            s, t,
            relation=rel,
            _target=t,
            weight=1,
            chunk_id=attrs.get("chunk_id"),
            source_doc=attrs.get("source_doc"),
        )

    # -- traversal ---------------------------------------------------------

    def match_entities(self, query: str, limit: int = 8) -> list[str]:
        """Find seed nodes for a query by substring match on node labels.

        Longer labels win: matching 'septic shock' is more informative than matching
        the 'shock' contained within it.
        """
        q = normalize_entity(query)
        if not q:
            return []
        q_tokens = set(q.split())
        scored: list[tuple[float, str]] = []
        for node in self.g.nodes:
            if node in q:
                scored.append((len(node) + 10.0, node))
                continue
            n_tokens = set(node.split())
            overlap = q_tokens & n_tokens
            if overlap:
                scored.append((len(overlap) / max(len(n_tokens), 1) * len(overlap), node))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:limit]]

    def traverse(
        self, seeds: list[str], *, max_hops: int | None = None, max_nodes: int | None = None
    ) -> list[GraphPath]:
        hops = max_hops if max_hops is not None else settings.graph_max_hops
        cap = max_nodes if max_nodes is not None else settings.graph_max_nodes
        paths: list[GraphPath] = []
        visited: set[str] = set()

        for seed in seeds:
            if seed not in self.g or len(visited) >= cap:
                continue
            frontier: list[tuple[str, list[str], list[tuple[str, str, str]]]] = [(seed, [seed], [])]
            for _ in range(hops):
                next_frontier = []
                for node, node_path, edge_path in frontier:
                    for _, tgt, data in self.g.out_edges(node, data=True):
                        if tgt in node_path or len(visited) >= cap:
                            continue
                        visited.add(tgt)
                        new_nodes = [*node_path, tgt]
                        new_edges = [*edge_path, (node, data.get("relation", "references"), tgt)]
                        chunk_ids = self._collect_chunk_ids(new_nodes)
                        paths.append(
                            GraphPath(
                                nodes=new_nodes,
                                edges=new_edges,
                                chunk_ids=chunk_ids,
                                # Decay with depth, reward corroborated edges.
                                score=data.get("weight", 1) / len(new_nodes),
                            )
                        )
                        next_frontier.append((tgt, new_nodes, new_edges))
                frontier = next_frontier
                if not frontier:
                    break

        paths.sort(key=lambda p: -p.score)
        return paths

    def _collect_chunk_ids(self, nodes: list[str]) -> list[str]:
        out: list[str] = []
        for n in nodes:
            out.extend(self.g.nodes.get(n, {}).get("chunk_ids", [])[:3])
        seen: set[str] = set()
        return [c for c in out if not (c in seen or seen.add(c))]

    def neighbors_of(self, entity: str, relation: str | None = None) -> list[tuple[str, str]]:
        key = normalize_entity(entity)
        if key not in self.g:
            return []
        return [
            (data.get("relation", "references"), tgt)
            for _, tgt, data in self.g.out_edges(key, data=True)
            if relation is None or data.get("relation") == relation
        ]

    # -- io ----------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            by_type[d.get("node_type", "unknown")] += 1
        by_rel: dict[str, int] = defaultdict(int)
        for _, _, d in self.g.edges(data=True):
            by_rel[d.get("relation", "unknown")] += 1
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "nodes_by_type": dict(by_type),
            "edges_by_relation": dict(by_rel),
        }

    def save(self, path: Path | None = None) -> None:
        p = path or settings.graph_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self.g, fh)
        log.info("graph.saved", path=str(p), **self.stats())

    @classmethod
    def load(cls, path: Path | None = None) -> KnowledgeGraph | None:
        p = path or settings.graph_path
        if not p.exists():
            return None
        with p.open("rb") as fh:
            g = pickle.load(fh)
        kg = cls(g)
        log.info("graph.loaded", **kg.stats())
        return kg


_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph | None:
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph.load()
    return _graph


def reset_graph() -> None:
    global _graph
    _graph = None
