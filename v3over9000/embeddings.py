"""
Deterministic low-dimensional graph embeddings for tensor diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class GraphEmbedding:
    node_xy: np.ndarray
    edge_mid_xy: np.ndarray
    edge_dir_xyz: np.ndarray


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _path_embedding(n: int) -> np.ndarray:
    x = np.arange(float(n), dtype=float)
    y = np.zeros(n, dtype=float)
    return np.column_stack([x, y])


def _cycle_embedding(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _star_embedding(n: int) -> np.ndarray:
    xy = np.zeros((n, 2), dtype=float)
    if n == 1:
        return xy
    theta = np.linspace(0.0, 2.0 * np.pi, num=n - 1, endpoint=False)
    xy[1:, 0] = np.cos(theta)
    xy[1:, 1] = np.sin(theta)
    return xy


def build_embedding(
    n: int,
    topology_name: str,
    edges: List[Tuple[int, int]],
) -> GraphEmbedding:
    """
    Build a stable embedding used by the TT spin-2 analyzer.

    The goal is reproducibility for diagnostics, not geometric uniqueness.
    """
    name = (topology_name or "path").lower()
    if name == "path":
        node_xy = _path_embedding(n)
    elif name == "cycle":
        node_xy = _cycle_embedding(n)
    elif name in {"star", "pyramid"}:
        node_xy = _star_embedding(n)
    else:
        # Fallback: circular layout.
        node_xy = _cycle_embedding(max(n, 3))[:n]

    if len(edges) == 0:
        return GraphEmbedding(
            node_xy=node_xy,
            edge_mid_xy=np.zeros((0, 2), dtype=float),
            edge_dir_xyz=np.zeros((0, 3), dtype=float),
        )

    edge_mid_xy = np.zeros((len(edges), 2), dtype=float)
    edge_dir_xyz = np.zeros((len(edges), 3), dtype=float)

    for edge_idx, (u, v) in enumerate(edges):
        p_u = node_xy[int(u)]
        p_v = node_xy[int(v)]
        edge_mid_xy[edge_idx] = 0.5 * (p_u + p_v)
        d2 = _unit(p_v - p_u)
        edge_dir_xyz[edge_idx] = np.array([d2[0], d2[1], 0.0], dtype=float)

    return GraphEmbedding(
        node_xy=node_xy,
        edge_mid_xy=edge_mid_xy,
        edge_dir_xyz=edge_dir_xyz,
    )

