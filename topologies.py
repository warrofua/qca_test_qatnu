from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Topology:
    name: str
    edges: List[Tuple[int, int]]
    probes: Tuple[int, int]


def _default_probes(N: int) -> Tuple[int, int]:
    if N == 1:
        return (0, 0)
    return (0, 1)


def get_topology(name: str, N: int) -> Topology:
    """
    Return the edge list and default probes for a given topology.

    Supported names: path, cycle, diamond (rhombus), bowtie (two triangles),
    pyramid (star).
    """
    name = (name or "path").lower()
    if N < 1:
        raise ValueError("Topology requires N >= 1")

    if name == "path":
        edges = [(i, i + 1) for i in range(N - 1)]
        probes = _default_probes(N)
    elif name == "cycle":
        if N < 3:
            raise ValueError("Cycle topology requires N >= 3")
        edges = [(i, i + 1) for i in range(N - 1)]
        edges.append((N - 1, 0))
        probes = _default_probes(N)
    elif name == "diamond":
        if N != 4:
            raise ValueError("Diamond (rhombus) topology currently defined only for N=4")
        # Vertices arranged in a rhombus: 0-1-2-3-0
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        probes = (0, 2)  # opposite vertices across the long diagonal
    elif name in {"bowtie", "kite"}:
        if N != 4:
            raise ValueError("Bowtie topology currently defined only for N=4")
        # Label vertices 0:top-left, 1:top-right, 2:center, 3:bottom
        edges = [(0, 2), (2, 1), (1, 3), (3, 0), (2, 3)]
        probes = (0, 1)
    elif name in {"pyramid", "star"}:
        if N < 2:
            raise ValueError("Pyramid/star topology requires N >= 2")
        edges = [(0, i) for i in range(1, N)]
        probes = (0, 1)
    else:
        raise ValueError(f"Unsupported topology '{name}'")

    return Topology(name=name, edges=edges, probes=probes)
