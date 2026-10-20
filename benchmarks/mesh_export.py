"""Mesh files for the website's geometry viewer (assets/meshes/{name}.json): one format for structured and unstructured benchmarks.

Format (schemaVersion 1), all indices 0-based:
  vertices    flat [x0, y0, x1, y1, ...]
  cellSize    vertices per cell (4 = quads, 3 = triangles)
  cells       flat vertex indices, cellSize per cell, in boundary order
  nodes       where the benchmark data lives: "vertices" (grid nodes) or "cells" (cell centres)
  grid        optional {"shape": [nx, ny]}: vertex k is grid node (k // ny, k % ny), as in fields[..., i, j]
  boundaries  [{"name", "faces": flat vertex pairs}]: boundary faces grouped by boundary condition
  extrusion   optional {"depth", "name"}: the mesh is one cell layer of a 3D case, `name` labels its two planes
  reference   optional [{"name", "polyline": flat xy}]: true geometry the mesh approximates (a cylinder under a masked grid)
"""
import json
from pathlib import Path

import numpy as np


def boundary_faces(cells):
    """Faces (vertex pairs) that belong to a single cell."""
    cells = np.asarray(cells)
    faces = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1]


def label_faces(vertices, faces, rules):
    """Group faces by the first rule (name, predicate on face midpoints (F, 2) -> mask) they match."""
    midpoints = np.asarray(vertices)[faces].mean(axis=1)
    remaining = np.ones(len(faces), dtype=bool)
    groups = {}
    for name, rule in rules:
        mask = remaining & rule(midpoints)
        if mask.any():
            groups[name] = np.concatenate([groups[name], faces[mask]]) if name in groups else faces[mask]
        remaining &= ~mask
    if remaining.any():
        raise ValueError(f"{remaining.sum()} boundary faces match no rule.")
    return groups


def structured_grid(x, y, valid):
    """Vertices and quads of an (nx, ny) grid with 'ij' indexing; quads touching an invalid node are dropped."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    vertices = np.stack([X.ravel(), Y.ravel()], axis=1)
    k = np.arange(X.size).reshape(X.shape)
    cells = np.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], axis=-1).reshape(-1, 4)
    return vertices, cells[np.asarray(valid).ravel()[cells].all(axis=1)]


def circle(center, radius, segments=64):
    """Closed polyline around a circle, as a flat xy list."""
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)


def write_mesh(path, vertices, cells, boundaries, nodes, *, grid_shape=None, extrusion=None, reference=None, decimals=5):
    """Write the viewer mesh file; `boundaries` maps names to (F, 2) vertex-pair arrays."""
    if nodes not in {"vertices", "cells"}:
        raise ValueError("nodes must be 'vertices' or 'cells'.")
    cells = np.asarray(cells)
    coordinates = lambda values: [round(float(v), decimals) for v in np.asarray(values).ravel()]
    integers = lambda values: [int(v) for v in np.asarray(values).ravel()]
    record = {
        "schemaVersion": 1,
        "nodes": nodes,
        **({"grid": {"shape": [int(n) for n in grid_shape]}} if grid_shape is not None else {}),
        "vertices": coordinates(vertices),
        "cellSize": int(cells.shape[1]),
        "cells": integers(cells),
        "boundaries": [{"name": name, "faces": integers(faces)} for name, faces in boundaries.items()],
        **({"extrusion": extrusion} if extrusion else {}),
        **({"reference": [{"name": r["name"], "polyline": coordinates(r["polyline"])} for r in reference]} if reference else {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Mesh: {len(cells)} cells, {len(np.asarray(vertices))} vertices -> {path} ({path.stat().st_size / 1e3:.0f} kB)")
