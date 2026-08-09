"""Symmetry-based displacement scoring for one-sided hip fractures."""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from fracturelens.core.geometry import FragmentMesh

LOGGER = logging.getLogger(__name__)


def find_midline_x_mm(label_vol: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float | None:
    """Sagittal midline x-coordinate in mm, in the original volume's coordinate frame."""
    sacrum = np.argwhere((label_vol >= 1) & (label_vol <= 10))
    if sacrum.size:
        return float(sacrum[:, 2].mean() * spacing_xyz[0])

    hip_centroids: list[float] = []
    for cid in (2, 3):
        lo = (cid - 1) * 10 + 1
        hi = cid * 10
        coords = np.argwhere((label_vol >= lo) & (label_vol <= hi))
        if coords.size:
            hip_centroids.append(float(coords[:, 2].mean() * spacing_xyz[0]))
    if len(hip_centroids) == 2:
        return float(sum(hip_centroids) / 2.0)
    return None


def _mesh_mm(mesh: FragmentMesh, spacing_zyx: tuple[float, float, float]):
    import trimesh

    return trimesh.Trimesh(
        vertices=(np.asarray(mesh.verts, dtype=float) + np.asarray(mesh.origin_zyx, dtype=float)) * np.asarray(spacing_zyx, dtype=float),
        faces=np.asarray(mesh.faces),
        process=False,
    )


def compute_symmetry_displacement(
    label_vol: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    fragment_meshes: dict[int, FragmentMesh],
    labels_by_cat: dict[int, list[int]],
) -> dict[int, float] | None:
    """Return per-fragment symmetry displacement for a one-sided hip fracture.

    The metric is translational only: centroid distance to the nearest healthy
    mirrored surface after one global rigid alignment, not per-fragment rotation
    or true point correspondence.
    """
    try:
        import trimesh
        import trimesh.registration

        li = labels_by_cat.get(2, [])
        ri = labels_by_cat.get(3, [])
        if not ((len(li) > 1 and len(ri) == 1) or (len(ri) > 1 and len(li) == 1)):
            return None
        fractured_cid, intact_cid = (2, 3) if len(li) > 1 else (3, 2)
        fractured_labels = labels_by_cat[fractured_cid]
        intact_label = labels_by_cat[intact_cid][0]
        if intact_label not in fragment_meshes or any(label not in fragment_meshes for label in fractured_labels):
            return None

        x_mid = find_midline_x_mm(label_vol, spacing_xyz)
        if x_mid is None:
            return None

        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        target_mesh = _mesh_mm(fragment_meshes[intact_label], spacing_zyx)
        target_mesh.vertices[:, 2] = 2.0 * x_mid - target_mesh.vertices[:, 2]
        target_mesh.invert()

        fractured_meshes = [_mesh_mm(fragment_meshes[label], spacing_zyx) for label in fractured_labels]
        source_mesh = trimesh.util.concatenate(fractured_meshes)
        matrix, _, _ = trimesh.registration.icp(
            source_mesh.vertices,
            target_mesh.vertices,
            max_iterations=50,
            reflection=False,
            scale=False,
        )

        displacement: dict[int, float] = {}
        for label, mesh in zip(fractured_labels, fractured_meshes):
            transformed = mesh.copy()
            transformed.apply_transform(matrix)
            try:
                _, distances, _ = target_mesh.nearest.on_surface([transformed.centroid])
                distance = float(distances[0])
            except ModuleNotFoundError:
                # Some trimesh proximity backends require optional rtree. Fall back
                # to nearest target vertex so report generation and tests remain
                # usable in minimal environments.
                distance = float(cKDTree(target_mesh.vertices).query(transformed.centroid)[0])
            displacement[int(label)] = distance
        return displacement
    except Exception:
        LOGGER.exception("Symmetry displacement computation failed")
        return None
