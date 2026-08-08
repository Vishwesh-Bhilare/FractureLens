"""Mesh and geometry helpers ported from the PENGWIN 3D prototype."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import marching_cubes

MIN_VOXELS_FOR_MESH = 20

try:
    import trimesh
    import trimesh.smoothing
    HAVE_TRIMESH = True
except ImportError:  # pragma: no cover - depends on optional environment
    trimesh = None  # type: ignore[assignment]
    HAVE_TRIMESH = False


@dataclass
class FragmentMesh:
    """Surface mesh and volume accounting for a single bone fragment."""
    verts: np.ndarray
    faces: np.ndarray
    ground_truth_volume_mm3: float
    raw_mesh_volume_mm3: float
    smoothed_mesh_volume_mm3: float
    smoothing_applied: bool


def build_fragment_mesh(frag_mask: np.ndarray, voxel_volume_mm3: float, taubin_iterations: int = 6) -> FragmentMesh:
    """Marching cubes (level=0.5) on the RAW mask, then optional Taubin smoothing."""
    verts, faces, _, _ = marching_cubes(frag_mask.astype(np.uint8), level=0.5)
    gt_vol = float(frag_mask.sum() * voxel_volume_mm3)
    raw_vol = gt_vol
    if HAVE_TRIMESH:
        try:
            raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            raw_vol = float(abs(raw_mesh.volume) * voxel_volume_mm3)
        except Exception:
            raw_mesh = None
    else:
        raw_mesh = None
    if not HAVE_TRIMESH or taubin_iterations <= 0 or raw_mesh is None:
        return FragmentMesh(verts, faces, gt_vol, raw_vol, raw_vol, False)
    smooth_mesh = raw_mesh.copy()
    trimesh.smoothing.filter_taubin(smooth_mesh, lamb=0.5, nu=0.53, iterations=taubin_iterations)
    smooth_vol = float(abs(smooth_mesh.volume) * voxel_volume_mm3)
    return FragmentMesh(np.asarray(smooth_mesh.vertices), np.asarray(smooth_mesh.faces), gt_vol, raw_vol, smooth_vol, True)


def fragment_bounding_box(cat_mask: np.ndarray, pad: int = 4) -> tuple[slice, slice, slice]:
    """Find a padded bounding box for a whole-bone mask, clipped to array bounds."""
    found = find_objects(cat_mask.astype(int))
    if not found or found[0] is None:
        raise ValueError("Cannot compute bounding box for an empty mask")
    slices = found[0]
    z0 = max(slices[0].start - pad, 0); z1 = min(slices[0].stop + pad, cat_mask.shape[0])
    y0 = max(slices[1].start - pad, 0); y1 = min(slices[1].stop + pad, cat_mask.shape[1])
    x0 = max(slices[2].start - pad, 0); x1 = min(slices[2].stop + pad, cat_mask.shape[2])
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def fracture_proximity_intensity(verts: np.ndarray, crop_label_vol: np.ndarray, this_label: int, sibling_labels: list[int], voxel_spacing: tuple[float, float, float], highlight_range_mm: float = 12.0) -> np.ndarray:
    """Return per-vertex fracture proximity intensity in [0, 1]."""
    if len(sibling_labels) <= 1:
        return np.ones(verts.shape[0])
    other_mask = (crop_label_vol != this_label) & (crop_label_vol != 0) & np.isin(crop_label_vol, sibling_labels)
    if other_mask.any():
        dist_mm = distance_transform_edt(~other_mask, sampling=voxel_spacing)
        idx = np.round(verts).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, dist_mm.shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, dist_mm.shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, dist_mm.shape[2] - 1)
        d = dist_mm[idx[:, 0], idx[:, 1], idx[:, 2]]
        return np.clip(d / highlight_range_mm, 0, 1)
    return np.ones(verts.shape[0])
