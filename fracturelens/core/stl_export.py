"""STL export helpers for FractureLens fragment meshes."""

from pathlib import Path

import numpy as np

from fracturelens.core.geometry import FragmentMesh


def _to_trimesh_mm(mesh: FragmentMesh, voxel_spacing: tuple[float, float, float]):
    import trimesh

    return trimesh.Trimesh(
        vertices=(np.asarray(mesh.verts, dtype=float) + np.asarray(mesh.origin_zyx, dtype=float)) * np.asarray(voxel_spacing, dtype=float),
        faces=np.asarray(mesh.faces),
        process=False,
    )


def export_fragment_stl(mesh: FragmentMesh, voxel_spacing: tuple[float, float, float], output_path: Path) -> Path:
    """Export one already-smoothed fragment mesh as an STL scaled to millimeters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _to_trimesh_mm(mesh, voxel_spacing).export(str(output_path))
    return output_path


def export_bone_stl(fragment_meshes: list[FragmentMesh], voxel_spacing: tuple[float, float, float], output_path: Path) -> Path:
    """Merge multiple already-smoothed fragment meshes into one millimeter-scale STL."""
    import trimesh

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = [_to_trimesh_mm(mesh, voxel_spacing) for mesh in fragment_meshes]
    trimesh.util.concatenate(meshes).export(str(output_path))
    return output_path
