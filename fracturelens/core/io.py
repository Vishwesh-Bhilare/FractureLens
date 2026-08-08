"""I/O helpers and label encoding for PENGWIN CT volumes."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_ROOT: Path = Path(__file__).resolve().parent.parent.parent / "data"

CATEGORIES: dict[int, str] = {1: "SA", 2: "LI", 3: "RI"}
CATEGORY_DISPLAY_NAMES: dict[int, str] = {1: "Sacrum", 2: "Left Hip Bone", 3: "Right Hip Bone"}


def decode_label(label: int) -> tuple[int, int]:
    """Decode a PENGWIN fragment label into ``(category_id, fragment_id)``."""
    category_id = (label - 1) // 10 + 1
    fragment_id = (label - 1) % 10 + 1
    return category_id, fragment_id


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load an .mha file. Returns (array shaped (z,y,x), spacing as (x,y,z) from SimpleITK)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def get_case_paths(root: Path, case_id: str) -> tuple[Path, Path]:
    """Returns (image_path, label_path) = (root/train/{case_id}.mha, root/labels/{case_id}.mha).
    Does NOT check existence -- caller decides how to handle missing files."""
    return root / "train" / f"{case_id}.mha", root / "labels" / f"{case_id}.mha"


def list_available_case_ids(root: Path) -> list[str]:
    """Scan root/labels/*.mha and return sorted case IDs with both image and label files."""
    label_dir = root / "labels"
    if not label_dir.exists():
        return []
    ids: list[str] = []
    for label_path in label_dir.glob("*.mha"):
        case_id = label_path.stem
        if (root / "train" / f"{case_id}.mha").exists():
            ids.append(case_id)
    return sorted(ids)
