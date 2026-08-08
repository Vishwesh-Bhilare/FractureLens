"""Headless 2D rendering helpers for CT slices and label overlays."""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from fracturelens.core.io import decode_label

CATEGORY_BASE_COLOR = {1: np.array([0.90, 0.25, 0.25]), 2: np.array([0.25, 0.55, 0.95]), 3: np.array([0.25, 0.80, 0.35])}


def window_ct(slice_2d: np.ndarray, level: int = 400, width: int = 1500) -> np.ndarray:
    """Apply a bone-friendly CT window/level and normalize to 0-1 for display."""
    lo, hi = level - width / 2, level + width / 2
    clipped = np.clip(slice_2d, lo, hi)
    return (clipped - lo) / (hi - lo)


def make_label_cmap(max_label: int) -> ListedColormap:
    """Color labels by bone category and fragment id, with transparent background."""
    colors = np.zeros((max_label + 1, 4))
    for label in range(1, max_label + 1):
        cat_id, frag_id = decode_label(label)
        base = CATEGORY_BASE_COLOR.get(cat_id, np.array([0.6, 0.6, 0.6]))
        shade = 0.5 + 0.5 * ((frag_id - 1) % 5) / 4
        rgb = np.clip(base * shade + (1 - shade) * 0.3, 0, 1)
        colors[label] = [rgb[0], rgb[1], rgb[2], 0.55]
    colors[0] = [0, 0, 0, 0]
    return ListedColormap(colors)


def _slice_for_axis(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return vol[index]
    if axis == 1:
        return vol[:, index, :]
    if axis == 2:
        return vol[:, :, index]
    raise ValueError("axis must be 0 (axial), 1 (coronal), or 2 (sagittal)")


def render_slice_png(image_vol: np.ndarray, label_vol: np.ndarray, axis: int, index: int) -> bytes:
    """Render one CT slice with label overlay to PNG bytes.

    axis: 0=axial(z), 1=coronal(y), 2=sagittal(x).
    """
    image_slice = _slice_for_axis(image_vol, axis, index)
    label_slice = _slice_for_axis(label_vol, axis, index)
    axis_names = {0: "Axial z", 1: "Coronal y", 2: "Sagittal x"}
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.imshow(window_ct(image_slice), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.imshow(label_slice, cmap=make_label_cmap(int(label_vol.max())), vmin=0, vmax=max(1, int(label_vol.max())), interpolation="nearest")
    ax.set_title(f"{axis_names[axis]}={index}")
    ax.axis("off")
    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def render_axial_slice_png(image_vol: np.ndarray, label_vol: np.ndarray, z_index: int) -> bytes:
    """Render one axial CT slice with label overlay to PNG bytes."""
    return render_slice_png(image_vol, label_vol, axis=0, index=z_index)
