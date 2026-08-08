"""
Viewer for the PENGWIN pelvic CT dataset (.mha files).

PENGWIN = Pelvic bone fracture segmentation challenge.
- data/train/*.mha   -> CT image volumes (grayscale, Hounsfield units)
- data/labels/*.mha  -> segmentation masks (integer labels: bone fragments)

Usage (run from anywhere -- default --root resolves relative to this file):
    python view_pengwin.py 001
    python view_pengwin.py 001 --slice 150
    python view_pengwin.py 001 --axis 0        # 0=axial(z), 1=coronal(y), 2=sagittal(x)
    python view_pengwin.py 001 --root /custom/path/to/data

Controls in the viewer window:
    Left/Right arrow keys, or the slider  -> scroll through slices
    'o' key                                -> toggle label overlay on/off

Requires:
    pip install SimpleITK matplotlib numpy
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch

# scripts/view_pengwin.py -> project_root/data
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"

CATEGORIES = {1: "SA (sacrum)", 2: "LI (left hip)", 3: "RI (right hip)"}
CATEGORY_BASE_COLOR = {
    1: np.array([0.90, 0.25, 0.25]),  # sacrum -> red family
    2: np.array([0.25, 0.55, 0.95]),  # left hip -> blue family
    3: np.array([0.25, 0.80, 0.35]),  # right hip -> green family
}


def load_volume(path: Path) -> np.ndarray:
    """Load an .mha file and return a numpy array shaped (z, y, x)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # sitk returns (z, y, x)
    spacing = img.GetSpacing()
    print(f"Loaded {path.name}: shape={arr.shape}, dtype={arr.dtype}, "
          f"spacing={spacing}, min={arr.min()}, max={arr.max()}")
    return arr


def window_ct(slice_2d: np.ndarray, level=400, width=1500) -> np.ndarray:
    """Apply a bone-friendly CT window/level and normalize to 0-1 for display."""
    lo, hi = level - width / 2, level + width / 2
    clipped = np.clip(slice_2d, lo, hi)
    return (clipped - lo) / (hi - lo)


def decode_label(label: int) -> tuple[int, int]:
    """label = 10*(category_id-1) + fragment_id  ->  (category_id, fragment_id)"""
    category_id = (label - 1) // 10 + 1
    fragment_id = (label - 1) % 10 + 1
    return category_id, fragment_id


def make_label_cmap(max_label: int):
    """Color by bone category (hue), shade by fragment id, 0 = transparent."""
    colors = np.zeros((max_label + 1, 4))
    for label in range(1, max_label + 1):
        cat_id, frag_id = decode_label(label)
        base = CATEGORY_BASE_COLOR.get(cat_id, np.array([0.6, 0.6, 0.6]))
        # vary lightness with fragment id so fragments of the same bone are distinguishable
        shade = 0.5 + 0.5 * ((frag_id - 1) % 5) / 4  # 0.5 - 1.0
        rgb = np.clip(base * shade + (1 - shade) * 0.3, 0, 1)
        colors[label] = [rgb[0], rgb[1], rgb[2], 0.55]
    colors[0] = [0, 0, 0, 0]  # background fully transparent
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def print_legend(label_vol: np.ndarray):
    unique_labels = sorted(int(l) for l in np.unique(label_vol) if l != 0)
    print("\nFragments present in this volume:")
    for label in unique_labels:
        cat_id, frag_id = decode_label(label)
        cat_name = CATEGORIES.get(cat_id, f"unknown({cat_id})")
        voxel_count = int((label_vol == label).sum())
        print(f"  label {label:2d}  ->  {cat_name}, fragment {frag_id}   ({voxel_count} voxels)")
    print()


def get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def main():
    parser = argparse.ArgumentParser(description="View a PENGWIN CT volume + label mask.")
    parser.add_argument("case_id", help="Case id, e.g. 001 (matches data/train/001.mha and data/labels/001.mha)")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                         help=f"Dataset root folder (default: {DEFAULT_ROOT})")
    parser.add_argument("--slice", type=int, default=None, help="Initial slice index (default: middle)")
    parser.add_argument("--axis", type=int, default=0, choices=[0, 1, 2],
                         help="0=axial(z, default), 1=coronal(y), 2=sagittal(x)")
    parser.add_argument("--level", type=int, default=400, help="CT window level (default 400, bone)")
    parser.add_argument("--width", type=int, default=1500, help="CT window width (default 1500, bone)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150, try 200+ on a hi-dpi display)")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    img_path = root / "train" / f"{args.case_id}.mha"
    lbl_path = root / "labels" / f"{args.case_id}.mha"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label not found: {lbl_path}")

    image = load_volume(img_path)
    label = load_volume(lbl_path)

    if image.shape != label.shape:
        print("WARNING: image and label shapes differ:", image.shape, label.shape)

    axis = args.axis
    n_slices = image.shape[axis]
    start_idx = args.slice if args.slice is not None else n_slices // 2
    start_idx = max(0, min(start_idx, n_slices - 1))

    print_legend(label)

    max_label = int(label.max())
    lbl_cmap = make_label_cmap(max_label)

    plt.rcParams["figure.dpi"] = args.dpi
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.15)

    img_slice = window_ct(get_slice(image, axis, start_idx), args.level, args.width)
    lbl_slice = get_slice(label, axis, start_idx)

    # interpolation="bilinear" gives noticeably smoother slices than the default
    # nearest-neighbor look, especially when the window is enlarged/zoomed
    im_display = ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    lbl_display = ax.imshow(lbl_slice, cmap=lbl_cmap, vmin=0, vmax=max_label, interpolation="nearest")
    ax.set_title(f"Case {args.case_id} | axis={axis} | slice {start_idx}/{n_slices - 1}", fontsize=11)
    ax.axis("off")

    legend_handles = [
        Patch(facecolor=CATEGORY_BASE_COLOR[cid], label=name)
        for cid, name in CATEGORIES.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.75)

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=start_idx, valstep=1)

    overlay_on = {"state": True}

    def update(val):
        idx = int(slider.val)
        img_slice = window_ct(get_slice(image, axis, idx), args.level, args.width)
        lbl_slice = get_slice(label, axis, idx)
        im_display.set_data(img_slice)
        lbl_display.set_data(lbl_slice)
        ax.set_title(f"Case {args.case_id} | axis={axis} | slice {idx}/{n_slices - 1}", fontsize=11)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        idx = int(slider.val)
        if event.key == "right":
            slider.set_val(min(idx + 1, n_slices - 1))
        elif event.key == "left":
            slider.set_val(max(idx - 1, 0))
        elif event.key == "o":
            overlay_on["state"] = not overlay_on["state"]
            lbl_display.set_visible(overlay_on["state"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\nControls: Left/Right arrows or slider = scroll slices, 'o' = toggle label overlay")
    plt.show()


if __name__ == "__main__":
    main()
