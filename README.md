# FractureLens — Pelvic CT Fracture Viewer & Report Generator

## 1. Overview

FractureLens is a desktop application for browsing pelvic CT scans, inspecting
bone fracture fragments in both 2D and 3D, and automatically generating a
structured fracture report (fragment counts, volumes, displacement, severity
metrics) per case. It combines two things already prototyped in this
conversation:

- A synced 2D/3D **viewer** (slice scrolling + interactive fracture-highlighted
  3D mesh)
- An automated **report generator** (per-fragment metrics, exported as
  PDF/HTML)

The target output is a single tool where a user picks a case and gets both a
visual inspection surface and a downloadable structured report from it.

**Why this pairing makes sense:** the viewer and the report share the same
underlying data-processing layer (load volume → decode labels → compute
per-fragment geometry). Building them together avoids duplicating that layer,
and the report becomes a natural "export" action from inside the viewer
rather than a separate disconnected tool.

---

## 2. Goals

- [ ] Load a case (`train/{id}.mha` + `labels/{id}.mha`) and browse it slice
      by slice in 3 orientations (axial/coronal/sagittal), CT image with
      label overlay.
- [ ] Render an interactive 3D model of the labeled bone fragments, with
      fracture-surface highlighting and an assembled/exploded toggle
      (already prototyped — see §9).
- [ ] Compute, per case, per-fragment: volume (mm³), surface area (mm²),
      centroid, bounding box, and minimum distance to each neighboring
      fragment of the same bone (the fracture gap).
- [ ] Derive simple case-level summary stats: total fragment count per bone,
      which bones are fractured vs. intact, a rough severity score.
- [ ] Export a shareable report per case (PDF and/or HTML) containing the
      above metrics plus embedded slice thumbnails and a static 3D render.
- [ ] Browse/filter across the whole dataset (case list with fragment counts,
      sortable/filterable) rather than only opening one file at a time.

## 3. Non-goals (out of scope for v1)

- No machine learning — this tool only visualizes/reports on the
  **ground-truth labels already provided** in the dataset. It does not
  predict labels on unlabeled data. (A future model could plug into the same
  UI later, but that's a separate project.)
- No DICOM support — sticking to `.mha` since that's what PENGWIN ships.
  DICOM import can be a stretch goal.
- No cloud/multi-user features — this is a local, single-user desktop tool.
- No mesh editing (surgeons manually repositioning fragments) — display and
  report only, not a planning tool. Could be a v2 direction.
- No real-time collaboration or annotation tools.

## 4. Target user / use case

Primary: **you**, using it as a portfolio project and a genuinely useful way
to explore this dataset. Secondary framing (useful for scoping decisions):
someone doing quick QA on a segmentation dataset, or a student learning
pelvic fracture anatomy, who wants to open a case and immediately understand
"what's broken and how badly" without writing analysis code each time.

---

## 5. Feature breakdown

### 5.1 Viewer (interactive app)

| Feature | Priority | Notes |
|---|---|---|
| Case picker / file browser | MVP | List of case IDs found under `train/` + `labels/`, show fragment count per case at a glance |
| 2D slice view, axial | MVP | Already prototyped (`view_pengwin.py`) |
| 2D slice view, coronal + sagittal | MVP | Extend existing axis-switch logic; ideally 3 synced panes at once |
| CT windowing controls (level/width slider) | MVP | Currently hardcoded to bone window; expose as UI sliders |
| Label overlay toggle + opacity slider | MVP | Already prototyped (`o` key); promote to UI control |
| 3D fragment mesh view | MVP | Already prototyped (`view_pengwin_3d.py`) — port from standalone Plotly HTML into embedded view |
| Fracture-surface heat highlighting | MVP | Already prototyped — reuse distance-transform logic |
| Assembled / exploded toggle | MVP | Already prototyped |
| Click a fragment in 3D → jump to that slice in 2D | Stretch | Nice synchronization feature, not required for v1 |
| Measurement tool (click two points, get mm distance) | Stretch | Useful add-on, moderate effort |
| Cross-case comparison view | Stretch | e.g. side-by-side two cases |

### 5.2 Report generator

| Feature | Priority | Notes |
|---|---|---|
| Per-fragment metrics: volume, surface area, centroid, bbox | MVP | Volume = voxel count × voxel volume; surface area from the mesh |
| Per-fragment nearest-neighbor distance (fracture gap) | MVP | Reuse distance-transform code already built |
| Per-bone summary: intact vs. fractured, fragment count | MVP | Simple aggregation |
| Case-level severity score | MVP (simple version) | Start naive: e.g. total fragment count across all 3 bones, or largest-gap metric. Refine later — this is inherently a design choice, not a "correct" formula |
| Static 3D render image embedded in report | MVP | Render + save PNG via the existing Plotly figure (headless export) |
| Slice thumbnails embedded in report | MVP | A few representative slices (e.g. through the largest fragment of each fractured bone) |
| Export as PDF | MVP | via `reportlab` or HTML→PDF (`weasyprint`) |
| Export as HTML (interactive) | Stretch | Reuse the existing interactive Plotly HTML output, wrap with metrics table |
| Batch report generation (whole dataset → one report each, or one combined CSV) | Stretch | Useful for the "case browser" sorting feature |

---

## 6. Architecture

```
fracturelens/
├── core/                     # shared processing logic (no UI dependencies)
│   ├── io.py                 # load .mha volumes, decode PENGWIN label scheme
│   ├── geometry.py           # marching cubes, per-fragment mesh extraction
│   ├── metrics.py            # volume/surface-area/distance/severity calculations
│   └── fracture_highlight.py # distance-transform-based fracture coloring
│
├── viewer/                   # interactive app
│   ├── app.py                # entry point
│   ├── slice_view.py         # 2D slice rendering + windowing
│   ├── mesh_view.py          # 3D fragment rendering (embedded)
│   └── case_browser.py       # dataset-wide case list UI
│
├── report/                   # report generation
│   ├── builder.py            # assembles metrics + images into a report object
│   ├── templates/            # HTML/PDF templates
│   └── export.py             # PDF/HTML writers
│
└── cli.py                    # `fracturelens report <case_id>` etc, for scripting/batch use
```

The key design decision: **`core/` has zero UI dependencies.** Both the
viewer and the report generator call into the same functions to load data
and compute geometry/metrics. This is what makes "combine 1 and 3" actually
easy instead of two separate codebases — the report is just a different
*consumer* of the same core pipeline the viewer uses to render.

### 6.1 Tech stack options

| Layer | Option A (recommended to start) | Option B |
|---|---|---|
| App shell | Python + PyQt6 / PySide6 (desktop) | Web app: FastAPI backend + React/Three.js frontend |
| 2D slice rendering | Matplotlib embedded in Qt canvas | Canvas/WebGL in browser |
| 3D rendering | Plotly (embed via `QWebEngineView`, reuses existing code) or `pyvista`/VTK for native performance | Three.js |
| Report export | `weasyprint` (HTML→PDF) or `reportlab` | same |
| Volume I/O | `SimpleITK` (already in use) | same |

**Recommendation:** start with **Option A**, since it reuses everything
already built (SimpleITK, Plotly, scikit-image, scipy) with the least new
surface area. A Qt shell embedding the existing Plotly HTML via
`QWebEngineView` is the fastest path from "two scripts" to "one app." Option
B is a good v2 if you want the tool to be shareable/browser-based later.

---

## 7. Data model (per fragment, computed once per case and cached)

```python
@dataclass
class FragmentMetrics:
    label: int                 # raw label value, e.g. 22
    category_id: int           # 1=SA, 2=LI, 3=RI
    category_name: str
    fragment_id: int           # 1-10
    voxel_count: int
    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: tuple[float, float, float]
    bbox_mm: tuple[float, float, float, float, float, float]
    nearest_neighbor_dist_mm: float | None  # None if bone is intact (no other fragment)
    nearest_neighbor_label: int | None

@dataclass
class CaseReport:
    case_id: str
    fragments: list[FragmentMetrics]
    fractured_bones: list[str]     # e.g. ["RI"]
    intact_bones: list[str]        # e.g. ["SA", "LI"]
    total_fragment_count: int
    severity_score: float          # v1: naive, documented formula, easy to revisit
```

Cache computed metrics to disk (e.g. `cache/{case_id}_metrics.json`) so
re-opening a case or regenerating a report doesn't redo the marching-cubes +
distance-transform work every time — that's the slow part.

---

## 8. Development phases

**Phase 0 — Consolidate what exists (few hours)**
Refactor `view_pengwin.py` and `view_pengwin_3d.py` into the `core/` module
structure above. No new features — just make the existing logic reusable and
UI-independent. This unblocks everything else.

**Phase 1 — Metrics + CLI report (short)**
Implement `core/metrics.py` and a bare CLI (`python cli.py report 009 -o
report.html`) that outputs a single static HTML report with the metrics
table + embedded static images. No interactive app yet — this is the
fastest path to a demoable, useful artifact, and validates the metrics logic
before investing in UI.

**Phase 2 — Desktop viewer shell**
Build the Qt app: case browser, 2D slice panes with windowing controls,
embedded 3D view (reuse Plotly HTML via `QWebEngineView`, or migrate to
`pyvista` if Plotly-in-Qt feels sluggish). Wire an "Export Report" button
that calls the same `core/` + `report/` code from Phase 1.

**Phase 3 — Polish + batch tooling**
Batch report generation across the dataset, case-list sorting/filtering by
severity or fragment count, caching, packaging (e.g. PyInstaller so it runs
without a manual venv setup).

**Stretch phase — pick from §5 stretch items** based on what's actually fun
by that point (click-to-sync 2D/3D, measurement tools, DICOM import, etc).

---

## 9. Reusable assets already built (starting point for Phase 0)

- `view_pengwin.py` — 2D slice viewer with CT windowing, label overlay,
  legend, category+fragment decoding (`label = 10*(category_id-1) +
  fragment_id`).
- `view_pengwin_3d.py` — 3D marching-cubes mesh reconstruction per fragment,
  fracture-surface heat highlighting via `scipy.ndimage.distance_transform_edt`,
  assembled/exploded toggle via Plotly `updatemenus`.
- Label decode scheme confirmed from `pengwin_utils.py` (official PENGWIN
  helper file): categories `SA=1, LI=2, RI=3`, fragments `1-10` per category.

These already cover most of `core/io.py`, `core/geometry.py`, and
`core/fracture_highlight.py` — Phase 0 is mostly extraction/refactoring, not
new logic.

---

## 10. Open questions / decisions to make before or during Phase 0

- **Severity score formula**: no established clinical formula was defined
  here — needs a first-pass definition (e.g. weighted combination of
  fragment count + max displacement) that you're comfortable calling
  "illustrative, not clinical."
- **Desktop (Qt) vs. web (FastAPI+React)** — affects nearly everything
  downstream; recommend deciding this before Phase 0 refactor so `core/`
  is structured with the right consumer in mind (though the split above
  should work for either).
- **Performance ceiling**: full-res marching cubes + distance transform per
  case — is per-case runtime (cache-once, reuse after) acceptable, or does
  the case browser need pre-computed metrics for *all* cases up front?
  Affects whether Phase 3's batch tooling should really be Phase 1.

---

## 11. Definition of done (v1)

A user can: launch the app → pick a case from a list → see 2D slices with
label overlay and windowing controls → see the 3D fracture-highlighted mesh
with assembled/exploded toggle → click "Export Report" → get a PDF/HTML file
with per-fragment metrics, a severity score, and embedded visuals — without
touching the command line after initial setup.
