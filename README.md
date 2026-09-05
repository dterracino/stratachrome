# Stratachrome

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Stratachrome** is an automated multi-color 3D relief printing pipeline designed for standard single-extruder FDM 3D printers and multi-material systems (Bambu Lab AMS, Orca Slicer).

By coupling **BiRefNet bilateral background segmentation** with **Beer-Lambert transmission models** and **CIE L\*C\*h color spaces**, Stratachrome decomposes any 2D image into an intelligent two-tier physical relief print. It solves the color-bleeding and dynamic-range compression limitations of standard single-stack heightmaps by seating a dedicated foreground tier atop a solid background pedestal, outputting watertight 2-manifold triangle meshes packaged inside ready-to-slice Bambu/Orca 3MF containers.

---

## Key Features

* **Intelligent Two-Tier Geometry**: Automatically isolates foreground subjects from background scenes using BiRefNet, creating a dedicated height budget for both and preventing color transmission bleed-through.
* **Predictive Optical Modeling**: Uses the Beer-Lambert law ($I = I_0 e^{-\alpha z}$) to accurately model the transmission distance ($TD$) and perceptual lightness ($L^*$) of real-world 3D printing filaments.
* **Perceptual Color Precision**: Direct integration with `color-match-tools` (`color_tools.conversions`) for accurate CIELCh and CIELAB color-space mapping.
* **Calibrated Layer Alignment**: Built with a dedicated 0.20 mm first-layer base for reliable bed adhesion and 0.10 mm layer increments matching standard slicer toolpaths.
* **Flexible Swap Modes**: Supports automated multi-material hardware changers (`--swap-mode ams`) or single-extruder pause triggers (`--swap-mode manual`).
* **Auto-Scaling Aspect Ratios**: Specify target maximum dimension in millimeters (`--size`); landscape and portrait images automatically scale to fit within your build plate envelope.
* **Slicer-Optimized Grid Resolution**: Built around standard 0.42 mm nozzle line widths and Arachne dynamic extrusion parameters, sampling up to 1000 px resolution for Nyquist fidelity without slicing lag or mesh bloat.
* **Watertight Manifold Meshes**: Generates counter-clockwise (CCW) wound top surfaces, flat baseplates, and perimeter skirts with zero degenerate triangles or non-manifold edges.
* **Native Bambu / Orca 3MF Packaging**: Exports Open Packaging Conventions (OPC) archives featuring decomposed model components (`3D/Objects/object_1.model`) and optional layer pause markers (`Metadata/custom_gcode_per_layer.xml`).

---

## Pipeline Architecture

```text
Input Image
    │
    ├──► [Segmentation] (BiRefNet) ─────────────► Alpha Matte
    │                                                   │
    └──► [Color Engine] (color_tools CIELCh) ──► Full L* Matrix
                                                        │
         ┌──────────────────────────────────────────────┘
         ▼
[Lightness Partitioning] ──► Background L* & Foreground L*
         │
         ▼
[Optical Modeling] ────────► Beer-Lambert Transmission LUTs
         │
         ▼
[Two-Tier Depth Mapper] ───► Solid Pedestal Elevation + Anti-Sheer Blend
         │
         ▼
[Watertight Mesh Builder] ─► 2-Manifold Triangular Mesh
         │
         ▼
[OPC 3MF Exporter] ────────► Bambu Studio / Orca Slicer .3mf Container (AMS / Manual)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/dterracino/stratachrome.git](https://github.com/dterracino/stratachrome.git)
cd stratachrome
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (CMD):
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

If you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheels first:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

Install Stratachrome in editable mode:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Quickstart & Usage

### 1. End-to-End Generation (Full 3MF)

Convert an image into a ready-to-print 3MF file using an AMS multi-material printer (max dimension 150 mm):

```bash
stratachrome -i assets/subject.png -o output/relief_project.3mf -s 150.0 --swap-mode ams
```

For single-extruder printers requiring manual layer pause prompts:

```bash
stratachrome -i assets/subject.png -o output/relief_project.3mf -s 150.0 --swap-mode manual
```

### 2. Segment and Inspect Alpha Layers

Isolate foreground and background components and inspect edge-feathering previews:

```bash
stratachrome-segment -i assets/subject.png -o output/segmentation/ --save-matte --feather 2
```

### 3. Mesh Verification Export (Binary STL)

Generate a quick manifold binary STL heightmap to verify topology or test physical dimensions:

```bash
stratachrome-mesh -i assets/subject.png -o output/test_mesh.stl -s 100.0 --max-height 2.4 --base-height 0.4
```

---

## Command-Line Interface

### `stratachrome` (Main CLI)

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | *Required* | Path to the source RGB image file (`.png`, `.jpg`, `.webp`). |
| `-o, --output` | `output/project.3mf` | Destination path for the generated Bambu/Orca 3MF container. |
| `-s, --size` | `150.0` | Target physical size in mm for the largest image dimension (X or Y). |
| `--max-dim` | `1000` | Maximum pixel edge used to rasterize the mesh height grid. |
| `--first-layer` | `0.20` | First layer bed-contact height in millimeters. |
| `--layer-height` | `0.10` | Standard vertical layer step height in millimeters. |
| `--swap-mode` | `ams` | Filament change mode: `ams` for multi-material auto-switching, `manual` for single-extruder pause triggers. |
| `--device` | `auto` | Compute device for transformer inference (`cuda` or `cpu`). |

### `stratachrome-segment`

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | *Required* | Input image path. |
| `-o, --output-dir` | `output` | Directory where segmented RGBA images are saved. |
| `--feather` | `2` | Radius in pixels for Gaussian boundary edge softening. |
| `--threshold` | `None` | Optional binarization cutoff (`0.0` to `1.0`) to force hard edges. |
| `--save-matte` | `False` | Also exports the raw single-channel grayscale alpha matte. |

### `stratachrome-mesh`

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | *Required* | Input image path. |
| `-o, --output` | `output/test_model.stl` | Destination path for the generated binary STL. |
| `-s, --size` | `100.0` | Target physical size in mm for the largest image dimension. |
| `--max-height` | `2.4` | Maximum Z elevation in millimeters. |
| `--base-height` | `0.4` | Solid base floor thickness in millimeters. |
| `--max-dimension` | `400` | Maximum pixel grid size for mesh decimation testing. |

---

## Printing Instructions (Bambu Studio & Orca Slicer)

1. **Import the 3MF**: Open Bambu Studio or Orca Slicer, go to `File > Open Project`, and select your generated `.3mf` file.
2. **Review Slice Settings**:
   * **First Layer Height**: Set to `0.20 mm` (matches `--first-layer`).
   * **Layer Height**: Set to `0.10 mm` (matches `--layer-height`).
   * **Infill**: `100% Rectilinear`.
   * **Walls/Perimeters**: `1` or `2` (Arachne wall generator recommended).
3. **Verify Pauses (Manual Mode Only)**: If you used `--swap-mode manual`, slice the plate and verify the layer pause markers generated from the script's terminal summary. If using `--swap-mode ams`, assign your filaments directly to the AMS slots in the slicer.
4. **Print**: Start the print with the base background filament loaded.

---

## Project Structure

```text
stratachrome/
├── pyproject.toml                 # Package definition, entry points, and dependencies
├── README.md                      # Documentation
├── src/
│   └── stratachrome/
│       ├── __init__.py            # Public API exports
│       ├── color_engine.py        # color_tools CIELCh bridge & filament L* calibration
│       ├── depth_mapper.py        # Two-tier height budget & solid pedestal enforcement
│       ├── export_3mf.py          # OPC/3MF packaging & layer pause metadata generator
│       ├── mesh_builder.py        # 2-manifold triangular mesh & binary STL generator
│       ├── optical_model.py       # Beer-Lambert transmission & monotonic L* sequencing
│       ├── pipeline.py            # End-to-end CLI pipeline orchestrator
│       ├── segment_cli.py         # Standalone segmentation & matte verification CLI
│       ├── segmentation.py        # BiRefNet background extraction & edge feathering
│       └── stl_test_cli.py        # Standalone mesh test & STL export CLI
└── tests/                         # Unit tests and regression suite
```

---

## Development

Run tests and style checks:

```bash
# Run pytest test suite
pytest

# Format codebase
black src/

# Run static type checking
mypy src/
```

---

## License

Stratachrome is licensed under the [MIT License](LICENSE).
