# Stratachrome

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Stratachrome** is an automated multi-color 3D relief printing pipeline designed for standard single-extruder FDM 3D printers and multi-material systems (Bambu Lab AMS, Orca Slicer).

Stratachrome can process the full image as one relief or use **BiRefNet background segmentation** to create stacked background and foreground tiers. It selects real Bambu Lab PLA Basic/Matte filaments, derives useful transition thickness from TD-driven optical modeling, maps image colors to simulated printable layer states, and packages a closed terraced mesh inside a ready-to-slice Bambu/Orca 3MF container.

---

## Key Features

* **Selectable Relief Modes**: Use the full image as one lightweight tier or isolate foreground subjects with BiRefNet and stack optically mapped background and foreground reliefs.
* **Adaptive Tier Palettes**: Resizes each tier to a 64-pixel maximum edge with nearest-neighbor sampling, quantizes it to twice the requested filament cap, and selects up to the requested 2–8 colors from `FilamentCollections.BAMBU_PLA_BASICMATTE`. Filaments may be reused across tiers.
* **Predictive Optical Modeling**: Uses each selected filament's measured color and TD value to simulate successive layers in linear CIE XYZ.
* **Perceptual Color Precision**: Uses `color-match-tools` for image/filament CIELAB conversion, filament records and collections, and vectorized CIEDE2000 comparisons.
* **Information-Driven Thickness**: Optimizes layer schedules by balancing reconstruction error against the diminishing perceptual value of each additional layer. Selected non-black filaments are removed only when their TD-derived minimums cannot fit the tier budget.
* **Adjustable TD Calibration**: Scales catalog transmission distances during optical simulation so predicted blending can be calibrated against physical filament and printer behavior.
* **Optically Mapped Relief**: Maps two-tier image colors to the nearest simulated CIELAB layer states using CIEDE2000. Single-tier mode maps the image's $L^*$ range onto its available layer heights.
* **Calibrated Layer Alignment**: Built with a dedicated 0.20 mm first-layer base for reliable bed adhesion and 0.10 mm layer increments matching standard slicer toolpaths.
* **Flexible Swap Modes**: Supports automated multi-material hardware changers (`--swap-mode ams`) or single-extruder pause triggers (`--swap-mode manual`).
* **Auto-Scaling Aspect Ratios**: Specify target maximum dimension in millimeters (`--size`); landscape and portrait images automatically scale to fit within your build plate envelope.
* **Slicer-Optimized Grid Resolution**: Built around standard 0.42 mm nozzle line widths and Arachne dynamic extrusion parameters, sampling up to 1000 px resolution for Nyquist fidelity without slicing lag or mesh bloat.
* **Flat Pixel Terraces**: Gives every resampled image pixel a horizontal plateau at its selected layer and joins neighboring elevations across narrow transition gaps.
* **Conservative Surface Reduction**: Reduces coplanar regions in the XY surface while preserving the known-good watertight perimeter skirt and bottom closure.
* **Native Bambu / Orca 3MF Packaging**: Exports Open Packaging Conventions (OPC) archives with layer changes and finish-aware Bambu PLA Basic/Matte X1C profiles.
* **Generation Diagnostics**: Reports selected filaments, layer allocations, convergence status, CIEDE2000 error, objective score, used height levels, and final mesh size.
* **Color Diagnostics**: Optionally saves the predicted optical color at every pixel and a fixed-scale CIEDE2000 error heatmap for comparing schedules.

---

## Pipeline Architecture

```text
Input Image
    │
    ├──► [Segmentation] (BiRefNet) ─────────────► Alpha Matte
    │                                                   │
    └──► [Color Engine] (color_tools CIELAB) ──► Full Lab Image
                                                        │
         ┌──────────────────────────────────────────────┘
         ▼
[64px Tier Quantization] ──► Representative Background & Foreground Colors
         │
         ▼
[Filament Search] ─────────► Bambu PLA Basic/Matte Candidates
         │
         ▼
[Schedule Optimization] ───► Useful TD/XYZ Transition Schedule
         │
         ▼
[Depth Mapper] ────────────► Single L* Relief or Full-Lab Two-Tier Pedestal
         │
         ▼
[XY Surface Reduction] ────► Watertight 2-Manifold Triangle Mesh
         │
         ▼
[OPC 3MF Exporter] ────────► Bambu Studio / Orca Slicer .3mf Container (AMS / Manual)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dterracino/stratachrome.git
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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

Skip foreground extraction and process the full image as one tier:

```bash
stratachrome -i assets/subject.png -o output/single.3mf --tier-mode single
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
| `--tier-mode` | `two` | `single` processes the full image without loading BiRefNet; `two` maps segmented background and foreground colors to stacked simulated optical states. |
| `--device` | `auto` | Compute device for transformer inference (`cuda` or `cpu`). |
| `--colors-per-tier` | `4` | Maximum filament count from 2 to 8 per tier. Each 64px nearest-neighbor tier image is quantized to twice this ceiling, providing extra matching candidates; duplicate matches can still collapse to fewer filaments and foreground selection prefers reusable background colors. |
| `--max-layers-per-tier` | `120` | Maximum total layers available to each tier. TD and perceptual fit determine how many layers are actually used. |
| `--td-scale` | `1.0` | Positive multiplier for catalog TD values. Values above `1.0` model greater transparency and generally require more layers; values below `1.0` model greater opacity and generally require fewer layers. |
| `--color-diagnostics` | `False` | Save `<output-stem>_color_preview.png` and `<output-stem>_delta_e_heatmap.png` beside the 3MF. |

The predicted-color preview shows the simulated optical state assigned to each pixel. The heatmap compares that state with the source using CIEDE2000: black indicates zero error, progressing through blue, cyan, and yellow to red at Delta E 30 or greater. The fixed scale allows direct comparison between different palettes and schedules.

### Understanding `--td-scale`

Transmission distance (TD) describes how much material is required to obscure the color beneath it. Stratachrome applies the scale before simulating each layer:

$$
\mathrm{TD}_{\mathrm{effective}} = \mathrm{TD}_{\mathrm{catalog}} \times \mathrm{td\_scale}
$$

The remaining contribution from the material below a layer of thickness $d$ is:

$$
T = 10^{-d / \mathrm{TD}_{\mathrm{effective}}}
$$

For a filament with a catalog TD of $4\,\mathrm{mm}$:

| `--td-scale` | Effective TD | Modeled behavior |
| --- | --- | --- |
| `0.5` | $2\,\mathrm{mm}$ | More opaque; the new color covers the substrate faster. |
| `1.0` | $4\,\mathrm{mm}$ | Uses the catalog value unchanged. |
| `2.0` | $8\,\mathrm{mm}$ | More transparent; the substrate remains visible through more layers. |

The setting does not rescale the source image or directly multiply model height. It changes the optical predictions used to choose filament thicknesses, so the optimized layer counts and resulting tier height can change indirectly. Larger values can also make a small `--max-layers-per-tier` budget insufficient.

Start with `1.0`. If printed upper colors hide lower colors faster than predicted, try a smaller value. If lower colors remain visible longer than predicted, try a larger value. Calibrate with the same filament, layer height, nozzle, and print settings intended for production.

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
│       ├── bambu_exporter.py       # High-level native Bambu/Orca project writer
│       ├── bambu_project.py        # 3MF XML, printer settings, and filament profiles
│       ├── color_engine.py        # 64px tier quantization, filament matching, and joint planning
│       ├── depth_mapper.py        # Two-tier height budget & solid pedestal enforcement
│       ├── export_3mf.py           # Generic OPC/3MF packaging helpers
│       ├── mesh_builder.py         # Three-axis reduced manifold mesh & binary STL generator
│       ├── optical_model.py       # TD/XYZ simulation, CIEDE2000 mapping, schedule optimization
│       ├── pipeline.py            # End-to-end CLI pipeline orchestrator
│       ├── resources/              # Packaged Bambu X1C project settings
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
