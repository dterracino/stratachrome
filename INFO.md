# StrataChrome

**StrataChrome** is a high-precision computational engine for optical filament layering in multi-color fused deposition modeling (FDM) 3D printing. By depositing semi-translucent colored filaments at sub-millimeter discrete layer increments (e.g., $0.04\,\text{mm} - 0.08\,\text{mm}$ steps above a solid base), StrataChrome models the subtractive and scattering optical interactions of light traveling through stratified polymer layers to render continuous-tone, high-fidelity color images in solid plastic.

---

## 1. Physical Science & Optical Theory

The visual appearance of layered translucent polymer melts is governed by transmission, absorption, and subsurface scattering. StrataChrome combines empirical transmission distance metrics with radiative transfer physics to accurately predict the color of layered strata before slicing.

### 1.1 Transmission Distance (TD) & Opacity

Transmission Distance ($\text{TD}$) is the benchmark metric defining the optical attenuation of 3D printing filaments:

* **Definition:** The thickness $z$ (in $\text{mm}$) of a solid filament layer required to attenuate transmitted light to an opacity threshold of $90\%$ ($10\%$ transmittance, or $T = 0.10$).
* **High-$\text{TD}$ Filaments ($\text{TD} \ge 4.0\,\text{mm}$):** Translucent and low-pigment filaments that permit incident light to penetrate deeply into underlying strata, creating smooth blends, gradients, and intermediate tonal transitions.
* **Low-$\text{TD}$ Filaments ($\text{TD} \le 1.0\,\text{mm}$):** Densely pigmented filaments (such as carbon black or titanium dioxide white) that serve as optical backdrops, light-blocking barriers, or high-reflectance substrates.

### 1.2 Forward Optical Propagation Models

StrataChrome evaluates optical color mixing using two complementary propagation models depending on pigment density and scattering characteristics.

#### 1. Beer-Lambert Attenuation (Simplified Transmittance)

For polymer media where forward transmission and absorption dominate over particulate backscattering:

$$
I(z) = I_0 \exp(-\alpha(\lambda) \cdot z)
$$

Where:

* $I_0$ is the incident light intensity.
* $z$ is the physical layer thickness in millimeters.
* $\alpha(\lambda)$ is the wavelength-dependent spectral attenuation coefficient.

Calibrating against the filament's measured $\text{TD}$ (where $I(\text{TD}) / I_0 = T_{\text{threshold}} = 0.10$):

$$
\alpha = -\frac{\ln(0.10)}{\text{TD}} \approx \frac{2.302585}{\text{TD}}
$$

#### 2. Kubelka-Munk 2-Flux Radiative Transfer Model

For pigmented polymers where multi-directional scattering from additive particles ($\text{TiO}_2$, colorant agglomerates) contributes significantly to surface reflectance, StrataChrome employs the two-flux Kubelka-Munk differential equations:

$$
\frac{dI}{dz} = -(K + S)I + SJ
$$

$$
-\frac{dJ}{dz} = -(K + S)J + SI
$$

Where:

* $I$ is the downward-propagating (incident) light flux.
* $J$ is the upward-propagating (reflected and backscattered) light flux.
* $K(\lambda)$ is the spectral absorption coefficient.
* $S(\lambda)$ is the spectral scattering coefficient.

For a filament layer of thickness $x$ deposited over an underlying substrate of reflectance $R_g$, the composite reflectance $R$ is expressed analytically as:

$$
R = \frac{1 - R_g(a - b \coth(b S x))}{a - R_g + b \coth(b S x)}
$$

With auxiliary parameters:

$$
a = 1 + \frac{K}{S}
$$

$$
b = \sqrt{a^2 - 1}
$$

For an optically infinitely thick layer ($x \to \infty$), the terminal reflectance $R_\infty$ simplifies to:

$$
R_\infty = 1 + \frac{K}{S} - \sqrt{\left(\frac{K}{S}\right)^2 + 2\left(\frac{K}{S}\right)}
$$

---

## 2. Mathematical Foundations & Colorimetry

Color accuracy requires mapping between non-linear display standards and perceptually uniform color spaces.

### 2.1 Color Space Transformations

#### 1. sRGB to Linear RGB

Removing the non-linear gamma companding function from standard 8-bit sRGB channels ($C_{\text{sRGB}} \in [0, 1]$):

$$
C_{\text{linear}} = \begin{cases} \frac{C_{\text{sRGB}}}{12.92}, & C_{\text{sRGB}} \le 0.04045 \\ \left(\frac{C_{\text{sRGB}} + 0.055}{1.055}\right)^{2.4}, & C_{\text{sRGB}} > 0.04045 \end{cases}
$$

#### 2. Linear RGB to CIE $XYZ$

Mapping to the device-independent CIE $1931$ standard observer system under the standard $\text{D}65$ illuminant:

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} 0.4124564 & 0.3575761 & 0.1804375 \\ 0.2126729 & 0.7151522 & 0.0721750 \\ 0.0193339 & 0.1191920 & 0.9503041 \end{bmatrix} \begin{bmatrix} R_{\text{linear}} \\ G_{\text{linear}} \\ B_{\text{linear}} \end{bmatrix}
$$

#### 3. CIE $XYZ$ to CIE $L^*a^*b^*$

Converting CIE $XYZ$ coordinates into the perceptually uniform CIE $L^*a^*b^*$ space normalized to the $\text{D}65$ reference white point $(X_n, Y_n, Z_n) = (95.047, 100.000, 108.883)$:

$$
L^* = 116 f\left(\frac{Y}{Y_n}\right) - 16
$$

$$
a^* = 500 \left[ f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right) \right]
$$

$$
b^* = 200 \left[ f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right) \right]
$$

Where the transfer function $f(t)$ is piecewise defined to maintain numerical stability near zero:

$$
f(t) = \begin{cases} t^{1/3}, & t > \left(\frac{6}{29}\right)^3 \approx 0.008856 \\ \frac{1}{3}\left(\frac{29}{6}\right)^2 t + \frac{4}{29}, & t \le \left(\frac{6}{29}\right)^3 \end{cases}
$$

#### 4. CIE $L^*a^*b^*$ to CIE $LCH$ (On-Demand Polar Representation)

When performing gamut slicing, chroma thresholding, or hue binning, coordinates are converted to polar chroma $C^*$ and hue angle $H^\circ$:

$$
C^* = \sqrt{(a^*)^2 + (b^*)^2}
$$

$$
H^\circ = \left( \operatorname{atan2}(b^*, a^*) \cdot \frac{180^\circ}{\pi} \right) \pmod{360^\circ}
$$

---

### 2.2 Perceptual Distance Metrics

To minimize human-perceptible color divergence between target pixel matrices and simulated polymer strata:

#### 1. Euclidean Metric (CIE76)

Used as a high-speed candidate filter to rapidly eliminate distant layer stacks:

$$
\Delta E_{ab}^* = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}
$$

#### 2. Comprehensive Perceptual Metric (CIEDE2000)

Used as the final objective loss function, accounting for human vision non-uniformities across luminance, chroma, and hue:

$$
\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C'}{k_C S_C}\right)\left(\frac{\Delta H'}{k_H S_H}\right)}
$$

Where:

* $S_L, S_C, S_H$ are compensation weighting functions for lightness, chroma, and hue.
* $R_T$ is the rotation function accounting for the non-elliptical interaction in the blue region ($H' \approx 275^\circ$).
* $k_L, k_C, k_H$ are parametric correction coefficients (set to $1.0$ under standard reference conditions).

#### 3. HyAB Hybrid Distance

Employed during image edge detection and regional segmentation where spatial luminance contrast must be weighted against chromatic boundaries:

$$
\Delta E_{\text{HyAB}} = |L_1^* - L_2^*| + \sqrt{(a_1^* - a_2^*)^2 + (b_1^* - b_2^*)^2}
$$

---

### 2.3 Discrete Layer Optimization Problem

Given an ordered stack of $N$ printable filaments $(F_1, F_2, \dots, F_N)$ deposited sequentially from bottom to top:

* **Initial base layer height:** $h_0$ (e.g., $0.16\,\text{mm}$ or $0.20\,\text{mm}$).
* **Slicing step increment:** $\Delta z$ (e.g., $0.04\,\text{mm}$ or $0.08\,\text{mm}$).
* **Integer layer count vector for pixel $(x, y)$:** $\mathbf{n}(x, y) = (n_1, n_2, \dots, n_N) \in \mathbb{N}_0^N$.

The physical thickness of filament $k$ at pixel $(x, y)$ is $z_k(x, y) = n_k(x, y) \cdot \Delta z$, yielding a cumulative surface height:

$$
Z(x, y) = h_0 + \sum_{k=1}^N n_k(x, y) \cdot \Delta z
$$

For each pixel coordinate $(x, y)$ with target color $\mathbf{C}_{\text{target}}(x, y) \in \text{Lab}$, the optimizer solves:

$$
\min_{\{n_1, \dots, n_N\}} \Delta E_{00}\left( \mathbf{C}_{\text{target}}(x, y), \operatorname{PredictColor}\left(F_1, n_1, \dots, F_N, n_N\right) \right)
$$

Subject to the physical FDM constraints:

1. **Monotonic Deposition:** Filaments are extruded in fixed sequence; filament $F_{k+1}$ cannot be placed beneath $F_k$.
2. **Maximum Permissible Height:** $Z(x, y) \le Z_{\max}$.
3. **Discrete Slicing Alignment:** $n_k \in \mathbb{N}_0$.

---

## 3. Architecture & Module Overview

```text
src/stratachrome/
├── __init__.py          # Package initialization and exports
├── bambu_exporter.py    # Native Bambu/Orca 3MF archive writer
├── bambu_project.py     # Bambu project XML, settings, and filament metadata
├── optical_model.py     # Beer-Lambert & Kubelka-Munk forward transmission prediction
├── color_engine.py      # Inverse optimization engine and LUT generation
├── depth_mapper.py      # 2D heightfield synthesis, quantization, and batch Lab conversion
├── segmentation.py      # Perceptual region clustering and hue-boundary isolation
├── mesh_builder.py      # Watertight manifold STL mesh generation with border walls
├── pipeline.py          # End-to-end orchestration CLI and execution controller
├── segment_cli.py       # Standalone CLI for image segmentation analysis
└── stl_test_cli.py      # Mesh inspection and verification CLI
```

### Module Responsibilities

* **`optical_model.py`:** Evaluates forward light attenuation and spectral reflectance through stacked polymer layers using transmission distance ($\text{TD}$) and absorption/scattering coefficients. Directly consumes `FilamentRecord` instances.
* **`color_engine.py`:** Solves the combinatorial inverse problem using dynamic programming and multi-phase pruning (CIE76 coarse pass followed by CIEDE2000 fine minimization). Uses `FilamentCollections.BAMBU_PLA_BASICMATTE` as default palette.
* **`depth_mapper.py`:** Converts the input image into discrete $Z$-height matrices, applying spatial smoothing filters and slicing step quantizations.
* **`segmentation.py`:** Partitions multi-subject scenes into distinct regions to permit localized filament sequences or isolated structural prints.
* **`mesh_builder.py`:** Transforms $Z$-height arrays into watertight 3D triangle meshes featuring solid backplates, stepped planar strata, and side borders.
* **`bambu_exporter.py`:** Packages triangulated meshes and swap schedules into native Bambu Studio and OrcaSlicer 3MF archives.
* **`bambu_project.py`:** Generates project XML, printer settings, filament profiles, and automatic or manual layer-change metadata.
* **`pipeline.py`:** Orchestrates the workflow from source image to sliced 3MF artifacts.

---

## 4. Integration with `color_tools`

StrataChrome leverages `color_tools` for verified colorimetry and physical filament data:

* **Direct Data Records:** Native `FilamentRecord` objects are passed directly into optical functions without intermediary wrappers, reading `name`, `rgb`, `lab`, and `td_value` directly.
* **On-Demand Derived Spaces:** Polar coordinates (LCH) are calculated on-the-fly via `color_tools.conversions.lab_to_lch` only when evaluating chromatic boundaries.
* **Standard Collections:** Defaults to `FilamentCollections.BAMBU_PLA_BASICMATTE`, giving direct access to measured transmission distances and color coordinates.
* **Loss Functions:** Uses `color_tools.distance.delta_e_ciede2000` for primary optimization and `color_tools.distance.delta_e_cie76` for early candidate pruning.

---

## 5. Quickstart & Usage

### Slicing an Image into a Multi-Color 3MF

```bash
python -m stratachrome.pipeline \
  --input-image assets/sample.png \
  --output-3mf output/sample_model.3mf \
  --base-height 0.20 \
  --step-height 0.04 \
  --max-height 2.40 \
  --collection BAMBU_PLA_BASICMATTE
```

### Running Regional Segmentation

```bash
python -m stratachrome.segment_cli \
  --input assets/character.png \
  --output-dir output/segments/ \
  --tolerance 5.0
```
