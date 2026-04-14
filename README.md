# Chromosome_Analysis_Pipeline

> TADs, compartments, and 3D genome structure modeling  
> **Work in Progress** — Last modified: 02/25/2026

---

## Overview

This project provides pipelines for analyzing chromosome organization through Hi-C modeling and 3D chromosome visualization. It is built on top of [polychrom](https://github.com/open2c/polychrom.git), an open-source polymer simulation toolkit licensed under the MIT License.

---

## Getting Started

### 1. Run a Polychrom Simulation

Generate an `.h5` trajectory file from a polychrom simulation. You can also download the included example file (`blocks_0-99.h5`) to skip this step.

> **GPU users:** It is recommended to run `Loop_Extrusion_3D_simu.py` and `polychrom_simu_get_h5.py` directly from the terminal.
>
> ```bash
> python Loop_Extrusion_3D_simu.py
> python polychrom_simu_get_h5.py
> ```

### 2. Visualize Chromosome Organization

Convert coordinates from `.h5` to `.csv` or `.npy` format, then visualize them using **Blender**.  
See the `visualization/` directory for details and scripts.

> **Note:** Rendering can take a long time — make sure to set `xlim` and `ylim` appropriately before running.

### 3. Run the Analysis Pipeline

For full, publication-ready analyses, see the `work_pipeline/` directory. This includes:

- **Hi-C contact map** generation
- **P(s) curve** (contact probability vs. genomic distance)
- **Gamma (γ) plot** estimation

---

## Project Structure

```
.
├── blocks_0-99.h5                  # Example simulation output
├── Loop_Extrusion_3D_simu.py       # Loop extrusion simulation script (GPU recommended)
├── polychrom_simu_get_h5.py        # Generate .h5 from polychrom
├── visualization/                  # Blender visualization scripts & guides
└── work_pipeline/                  # Hi-C map, P(s), and gamma analysis
```

---

## License

This project uses [polychrom](https://github.com/open2c/polychrom.git), which is licensed under the [MIT License](https://opensource.org/licenses/MIT).
