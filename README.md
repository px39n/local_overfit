# Local Overfit - Figure Generation Code

[![DOI](https://zenodo.org/badge/1115351546.svg)](https://doi.org/10.5281/zenodo.17915486)

This repository contains the code for generating all figures in the paper:

**"Countering Local Overfitting for Equitable Spatiotemporal Modeling"**

*Zhehao Liang, Stefano Castruccio, Paola Crippa*

## Repository Structure

```
local_overfit/
├── data/                          # Data files for figure generation
│   ├── processed/                 # Preprocessed data ready for plotting
│   └── raw/                       # Raw experimental results
├── source_code/                   # Helper functions and utilities
│   ├── __init__.py
│   ├── plotting.py                # Common plotting utilities
│   ├── data_loader.py             # Data loading functions
│   └── styles.py                  # Figure styles and color schemes
├── main_figures.ipynb             # Main text figures (Fig 1-5)
├── supplementary_figures.ipynb    # Supplementary/Appendix figures
└── README.md                      # This file
```

## Figures Overview

### Main Figures (`main_figures.ipynb`)

| Figure | Description |
|--------|-------------|
| Fig 1  | (a) Spatial distribution of TOAR stations, (b) Density distribution, (c) Validation results |
| Fig 2  | Local overfitting: (a) Model complexity vs R², (b) Training epochs vs R², (c) Regional predictions |
| Fig 3  | OG ensemble approach: HV × LV model combinations |
| Fig 4  | OG mitigation of local overfitting |
| Fig 5  | Spatial equity assessment: Accuracy surfaces and quartile analysis |

### Supplementary Figures (`supplementary_figures.ipynb`)

| Figure | Description |
|--------|-------------|
| Fig S1 | Full model family comparison |
| Fig S2 | Additional validation strategies |
| Fig S3 | Ablation studies |
| ...    | ... |

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install cartopy geopandas  # For geographic plots
```

## Usage

1. Ensure all data files are in the `data/` directory
2. Open the desired notebook in Jupyter
3. Run all cells to generate figures

```bash
jupyter notebook main_figures.ipynb
```

## Data Description

### Core Data Files

| File | Description | Used By |
|------|-------------|---------|
| `training/Site_Observation_with_feature.pkl` | Station observations with 22 features | Fig 1a, 1b |
| `ozone.nc` | Gridded ozone data (10km resolution) | Fig 1b, 5b, 5d |
| `ne_10m_land.shp` | Natural Earth land boundaries | All map plots |

### Experimental Results

| Directory | Description | Used By |
|-----------|-------------|---------|
| `appendix/represent_10CV_100k/` | 10-fold CV results (100k samples) | Fig 1c |
| `appendix/5CV_validation_vs_parameter/` | Parameter complexity analysis | Fig 2a, 4a |
| `appendix/5-Fold-Epoch/` | Training epoch analysis | Fig 2b |
| `appendix/5-Fold-nn_grid/epoch_analysis/` | OG epoch analysis | Fig 4b |
| `appendix/10-Fold/` | OG ensemble results (all HV×LV combos) | Fig 3 |
| `appendix/geo_prediction/` | Spatial prediction NetCDF files | Fig 2c, 4c |

### Analysis Data

| File | Description | Used By |
|------|-------------|---------|
| `Analysis/validation/df_analysis.pkl` | Accuracy analysis across conditions | Fig 5a-d |
| `Analysis/validation/df_analysis_Station.pkl` | Station-level accuracy | Fig 5 |
| `Analysis/validation/df_analysis_Grid.pkl` | Grid-level accuracy | Fig 5 |
| `Analysis/Figure1_Region_based/*.csv` | Region-based CV results | Fig 5e |

## Citation

If you use this code, please cite:

```bibtex
@article{liang2025og,
  title={Countering Local Overfitting for Equitable Spatiotemporal Modeling},
  author={Liang, Zhehao and Castruccio, Stefano and Crippa, Paola},
  journal={},
  year={2025}
}
```

## Related Packages

This figure generation code accompanies two Python packages developed as part of this research:

### og-learn

**Overfit-to-Generalization Framework for Equitable Spatiotemporal Modeling**

The `og-learn` package implements the OG framework described in the paper, enabling:
- Two-stage training with HV/LV model combinations
- Density-aware sampling strategies
- Pseudo-label generation with spatial perturbation

| | |
|---|---|
| **Repository** | [github.com/px39n/og-learn](https://github.com/px39n/og-learn) |
| **Installation** | `pip install og-learn` |
| **Documentation** | [px39n.github.io/og-learn](https://px39n.github.io/og-learn) |

```python
from og_learn import OGFramework
from og_learn.models import LightGBMOverfitter, TransformerRegressor

og = OGFramework(
    hv_model=LightGBMOverfitter(),
    lv_model=TransformerRegressor(),
)
og.fit(X_train, y_train, coords=coordinates)
predictions = og.predict(X_test)
```

---

### geoequity

**Spatial Equity Assessment and Visualization for Machine Learning Models**

The `geoequity` package provides tools for diagnosing spatial equity in ML models:
- Accuracy surface modeling (GAM + SVM)
- Density-aware performance assessment
- Spatial equity visualization

| | |
|---|---|
| **Repository** | [github.com/px39n/geoequity](https://github.com/px39n/geoequity) |
| **Installation** | `pip install geoequity` |
| **Documentation** | [px39n.github.io/geoequity](https://px39n.github.io/geoequity) |

```python
from geoequity import AccuracySurface
from geoequity.visualization import plot_accuracy_matrix, plot_equity_map

surface = AccuracySurface()
surface.fit(r2_scores, densities, sample_sizes, coordinates)
fig = plot_accuracy_matrix(surface)
```

---

## License

MIT License - see the main repository for details.

