# MCR-ALS

This Python implementation was adopted from the [MCR-ALS 2.0 toolbox](https://zenodo.org/records/6334791):

> Joaquim Jaumot, Romà Tauler& Anna de Juan. (2022). MCR-ALS 2.0 toolbox [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.6334791.

MCR-ALS resolves a measured data matrix into concentration and spectral profiles using alternating least squares. This repository contains a NumPy/SciPy implementation of the numerical routines, a Python API, a desktop interface, and automated tests against deterministic MATLAB reference results.

## How the codebase works

The core model is:

```text
data ? concentrations @ spectra
```

The solver alternates between estimating concentrations from the current spectra and estimating spectra from the current concentrations. At each half-iteration it applies the configured constraints, then evaluates residuals, lack of fit, explained variance, and convergence. The best solution encountered is retained.

The main modules are:

- `mcr_als/solver.py` ? ALS iteration, convergence, multi-experiment handling, and result assembly.
- `mcr_als/options.py` ? dataclasses for solver and constraint configuration.
- `mcr_als/helpers.py` ? rank reproduction, nonnegative least squares, unimodality, closure, normalization, and multilinear helpers.
- `mcr_als/correlation.py`, `weighted.py`, and `kinetics.py` ? advanced constraint and fitting routines.
- `mcr_als/gui.py` and `gui_state.py` ? desktop interface and GUI configuration state.
- `tests/` ? unit, end-to-end, GUI-state, and MATLAB parity tests.

All numerical inputs are converted to NumPy `float64` arrays. The implementation follows the original MATLAB iteration order, constraint order, stopping rule, and output formulas where supported. Small differences can still occur between MATLAB and SciPy because their LAPACK/BLAS implementations may differ.

## Installation

Python 3.10 or newer is required. From a clone of this repository:

```bash
python -m venv .venv
```

Activate the environment, then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs NumPy, SciPy, Matplotlib, and OpenPyXL.

## Data format

The main `data` input must be a finite, numeric, two-dimensional matrix:

- rows are observations, samples, or time points;
- columns are measured variables, wavelengths, masses, or other channels.

Do not include row names, column names, or metadata inside the numeric matrix. Remove spreadsheet headers and index columns before fitting.

The initial estimate may be either:

- a concentration estimate with one row per data row and one column per component; or
- a spectral estimate with one row per component and one column per data column.

The solver infers which kind was supplied by matching its dimensions to the data matrix. Transposed forms are accepted when their orientation can be inferred unambiguously.

The desktop interface can load numeric matrices from CSV, TSV, TXT, DAT, NPY, NPZ, MAT, and XLSX files. MAT and NPZ files may contain multiple arrays; select or name the intended array in the interface.

## Python API

```python
from mcr_als import MCRALSOptions, mcr_als

# `data` and `initial_estimate` are two-dimensional NumPy-compatible arrays.
options = MCRALSOptions(max_iterations=100, tolerance=0.1)
result = mcr_als(data, initial_estimate, options)

concentrations = result.concentrations
spectra = result.spectra
reconstructed = result.reconstructed_data
residual_pca = result.residual_pca
residual_experimental = result.residual_experimental
print(result.lack_of_fit, result.r_squared, result.iterations)
```

See `MCRALSOptions` and its nested constraint dataclasses in `mcr_als/options.py` for the complete configuration surface.

## Desktop interface

Launch the GUI after installation:

```bash
mcr-als-gui
```

or:

```bash
python -m mcr_als
```

The interface loads the data and initial estimate, exposes the ALS and constraint settings, runs the calculation, plots the resolved profiles, and saves the numerical outputs.

## Tests

Install pytest and mypy for development:

```bash
python -m pip install pytest mypy
```

Run the test suite:

```bash
python -m pytest -q
```

Run static type checking:

```bash
python -m mypy mcr_als
