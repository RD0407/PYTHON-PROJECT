# FTIR Mixture Deconvolution

A simple Python project for quantifying the concentrations of three pure components (A, B, C) in a mixture using FTIR spectral data.

## What does this project do?

- Loads FTIR spectra of three pure reference compounds and their mixture from `.npy` files.
- Uses linear least squares to fit the mixture spectrum as a combination of pure component spectra.
- Determines and prints the concentration of each pure component in the mixture.
- Visualizes the measured mixture spectrum, the best-fit reconstructed mixture, and the pure component spectra for comparison.

## Files

- `components.npy` — FTIR spectra of pure components (A, B, C), one row per component.
- `mixture_spectrum.npy` — FTIR spectrum of a mixture of these components.

## How to Use

1. Place your `.npy` files in the project directory.
2. Run the provided Python script (see `FTIR-Concentration-Fitting.py`).
3. The script prints out the concentrations and displays a comparison plot.
