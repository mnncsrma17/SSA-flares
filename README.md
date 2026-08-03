# SSA-Flare

A Python-based analysis pipeline for applying **Singular Spectrum Analysis (SSA)** to **GOES XRS-B solar flare data**.

## Workflow

**GOES XRS-B FITS → preprocessing → detrending → PSD → window length (L) → Gram matrix → eigendecomposition → SSA/SSD reconstruction**

### Main features

* Extracts GOES XRS-A and XRS-B flux from FITS files
* Selects and preprocesses XRS-B data
* Applies logarithmic transformation and detrending
* Uses Welch PSD to estimate the dominant period
* Automatically determines the SSA window length `L`
* Computes the SSA Gram matrix with memory-efficient chunking
* Performs eigendecomposition
* Reconstructs individual SSA/SSD components
* Produces diagnostic plots of the flux, PSD, reconstruction, components, and eigenvalue spectrum

## Requirements

```bash
pip install numpy scipy matplotlib astropy
```

## Usage

Update the `file_path` in the Python script to point to a GOES XRS FITS file, then run:

```bash
python your_script.py
```

## Data

The analysis is designed for GOES XRS FITS files containing XRS-A and XRS-B flux measurements.

**Note:** Large FITS datasets and generated plots/results should not be committed to the repository.
