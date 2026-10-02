# SSA of GOES XRS-B Solar X-ray Flux

A single-script pipeline (`ssa_updated.py`) that applies **Singular Spectrum Analysis (SSA)** to one day of GOES-13/14/15 XRS-B data to isolate quasi-periodic pulsations (QPPs) in solar flares, then characterises the residual noise and tests whether each SSA component behaves as a Markov process.

## Data

- GOES-13, -14 or -15 FITS files (≈2 s cadence), XRS-B channel (1–8 Å), read from HDU 2.
- Input is set by one variable at the top of the script:

```python
file_path = "/path/to/go1520131028.fits"
OBSERVATION_LABEL = "GOES-15 | 28 October, 2013"   # stamped on every figure
```

## Pipeline

| Step | What it does |
|---|---|
| 1. Load & clean | Read time/flux, split XRS-A/XRS-B, drop non-finite and non-positive samples |
| 2. Preprocess | `log10` transform → linear detrend → coarse-graining (scale 5, ≈10.24 s cadence) |
| 3. Welch PSD | 4th-order zero-phase Butterworth high-pass (300 min cutoff) → Welch PSD → dominant period (5–300 min band, `find_peaks` on log-PSD) |
| 4. SSA | Window length `L = 1.2 × dominant period` (capped at 0.45 N, forced even); Gram matrix `XᵀX` built in chunks from the lagged trajectory matrix; eigendecomposition with `eigh` |
| 5. Grouping | Component 1 = trend; components 2…L grouped by `floor(log10(λᵢ/λ₁))`; scales below `NOISE_FLOOR_EXPONENT` (default −4) form the noise/residual group |
| 6. Reconstruction | Diagonal averaging per group; reconstruction is checked against the input (max error ~1e-15) |
| 7. Noise diagnostics | Mean, std, skewness, excess kurtosis; robust-bandwidth KDE vs Gaussian; Welch spectral slope β (PSD ∝ f^−β); stability index α via ECF regression with a Fama–Roll quantile cross-check |
| 8. State dependence | Pearson/Spearman correlation (linear and log–log) of every non-noise group against the noise group |
| 9. Markov time scale | ACF and increment-ACF decorrelation time (|ACF| ≤ 0.10 for 5 consecutive lags) → candidate τ per component |
| 10. Chapman–Kolmogorov | 30-bin quantile transition matrices; compares the direct 2τ propagator with the τ·τ prediction (KS and L1 distances at the 10th/50th/90th percentile states) |
| 11. Component correlations | Pearson and W-correlation matrices of the reconstructed components |

**Note:** if no group falls below the noise floor, the last (lowest-energy) group is used as a proxy noise component and a `[WARN]` is printed.

## Outputs

Written under `/content/SSA_components/`:

```
SSA1.csv … SSAn.csv                      # reconstructed components (time, amplitude)
SSA_noise_diagnostics(.csv, _summary.csv)
state_dependence/                        # scatter figure + correlation CSV
markov_analysis/                         # ACF figure + candidate_markov_timescales.csv
  └─ CK_test/                            # per-component CK figures + CK_consistency_results.csv
correlation_analysis/                    # Pearson & W-correlation matrices (CSV + PNG)
```

## Requirements

Python 3, `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`

```bash
pip install numpy scipy pandas matplotlib astropy
```

## Usage

1. Set `file_path`, `OBSERVATION_LABEL`, and (optionally) `COARSE_SCALE`, `MAX_PERIOD`/`MIN_PERIOD`, `NOISE_FLOOR_EXPONENT`.
2. Update the hard-coded output paths (`/content/...`) if you are not running in Google Colab.
3. Run the script top to bottom; all sections depend on the preceding ones (the state-dependence step reads the CSVs saved by the reconstruction step).

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `COARSE_SCALE` | 5 | Block-averaging factor |
| `MAX_PERIOD` / `MIN_PERIOD` | 300 / 5 min | QPP search band and high-pass cutoff |
| `NOISE_FLOOR_EXPONENT` | −4 | Eigenvalue scale below which components are labelled noise |
| `MARKOV_ACF_THRESHOLD` / `_CONSECUTIVE` | 0.10 / 5 | Decorrelation criterion |
| `CK_N_BINS` | 30 | States in the transition matrix |
