# Minimum Message Length (MML) Probabilistic Principal Component Analysis

This repository contains a MATLAB implementation of the **MML Probabilistic Principal Component Analysis** from the paper:

**Enes Makalic, Daniel F. Schmidt, _MML Probabilistic Principal Component Analysis_, arXiv:2209.14559 [stat.ME], 2026.**

## Usage

`generate_pca_data` simulates data from a probabilistic PCA model. For example, N = 100 observations in K = 10 dimensions with J = 2 components, residual variance 1 and per-component signal-to-noise ratio 5:

```matlab
[~, ~, truth, X] = generate_pca_data(100, 10, 'J', 2, 'sigma2', 1, ...
                                     'SNR', 1, 'LoadingDesign', 'equal');
```

Here `SNR` is the total signal-to-noise ratio, sum(rho_j)/K, so `SNR = 1` gives rho_1 = rho_2 = 5. `truth` holds the generating parameters.

`mml_ppca_fp` selects the number of principal components and estimates the model parameters:

```matlab
res = mml_ppca_fp(X);                   % select J from 0,...,K-1
res = mml_ppca_fp(X, 'NumFactors', 2);  % fit a given J
res.J, res.tau                          % selected rank and residual variance
```

## Reproducing the tables in the paper

Requirements: MATLAB with the Parallel Computing Toolbox and a C compiler for MEX, Python 3, and two toolboxes by Tom Minka:

- the PCA toolbox (`laplace_pca`): https://tminka.github.io/papers/pca/
- lightspeed: https://github.com/tminka/lightspeed

Place them in `external/Laplace_PCA` and `external/lightspeed` in this folder, or add them to the **end** of the MATLAB path.

Then run, from this folder:

```matlab
run_all          % Tables 1-4 and S2-S10: 10^5 replications each (about 1 hour on 12 workers)
sim_asymptotic   % Table S1: 2x10^4 replications at N = 100, 1000, 10000 (about 6 hours on 14 workers)
```

`sim_asymptotic` needs only MATLAB and the Parallel Computing Toolbox.

```
python3 scripts/make_tables.py
```

`run_all` and `sim_asymptotic` write the simulation results to `results/`, and `make_tables.py` writes the LaTeX tables to `results/tex/`:

| Results file | Tables |
|---|---|
| `final_compare_equal.csv` | Tables 1–2; Supplementary Tables S2–S4 |
| `final_compare_geometric.csv` | Table 2; Supplementary Tables S5–S7 |
| `final_compare_highdim.csv` | Table 2; Supplementary Tables S8–S10 |
| `final_modelsel.csv` | Tables 3–4 |
| `final_asymptotic.csv` | Supplementary Table S1 |
