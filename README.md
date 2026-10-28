# Minimum Message Length (MML) Probabilistic Principal Component Analysis

This repository contains a MATLAB implementation of the **MML Probabilistic Principal Component Analysis** from the paper:

**Enes Makalic, Daniel F. Schmidt, _MML Probabilistic Principal Component Analysis_, arXiv:2209.14559 [stat.ME], 2026.**

## Usage

`mml_ppca_fp` selects the number of principal components and estimates the model parameters:

```matlab
res = mml_ppca_fp(X);                   % select J from 0,...,K-1
res = mml_ppca_fp(X, 'NumFactors', 2);  % fit a given J
```

## Reproducing the tables in the paper

Requirements: MATLAB with the Parallel Computing Toolbox and a C compiler for MEX, Python 3, and two toolboxes by Tom Minka:

- the PCA toolbox (`laplace_pca`): https://tminka.github.io/papers/pca/
- lightspeed: https://github.com/tminka/lightspeed

Place them in `external/Laplace_PCA` and `external/lightspeed` in this folder, or add them to the **end** of the MATLAB path.

Then run, from this folder:

```matlab
run_all          % all simulations, 10^5 replications each (about 1 hour on 12 workers)
```

```
python3 scripts/make_tables.py
```

`run_all` writes the simulation results to `results/`, and `make_tables.py` writes the LaTeX tables to `results/tex/`:

| Results file | Tables |
|---|---|
| `final_compare_equal.csv` | Table 1; Supplementary Tables S1–S3 |
| `final_compare_geometric.csv` | Supplementary Tables S4–S6 |
| `final_compare_highdim.csv` | Supplementary Tables S7–S9 |
| `final_modelsel.csv` | Tables 2–3 |
