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
res = mml_ppca_fp(X, 'Verbose', true);                   % select J from 0,...,K-1
res = mml_ppca_fp(X, 'NumFactors', 2, 'Verbose', true);  % fit a given J
res.J, res.tau                                           % selected rank and residual variance
```

## Reproducing the tables in the paper

Requirements: MATLAB with the Parallel Computing Toolbox, Python 3, and two toolboxes by Tom Minka:

- the PCA toolbox (`laplace_pca`): https://tminka.github.io/papers/pca/
- lightspeed (for `row_sum`, used by `laplace_pca`): https://github.com/tminka/lightspeed

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

## Checking the bias formulas (Theorem 1)

`checks/bias_full.m` compares the empirical bias of the ML and MML87 estimates of the residual variance with the first-order formulas of Theorem 1, over K = 5, 10, 20, 50, J = 1, 2, 4, three signal strengths and N = 50 to 5000 (10^5 replications per setting, about 7 minutes on 12 workers). `checks/bias_check.m` is a smaller version with K = 10. Run, from this folder:

```matlab
addpath('checks');
T = bias_full;   % table of N*bias: formula vs empirical, with standard errors
```

## Checking Corollary 1 and the entropy-loss remark (Section 5)

`checks/cor_sim.m` computes the Stein and entropy losses of the ML and MML87 covariance estimates in the proportional regime K = gamma*N, for gamma = 0.25, 0.5, 1, one or two spikes and N = 100 to 1600. It compares the mean of L(ML) - L(MML87) with its limit from Corollary 1 (Stein loss, always positive) and from the remark after it (entropy loss, either sign), and reports how often MML87 has the smaller loss. With 5000 replications per setting it takes about 23 minutes on 14 workers. Run, from this folder:

```matlab
addpath('checks');
T = cor_sim(5000, [100 200 400 800 1600]);   % simulated vs limiting loss differences
```

The default settings have one or two spikes. To run three to five spikes, pass the settings as rows of {gamma, rho}, with an optional base seed (about 21 minutes for the following):

```matlab
cfg = { 0.25, [10 5 2];         0.5, [10 5 3];         1, [20 10 5]; ...
        0.25, [20 10 5 2];      0.5, [20 10 5 2.5];    1, [40 20 10 5]; ...
        0.25, [20 10 5 2 1.5];  0.5, [20 10 5 3 2.5];  1, [40 20 10 6 4]; ...
        0.25, [2.5 2 1.5];      0.25, [3 2.5 2 1.75 1.5] };
T = cor_sim(5000, [100 200 400 800 1600], cfg, 2000);
```
