# Minimum Message Length (MML) Probabilistic Principal Component Analysis

This repository contains a MATLAB implementation of the **MML Probabilistic Principal Component Analysis** from the paper:

**Enes Makalic, Daniel F. Schmidt, _MML Probabilistic Principal Component Analysis_, arXiv:2209.14559 [stat.ME], 2026.**

## Usage

To re-create Table 1 in the paper, use **`sim_param_est.m`**. For example:

```matlab
niter = 1e5;
SNR = 1;
N = 25;
K = 10;
J = 2;
sim_param_est(niter, SNR, N, K, J);
```

To re-create Tables 2 and 3 from the paper, use **`sim_model_sel.m`**. Note, to run the model selection code below you must have [Thomas Minka's PCA toolbox](https://tminka.github.io/papers/pca/) installed
and in the MATLAB path. For example:

```matlab
niter = 1e5;
SNR = 1;
N = 100;
K = 10;
Jtrue = 1;
Jmax = 5;
sim_model_sel(niter, SNR, N, K, Jtrue, Jmax);
```
