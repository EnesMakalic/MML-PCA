%PCA_GIC    Generalized Information Criterion for PCA model selection
%   RGIC = PCA_GIC(LAM, S2, R, N) computes the Generalized Information
%   Criterion (GIC) for a probabilistic PCA model with R latent factors
%   given the eigenvalues LAM of the sample covariance matrix, the residual
%   variance estimate S2, and sample size N.
%
%   [RGIC, NEGLL, DOF] = PCA_GIC(LAM, S2, R, N) also returns the negative
%   log-likelihood NEGLL and the effective degrees of freedom DOF.
%
%   Description:
%   The Generalized Information Criterion (GIC) is an improved variant of
%   Akaike's Information Criterion (AIC) specifically designed for PCA
%   model selection. GIC provides better finite-sample performance than
%   standard AIC by using a more accurate estimate of the effective degrees
%   of freedom that accounts for the bias in eigenvalue estimation.
%
%   The GIC is defined as:
%     GIC(r) = -2*log-likelihood + (2/n)*DOF
%
%   where DOF is the effective degrees of freedom that includes:
%     - Number of rotational parameters: nchoosek(r,2) for r >= 2, 0 for r=1
%     - Number of factor lengths: r
%     - Number of variance parameters: p (dimension)
%     - Eigenvalue estimation bias correction term
%     - Cross-term correction for signal-noise interaction
%
%   The model with the smallest GIC value is selected.
%
%   Inputs:
%     lam - p×1 vector of eigenvalues (descending order)
%           Eigenvalues of the sample covariance matrix.
%           Must satisfy: lam(1) >= lam(2) >= ... >= lam(p) > 0
%
%     s2  - Residual variance estimate (scalar > 0)
%           Typically the maximum likelihood estimate:
%           s2 = mean(lam(r+1:p))
%
%     r   - Number of latent factors (positive integer)
%           The number of principal components being evaluated.
%           Must satisfy: 1 <= r <= p-1
%
%     n   - Sample size (positive integer)
%           Number of observations used to compute the sample covariance.
%
%   Outputs:
%     rgic  - GIC value for the r-factor model (scalar)
%             Lower values indicate better model fit.
%             Compare across different values of r to select best model.
%
%     negll - Negative log-likelihood (scalar)
%             -log p(X | theta) where theta = (A, sigma^2)
%             Measures goodness of fit (lower is better).
%
%     dof   - Effective degrees of freedom (scalar)
%             Adjusted parameter count accounting for estimation bias.
%             Always >= number of free parameters.
%
%   Algorithm:
%     1. Compute effective degrees of freedom:
%        - Rotational parameters: nchoosek(r,2) if r >= 2, else 0
%        - Scale parameters: r (factor lengths)
%        - Variance parameters: p (diagonal elements)
%        - Eigenvalue variance correction: sum(lam(r+1:p)^2) / ...
%        - Signal-noise interaction: double sum over j,l
%
%     2. Compute negative log-likelihood:
%        negll = (p-r)*log(s2) + sum(log(lam(1:r)))
%
%     3. Apply GIC penalty:
%        rgic = negll + (2/n)*dof
%
%   Examples:
%     % Example 1: Evaluate GIC for different numbers of components
%     [delta, U, params, X] = generate_pca_data(100, 10, 'J', 2, 'SNR', 4);
%     Jmax = 5;
%     gic_values = zeros(Jmax, 1);
%     
%     for r = 1:Jmax
%         s2_ml = mean(delta(r+1:end));
%         gic_values(r) = pca_gic(delta, s2_ml, r, 100);
%     end
%     
%     [~, best_r] = min(gic_values);
%     fprintf('GIC selected %d components\n', best_r);
%
%   Notes:
%     - Input eigenvalues LAM must be sorted in descending order
%     - Residual variance S2 should be positive and less than lam(r)
%     - For r = p, the model is saturated (no degrees of freedom left)
%     - GIC is invariant to scaling of the data
%     - Unlike BIC, GIC penalty is O(1/n) not O(log(n)/n)
%
%   Validation:
%     The function does not validate inputs. Ensure:
%       - lam is sorted: lam(1) >= lam(2) >= ... >= lam(p) > 0
%       - s2 > 0 and typically s2 < lam(r)
%       - 1 <= r < p
%       - n > p (for well-posed estimation)
%
%   See also ESTIMATE_TAU_MLE, ESTIMATE_TAU_MML, BIC_PCA, LAPLACE_PCA,
%   SIGNFLIPPA, PCA_NML
%
%   References:
%     [1] H. Hung, S.-Y. Huang, and C.-K. Ing, "A generalized information
%         criterion for high-dimensional PCA rank selection," Statistical
%         Papers, vol. 63, no. 4, pp. 1295-1321, 2022.
%
%     [2] E. Makalic and Daniel F. Schmidt, "MML Probabilistic Principal
%         Component Analysis," arXiv:2209.14559 [stat.ME], 2026.
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2024-
function [rgic, negll, dof] = pca_gic(lam, s2, r, n)

p = length(lam);

%% calculate DOF
sumterm = 0;
for j = 1:r
    for l = (r+1):p
        sumterm = sumterm + (lam(l) * (lam(j) - s2))/(s2*(lam(j) - lam(l)));
    end
end

ix = (r+1):p;
if(r < 2)
    dof = r + p + sum(lam(ix).^2)/(p-r) / ( sum(lam(ix))/(p-r) )^2 + sumterm;
else
    dof = nchoosek(r,2) + r + p + sum(lam(ix).^2)/(p-r) / ( sum(lam(ix))/(p-r) )^2 + sumterm;
end

%% negative log-likelihood
negll = (p-r)*log(s2) + sum(log(lam(1:r)));

%% GIC for this r
rgic = negll + (2/n)*dof;

end
