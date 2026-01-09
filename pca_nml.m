%PCA_NML    Normalized Maximum Likelihood criterion for PCA model selection
%   SC = PCA_NML(X, M, ~, DELTA, SIGMA_ML, S2_ML) computes the Normalized
%   Maximum Likelihood (NML) stochastic complexity for a probabilistic PCA
%   model with M latent factors given data X, eigenvalues DELTA, ML
%   covariance estimate SIGMA_ML, and ML residual variance S2_ML.
%
%   [SC, PC, NEGLL] = PCA_NML(X, M, ~, DELTA, SIGMA_ML, S2_ML) also returns
%   the parametric complexity PC and negative log-likelihood NEGLL.
%
%   [SC, PC, NEGLL] = PCA_NML(X, M, ~, DELTA, SIGMA_ML, S2_ML, STYPE)
%   specifies the method for computing the discretization parameter s.
%
%   Description:
%   The Normalized Maximum Likelihood (NML) is a model selection criterion
%   based on the minimum description length (MDL) principle. The NML
%   stochastic complexity is defined as: 
%     SC = -log p(X|theta_ML) + log( integral p(X|theta) dX )
%        = NegLL + PC
%
%   where PC (parametric complexity) accounts for model complexity via the
%   volume of the parameter space, properly normalized. The model with the
%   smallest SC is selected.
%
%   Inputs:
%     x        - N×K data matrix
%                N observations (rows) of K-dimensional data (columns).
%                Should typically be centered (mean = 0) for PCA.
%
%     m        - Number of latent factors (positive integer)
%                The number of principal components being evaluated.
%                Must satisfy: 1 <= m < K
%
%     delta    - K×1 vector of eigenvalues (descending order)
%                Eigenvalues of the sample covariance matrix (1/N)*X'*X.
%                Must satisfy: delta(1) >= delta(2) >= ... >= delta(K) > 0
%                Note: Currently unused in computation but kept for API.
%
%     Sigma_ml - K×K covariance matrix estimate (ML estimate)
%                Maximum likelihood estimate of the covariance matrix.
%                Used to compute the negative log-likelihood.
%                Must be symmetric positive definite.
%
%     s2_ml    - Residual variance estimate (ML estimate, scalar > 0)
%                Maximum likelihood estimate of noise variance.
%                Note: Currently unused in computation but kept for API.
%
%     stype    - Discretization method (optional, default: 0)
%                Determines how to compute the discretization parameter s:
%                  0 - Fixed: s = 64 (default, most robust)
%                  1 - Adaptive: s = ceil((log2(trace(S_int)) - log2(K))/2)
%                  2 - Max eigenvalue: s = ceil(log2(max(eig(S_int)))/2)
%                where S_int is the integer-scaled sample covariance.
%
%   Outputs:
%     sc    - Stochastic complexity (scalar)
%             Total NML codelength = PC + NegLL.
%             Lower values indicate better model fit.
%             Compare across different values of m to select best model.
%
%     pc    - Parametric complexity (scalar)
%             Measures model complexity via parameter space volume.
%             Includes discretization effects and finite-sample corrections.
%             Always positive.
%
%     negll - Negative log-likelihood (scalar)
%             -log p(X | theta_ML) where theta_ML = (A_ML, sigma^2_ML)
%             Measures goodness of fit (lower is better).
%
%   Examples:
%     % Example 1: Evaluate NML for different numbers of components
%     [delta, U, params, X] = generate_pca_data(100, 10, 'J', 2, 'SNR', 4);
%     N = size(X, 1); K = size(X, 2);
%     Sx = (X'*X)/N;
%     Jmax = 5;
%     nml_values = zeros(Jmax, 1);
%     
%     for m = 1:Jmax
%         s2_ml = mean(delta(m+1:end));
%         [~, ~, Sigma_ml] = estimate_tau_mle(delta, U, m);
%         nml_values(m) = pca_nml(X, m, [], delta, Sigma_ml, s2_ml, 1);
%     end
%     
%     [~, best_m] = min(nml_values);
%     fprintf('NML selected %d components\n', best_m);
%
%   See also ESTIMATE_TAU_MLE, ESTIMATE_TAU_MML, PCA_GIC, BIC_PCA,
%   LAPLACE_PCA, SIGNFLIPPA, MML_PCA_CODELENGTH
%
%   References:
%     [1] B. Mera, P. Mateus, and A. M. Carvalho, "Model complexity in
%         statistical manifolds: The role of curvature," IEEE Trans. Inf.
%         Theory, vol. 68, no. 9, pp. 5619-5636, 2022.
%
%     [2] A. Tavory, "Determining principal component cardinality through
%         the principle of minimum description length," in Machine Learning,
%         Optimization, and Data Science, 2019, pp. 655-666.
%
%     [3] E. Makalic and Daniel F. Schmidt, "MML Probabilistic Principal
%         Component Analysis," arXiv:2209.14559 [stat.ME], 2026.
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2024-
function [sc, pc, negll] = pca_nml(x, m, ~, delta, Sigma_ml, s2_ml, stype)

[N,K] = size(x);

%% Determine s
if(nargin < 7)
    stype = 0;
end

prec = 15;
xs = round( bsxfun(@times, x, 10^(prec)) );
Sint = (xs'*xs) / N;
switch stype
    case 0
        s = 64;
    case 1
        s = ceil( (log2(trace(Sint)) - log2(K))/2 );
    case 2
        [~,e,~] = svd(Sint); 
        s = ceil( log2( max(diag(e)) ) / 2 );
    otherwise
        error('invalid s estimate');
end

%% Compute stochastic complexity
% Negative log-likelihood
negll = -sum(mvnormpdfln(x', zeros(1,K)', [], Sigma_ml));

% Parametric complexity
%logIs = s*log(2)*m*(m-1)/2 + (m*log(s*log(2)) + m*(m-1)/4*log(2));
logIs = m*log(s*log(2)) + (2*s + 1)*m*(m-1)/4*log(2);
logVol = -(3/2)*m*log(2) - gammaln(m+1) + m*log(2) + m*(m+1)/4*log(pi) - sum( gammaln((1:m)/2) ) + logIs;
pc = m*(m+1)/4*log(N/2/pi) + logVol - (m+2)*m*(m-1)/(24*N);

% Stochastic complexity
sc = pc + negll;

end
