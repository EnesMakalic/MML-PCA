%SIGNFLIPPA    SignFlip Parallel Analysis for principal component selection
%   K = SIGNFLIPPA(X) estimates the number of significant principal 
%   components in data matrix X using the SignFlip Parallel Analysis method.
%   The method uses random sign-flipping permutations to construct a null
%   distribution of singular values and compares the observed singular 
%   values against the 95th percentile threshold.
%
%   K = SIGNFLIPPA(X, T) uses T permutations to construct the null 
%   distribution (default: T = 10).
%
%   K = SIGNFLIPPA(X, T, CUTOFF) uses the CUTOFF-th percentile as the 
%   significance threshold, where CUTOFF must be in the range [50, 100]
%   (default: CUTOFF = 95).
%
%   Description:
%   SignFlip Parallel Analysis is a permutation-based method for determining
%   the number of principal components. For each of T iterations, the method:
%     1. Randomly flips the sign of each element in X with probability 0.5
%     2. Computes the singular values of the sign-flipped matrix
%     3. Stores these null singular values
%   
%   The k-th component is deemed significant if the k-th data singular value
%   exceeds the CUTOFF-th percentile of the k-th null singular values. The
%   algorithm stops at the first non-significant component.
%
%   Algorithm:
%     For each permutation t = 1,...,T:
%       - Generate random sign matrix R (elements are ±1)
%       - Compute SVD of X .* R
%       - Store singular values in null distribution
%     
%     For each component i = 1,...,p:
%       - Compute CUTOFF-th percentile of i-th null singular values
%       - If data singular value S(i) > threshold, increment k
%       - Else break (remaining components not significant)
%
%   Inputs:
%     X      - n×p data matrix (numeric, real-valued)
%              n = number of observations (rows)
%              p = number of variables (columns)
%     T      - Number of permutations (default: 10)
%              Positive integer. Higher values give more stable estimates
%              but increase computational cost. Typical values: 10-100.
%     CUTOFF - Percentile threshold (default: 95)
%              Must be in range [50, 100]. Common values:
%              90 (liberal), 95 (standard), 99 (conservative)
%
%   Outputs:
%     k      - Estimated number of significant principal components
%              Non-negative integer in range [0, p]
%
%   Examples:
%     % Example 1: Basic usage with default parameters
%     [~, ~, ~, X] = generate_pca_data(100, 10, 'J', 2, 'SNR', 4);
%     k = signflippa(X);
%
%     % Example 2: Use 100 permutations and 99th percentile
%     k = signflippa(X, 100, 99);
%
%     % Example 3: More liberal component selection
%     k = signflippa(X, 50, 90);
%
%     % Example 4: Compare with other methods
%     k_spa = signflippa(X);
%     [k_bic, ~, ~] = bic_pca(X);
%     fprintf('SignFlip PA: %d, BIC: %d\n', k_spa, k_bic);
%
%   See also BIC_PCA, LAPLACE_PCA, PCA_GIC, ESTIMATE_TAU_MML
%
%   References:
%     [1] D. Hong, Y. Sheng, and E. Dobriban, "Selecting the number of 
%         components in PCA via random signflips," arXiv preprint, 2023.
%
%     [2] E. Makalic and Daniel F. Schmidt, "MML Probabilistic Principal 
%         Component Analysis," arXiv:2209.14559 [stat.ME], 2026.
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2024-
function k = signflippa(X, T, cutoff)
    arguments
        X (:,:) double {mustBeNumeric,mustBeReal}
        T (1,1) double {mustBePositive,mustBeInteger} = 10
        cutoff (1,1) double {mustBePositive,mustBeReal,mustBeInRange(cutoff,50,100)} = 95
    end    

% Centre data
mu = mean(X);
X = bsxfun(@minus, X, mu);

% Compute data singular values
[n,p] = size(X);
S = svd(X);

% permutations
Snull = zeros(T, p);
for t = 1:T
    % flip signs of X at random
    R = ones(n,p);
    R(rand(n*p,1) < 0.5) = -1; 
    Snull(t,:) = svd(X .* R)';    
end

% Determine k by computing how many leading data singular values are
% greater than the cutoff percentiles of the null distribution singular
% values
k = 0;
for i = 1:p
    alpha = prctile(Snull(:,i), cutoff);

    if(S(i) > alpha)
        k = k + 1;
    else
        break;
    end
end

end