function [eigenvalues, eigenvecs, true_params, x] = generate_pca_data(N, K, options)
% GENERATE_PCA_DATA Generate sample data for probabilistic PCA model
%
% Description:
%   Generates N samples from a K-dimensional probabilistic PCA model as
%   described in the MML PCA paper. The model is:
%   x_i ~ N(0_K, Sigma), where Sigma = A*A' + sigma^2*I_K
%
% Syntax:
%   [eigenvalues, eigenvecs, true_params] = generate_pca_data(N, K)
%   [eigenvalues, eigenvecs, true_params] = generate_pca_data(N, K, 'Name', Value)
%
% Inputs:
%   N - Number of samples (required)
%   K - Dimensionality of data (required)
%
% Optional Name-Value Pairs:
%   'J'          - Number of latent factors
%   'sigma2'     - Residual variance 
%   'alpha'      - Factor lengths, J×1 vector (default: random half-Cauchy)
%   'SNR'        - Signal-to-noise ratio 
%                  SNR = (1/(K*sigma^2)) * sum(alpha_j^2)
%   'seed'       - Random seed for reproducibility (default: not set)
%
% Outputs:
%   eigenvalues - K×1 vector of eigenvalues of sample covariance (sorted descending)
%   eigenvecs   - KxJ matrix of eigenvectors of the sample covariance
%   true_params - Structure containing true generating parameters:
%                 .A       - K×J factor loading matrix
%                 .sigma2  - Residual variance
%                 .alpha   - J×1 vector of factor lengths
%                 .Sigma   - K×K true covariance matrix
%  x            - NxK data matrix
%
% Example:
%   % Generate data with default parameters
%   [eigs, eigv] = generate_pca_data(100, 10);
%   
%   % Generate data with specific SNR
%   [eigs, eigv, params] = generate_pca_data(50, 10, 'J', 2, 'seed', 42);
%
% Reference:
% E. Makalic and Daniel F. Schmidt, MML Probabilistic Principal Component
% Analysis, arXiv:2209.14559 [stat.ME], 2026.
arguments
    N (1,1) {mustBePositive, mustBeInteger} = 100
    K (1,1) {mustBePositive, mustBeInteger} = 10

    options.J {mustBeEmptyOrPositiveInteger} = []
    options.sigma2 {mustBeEmptyOrPositiveScalar} = []
    options.alpha (:,1) {mustBePositive} = []
    options.SNR {mustBeEmptyOrPositiveScalar} = []    

    options.seed {mustBeInteger} = []

    options.sigma_min (1,1) {mustBePositive} = 1
    options.sigma_max (1,1) {mustBePositive} = 10
end

%% Extract options and set defaults that depend on other parameters
if isempty(options.J)
    J = 0;
else
    J = options.J;
end
sigma2 = options.sigma2;
sigma_min = options.sigma_min;
sigma_max = options.sigma_max;
alpha = options.alpha;
SNR = options.SNR;
seed = options.seed;

if(sigma_min >= sigma_max)
    warning('Swapping [a,b] the domain of sigma2');
    a = sigma_max; sigma_max = sigma_min; sigma_min = a;
end

if ~isempty(alpha)
    % Use provided alpha
    if length(alpha) ~= J
        error('Length of alpha must equal J=%d', J);
    end
    alpha = alpha(:); % ensure column vector
end

%% Set random seed if specified
if ~isempty(seed)
    rng(seed);
end

%% Determine sigma2
if(isempty(sigma2)) % if sigma2 not specified...
    a = sigma_min; b = sigma_max;

    % Sample sigma2 from distribution h(sigma) \propto 1/sigma
    u = rand(1) * (log(b) - log(a)) + log(a); % uniform in [log a, log b]
    sigma2 = exp(2*u);
end

%% Determine alpha
if(isempty(alpha))
    % Sample factor lengths from half-Cauchy distribution
    alpha = abs(trnd(1, J, 1)); 
end

%% If SNR was specified ...
if ~isempty(SNR)
    c = sqrt(K*SNR*sigma2 / sum(alpha.^2)); %  ensures SNR = sum(alpha.^2) / (K*s2)
    alpha = alpha*c;    
end

%% Determine factor loading matrix A
switch(J)
case 0 % No latent factors
    A = 0;
otherwise % J > 0
    R = randn(K,J);     % Sample directions uniformly from a unit n-sphere
    R = bsxfun(@rdivide, R, sqrt(sum(R.^2)));
    A = R*diag(alpha);
end

%% Construct true covariance matrix Sigma = A*A' + sigma^2*I_K
Sigma = A*A' + sigma2*eye(K);

%% Sample data
x = mvnrnd(zeros(K,1),Sigma,N); 
Sx = (x'*x)/N;                  % sample variance-covariance matrix
[U,S,~] = svd(Sx,0);            % SVD of Sx

%% Return values
eigenvalues = diag(S);     
eigenvecs = U;

true_params.A = A;
true_params.sigma2 = sigma2;
true_params.alpha = alpha;
true_params.Sigma = Sigma;

end

%% Local validation functions
function mustBeEmptyOrPositiveScalar(x)
    if ~isempty(x)
        mustBeNumeric(x);
        mustBeScalarOrEmpty(x);
        mustBePositive(x);
    end
end

function mustBeEmptyOrPositiveInteger(x)
    if ~isempty(x)
        mustBeInteger(x);
        mustBeScalarOrEmpty(x);
        mustBePositive(x);
    end
end
