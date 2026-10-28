function [tau, alpha] = estimate_tau_ply(s, N, J)
%ESTIMATE_TAU_PLY Bias-corrected noise variance of Passemier, Li and Yao (2017).
%
% [TAU, ALPHA] = ESTIMATE_TAU_PLY(S, N, J) returns the bias-corrected
% estimate of the isotropic noise variance in probabilistic PCA with J
% spikes, together with the spike estimates ALPHA (estimates of the
% non-null eigenvalues of A*A').
%
% Inputs
%   s - K x 1 sample covariance eigenvalues (descending) computed from
%       centred data with divisor N, i.e. S = X_c'*X_c/N
%   N - sample size
%   J - number of spikes (0 <= J < K)
%
% Following Passemier, Li and Yao (2017, JRSS-B 79:51-67), the sample
% covariance uses divisor n-1 and c_n = p/(n-1). With sigma2 the ML
% estimate (mean of the p-m smallest eigenvalues), eq. (9) is
%
%   sigma2_* = sigma2 + b(sigma2)/(p-m) * sigma2 * sqrt(2 c_n),
%   b(sigma2) = sqrt(c_n/2) * (m + sigma2 * sum_i 1/alpha_i),
%
% where alpha_i is estimated by inverting lambda_i = sigma2*phi(alpha_i^*),
% phi(a) = a + c a/(a-1), alpha_i^* = alpha_i/sigma2 + 1. The inversion is
%
%   alpha_i^* = ell(lambda_i/sigma2),
%   ell(x) = [(x+1-c) + sqrt((x+1-c)^2 - 4x)]/2,
%
% evaluated at the ML sigma2 (one-step plug-in, as in the paper). Sample
% eigenvalues at or below the bulk edge sigma2*(1+sqrt(c))^2 are set to the
% edge, where alpha_i^* = 1 + sqrt(c). The returned ALPHA are the spike
% estimates sigma2_* * (ell(lambda_i/sigma2_*) - 1) at the corrected
% variance (zero for spikes at or below the edge).

s = s(:);
K = numel(s);
lam = s*N/(N-1);                 % eigenvalues of S with divisor n-1
c = K/(N-1);

sigma2 = mean(lam(J+1:K));
if J == 0
    tau = sigma2;
    alpha = zeros(0,1);
    return
end

astar = local_ell(lam(1:J)/sigma2, c);
b = sqrt(c/2)*(J + sum(1./(astar - 1)));     % sigma2*sum(1/alpha_i)
tau = sigma2 + b/(K - J)*sigma2*sqrt(2*c);

astar = local_ell(lam(1:J)/tau, c);
alpha = tau*(astar - 1);
alpha(lam(1:J)/tau <= (1+sqrt(c))^2) = 0;
end

function a = local_ell(x, c)
edge = (1 + sqrt(c))^2;
x = max(x, edge);
a = ((x + 1 - c) + sqrt(max((x + 1 - c).^2 - 4*x, 0)))/2;
end
