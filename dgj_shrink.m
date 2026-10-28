function [tau, x] = dgj_shrink(s, N, J, loss, mpMedian, tauIn)
%DGJ_SHRINK Optimal eigenvalue shrinkage of Donoho, Gavish and Johnstone (2018).
%
% [TAU, X] = DGJ_SHRINK(S, N, J, LOSS) shrinks the J leading sample
% eigenvalues with the asymptotically optimal shrinker for LOSS and returns
% the noise variance TAU and the signal strengths X (shrunken eigenvalue
% minus TAU), so that Sigmahat = U_J*diag(X)*U_J' + TAU*I.
%
% Inputs
%   s        - K x 1 sample covariance eigenvalues (descending) computed
%              from centred data with divisor N
%   N        - sample size
%   J        - number of leading eigenvalues to shrink
%   loss     - 'entropy' : optimal for tr(Sigmahat^-1 Sigma) - log det - K
%                          = 2 KL(N(0,Sigma) || N(0,Sigmahat))
%              'stein'   : optimal for tr(Sigma^-1 Sigmahat) - log det - K
%                          = 2 KL(N(0,Sigmahat) || N(0,Sigma))
%   mpMedian - (optional) median of the Marchenko-Pastur law with ratio
%              gamma = K/(N-1); computed if omitted
%   tauIn    - (optional) noise variance to use instead of the median
%              estimate; DGJ (Section 9.2) allow any consistent estimate
%
% Following DGJ (Ann. Statist. 46:1742-1778), with gamma = p/n,
%   ell(lambda) = [(lambda+1-gamma) + sqrt((lambda+1-gamma)^2 - 4 lambda)]/2,
%   c^2(ell)    = (1 - gamma/(ell-1)^2)/(1 + gamma/(ell-1)),  s^2 = 1 - c^2,
%   Stein:   eta = ell/(c^2 + ell s^2),
%   Entropy: eta = ell c^2 + s^2,
% and eta = 1 for lambda <= (1+sqrt(gamma))^2. The unknown noise level is
% estimated by sigma2 = median(lambda)/mu_gamma (DGJ Section 9.2), and the
% shrinker is applied to lambda/sigma2 and rescaled by sigma2. The sample
% covariance uses divisor n = N-1 to account for the estimated mean.

s = s(:);
K = numel(s);
n = N - 1;
lam = s*N/n;
gamma = K/n;
if nargin < 5 || isempty(mpMedian)
    mpMedian = mp_median(gamma);
end

if nargin >= 6 && ~isempty(tauIn)
    tau = tauIn;                 % an estimate of sigma^2 (e.g. PLY)
else
    tau = median(lam)/mpMedian;
end
if J == 0
    x = zeros(0,1);
    return
end

z = lam(1:J)/tau;
eta = ones(J,1);
above = z > (1 + sqrt(gamma))^2;
ell = ((z + 1 - gamma) + sqrt(max((z + 1 - gamma).^2 - 4*z, 0)))/2;
c2 = (1 - gamma./(ell - 1).^2)./(1 + gamma./(ell - 1));
s2 = 1 - c2;
switch lower(loss)
    case 'stein'
        e = ell./(c2 + ell.*s2);
    case 'entropy'
        e = ell.*c2 + s2;
    otherwise
        error('dgj_shrink:loss','LOSS must be ''stein'' or ''entropy''.');
end
eta(above) = e(above);
x = tau*(eta - 1);
end

