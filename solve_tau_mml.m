function [tau, x, info] = solve_tau_mml(s, N, J, varargin)
%SOLVE_TAU_MML Regular MML87 residual variance and loadings for a fixed rank.
%
% [TAU, X, INFO] = SOLVE_TAU_MML(S, N, J) returns the regular interior
% MML87 estimate of the residual variance TAU and the squared loading
% magnitudes X (J x 1) for probabilistic PCA with J components, given the
% sample covariance eigenvalues S (descending, divisor N) and sample size N.
%
% Algorithm (Algorithm 1 of the paper, with an existence certificate)
%   1. Damped fixed-point iteration
%        tau <- (1-w)*tau + w*g(tau),
%        g(tau) = [B + C*tau^2*sum_j 1/xhat_j(tau)]/A,
%      from tau0 = min{tauML, tauMax/2}, halving w whenever an update leaves
%      the admissible interval (0, tauMax). It stops when the undamped
%      residual satisfies |g(tau)-tau| <= Tol*tau. The iterate is used only
%      as a starting value: it is not accepted until step 2 certifies it.
%   2. Every fit is then certified with the scaled score
%        H(t) = A*t - B - C*t^2*sum_j 1/xhat_j(t) = 2 t^2 dI/dt.
%      H is strictly concave on (0, tauMax) with H(0+) = -B < 0, so a
%      regular interior local minimum exists if and only if max H > 0; it is
%      then unique and is the first zero of H. The maximiser of H and that
%      zero are found by bracketed root finding.
%
% Coefficients (em = 1 if the mean is estimated, 0 otherwise):
%   A = N(K-J) - JK - em(K-J),   B = N*sum_{k>J} s_k,   C = K-J+1,
%   a = N+K-em,  p = N+2K-J+1-em = a+C,  q_j = N*s_j,
%   xhat_j(t) = {q_j - p t + sqrt(D_j(t))}/(2a),  D_j(t) = (q_j-p t)^2 - 4aC t^2,
%   tauMax = q_J/(p + 2 sqrt(aC)).
% Since C t^2/xhat_j(t) = {q_j - p t - sqrt(D_j(t))}/2,
%   H(t) = (A + J p/2) t - B - sum_j q_j/2 + sum_j sqrt(D_j(t))/2.
% For J = 0 the estimate is tau = B/A.
%
% Name-value options
%   'EstimateMean'  (default true)   'Tol'        (default 1e-12)
%   'MaxIter'       (default 10000)  'Damping'    (default 0.5)
%   'MinDamping'    (default 1e-8)
%
% INFO fields
%   status      'fixedpoint'  fixed point converged and was certified by H
%               'fallback'    fixed point failed; minimum found from H
%               'none'        no regular interior minimum (A <= 0, empty
%                             admissible interval, or max H <= 0)
%               'tangent'     max H is zero to working precision, so the
%                             score touches zero without a sign change
%               'unresolved'  numerical failure; existence not determined
%   ok          true for 'fixedpoint' and 'fallback'
%   iterations  fixed-point iterations used
%   tauMax, tauRatio (= tau/tauMax), maxH, A, B, C

p = inputParser;
p.addParameter('EstimateMean', true, @islogical);
p.addParameter('Tol', 1e-12, @(v) isscalar(v) && v > 0);
p.addParameter('MaxIter', 10000, @(v) isscalar(v) && v >= 1);
p.addParameter('Damping', 0.5, @(v) isscalar(v) && v > 0 && v <= 1);
p.addParameter('MinDamping', 1e-8, @(v) isscalar(v) && v > 0 && v <= 1);
p.parse(varargin{:});
o = p.Results;

s = s(:); K = numel(s); em = double(o.EstimateMean);
A = N*(K-J) - J*K - em*(K-J);
B = N*sum(s(J+1:end));
C = K-J+1;
tau = NaN; x = zeros(J,1)*NaN;
info = struct('status','none','ok',false,'iterations',0,'tauMax',NaN, ...
    'tauRatio',NaN,'maxH',NaN,'A',A,'B',B,'C',C);

if J == 0
    if A > 0 && B > 0
        tau = B/A; x = zeros(0,1);
        info.status = 'fixedpoint'; info.ok = true;
    end
    return
end
if A <= 0 || B <= 0 || s(J) <= 0, return; end   % no regular solution

a  = N + K - em;
pp = N + 2*K - J + 1 - em;                        % = a + C
q  = N*s(1:J);
tauMax = q(J)/(pp + 2*sqrt(a*C));
info.tauMax = tauMax;
if ~(isfinite(tauMax) && tauMax > 0), return; end

%% 1. Damped fixed-point iteration (Algorithm 1)
tauML = B/(N*(K-J));
t = min(tauML, 0.5*tauMax);
w = o.Damping;
fpOK = false;
for it = 1:o.MaxIter
    xt = xprof(t);
    if any(~isfinite(xt)), break; end
    g = (B + C*t^2*sum(1./xt))/A;
    if abs(g - t) <= o.Tol*t
        fpOK = Hprime(t) > 0;                     % upward crossing: a minimum
        break
    end
    tn = (1-w)*t + w*g;
    while ~(tn > 0 && tn < tauMax)
        w = w/2;
        if w < o.MinDamping, break; end
        tn = (1-w)*t + w*g;
    end
    if w < o.MinDamping, break; end
    t = tn;
end
info.iterations = it;

%% 2. Existence certificate from the concave scaled score H (always applied)
% A small fixed-point residual and H'(tau) > 0 do not by themselves prove
% that H crosses zero: near tangency, max H can be slightly negative. A fit
% is therefore accepted only when max H > 0, and the estimate is the zero
% of H on (0, tstar), located within a sign-changing bracket.
try
    % Work in the scale-free variable u = tau/tauMax in (0,1): fzero's
    % stopping rule is relative to max(|u|,1), so tolerances then mean the
    % same thing at every scale of the data.
    Hu  = @(u) H(u*tauMax);
    Hpu = @(u) Hprime(u*tauMax);
    lo = 1e-14; hi = 1 - 1e-14;
    opts = optimset('TolX', 1e-15, 'Display', 'off');
    if Hpu(hi) >= 0
        ustar = hi;                               % H increasing on the interval
    else
        ustar = fzero(Hpu, [lo, hi], opts);
    end
    tstar = ustar*tauMax;
    Hmax = H(tstar);
    info.maxH = Hmax;
    scale = B + sum(q);                           % magnitude of the terms in H
    if Hmax < -1e-13*scale
        info.status = 'none'; return
    elseif Hmax <= 1e-13*scale
        info.status = 'tangent'; return
    end
    ut = t/tauMax;
    if fpOK && ut > lo && ut < ustar && H(t) < 0
        br = [ut, ustar];                         % fixed point lies just below the root
    elseif fpOK && ut > lo && ut < ustar
        br = [lo, ut];
    else
        br = [lo, ustar];
    end
    tau = fzero(Hu, br, opts)*tauMax;
    x = xprof(tau);
    if any(~isfinite(x)) || Hprime(tau) <= 0
        error('solve_tau_mml:check', 'Certified solution failed its checks.');
    end
    if fpOK, info.status = 'fixedpoint'; else, info.status = 'fallback'; end
    info.ok = true; info.tauRatio = tau/tauMax;
catch
    tau = NaN; x = NaN(J,1);
    info.status = 'unresolved';
end

    function xx = xprof(tt)
        d = q - pp*tt; D = d.^2 - 4*a*C*tt^2;
        if ~(tt > 0 && tt < tauMax) || any(d <= 0) || any(D <= 0)
            xx = NaN(J,1); return
        end
        xx = (d + sqrt(D))/(2*a);
    end
    function v = H(tt)
        d = q - pp*tt; D = max(d.^2 - 4*a*C*tt^2, 0);
        v = (A + J*pp/2)*tt - B - sum(q)/2 + sum(sqrt(D))/2;
    end
    function v = Hprime(tt)
        d = q - pp*tt; D = max(d.^2 - 4*a*C*tt^2, realmin);
        v = A + J*pp/2 + sum((-2*pp*d - 8*a*C*tt)./(4*sqrt(D)));
    end
end
