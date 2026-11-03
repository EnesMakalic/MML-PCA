function T = thm2_sim(R, Ns, cfg, seed0)
%THM2_SIM Monte Carlo check of Theorem 2 (fixed-dimension CLT) for MML87
% and ML. theta = (tau, eta_1..eta_J), eta_j = tau + x_j. Theorem 2 is
% equivalent to sqrt(N)(theta_hat - theta) => N(0, diag(v, 2 eta_j^2)),
% v = 2 tau^2/(K-J). Reports, for each estimator: the largest |mean| of the
% standardised errors Z = sqrt(N)(theta_hat - theta)./sd_limit, the range of
% their variances (limit 1), the largest |correlation| (limit 0), the
% coverage of 95% plug-in Wald intervals for tau and each x_j, and the
% size of sqrt(N)(tau_MML - tau_ML) (limit 0). tau = 1, mean estimated:
% N*S ~ W_K(N-1, Sigma) via the Bartlett decomposition. MML87 fits without
% a regular solution are completed by ML at the same J. CFG is a cell
% array of {K, rho} rows; SEED0 is the base seed (default 3000). The
% defaults (four designs, N = 50 to 1e5, R = 2e5) take about 75 seconds on
% 14 workers.
if nargin < 1, R = 2e5; end
if nargin < 2, Ns = [50 200 1000 5000 20000 100000]; end
if nargin < 3 || isempty(cfg)
    cfg = {6, [5 2]; 10, 3; 20, [10 5 2]; 5, 1};   % {K, rho}
end
if nargin < 4, seed0 = 3000; end
rows = {}; id = 0;
for ic = 1:size(cfg,1)
    K = cfg{ic,1}; rho = cfg{ic,2}(:); J = numel(rho); m = K - J;
    eta = 1 + rho; v = 2/m; sdl = sqrt([v; 2*eta.^2]);
    for N = Ns
        id = id + 1; nu = N - 1;
        sd = sqrt([eta; ones(m,1)]);
        thML = zeros(R,J+1); thM = zeros(R,J+1); reg = false(R,1);
        covML = false(R,J+1); covM = false(R,J+1);
        parfor r = 1:R
            st = RandStream('Threefry','Seed',seed0+id); st.Substream = r;
            RandStream.setGlobalStream(st);
            A = tril(randn(K,K),-1);
            A(1:K+1:end) = sqrt(2*randg((nu - (0:K-1))/2));
            L = sd.*A; S = (L*L')/N;
            s = sort(eig((S+S')/2),'descend'); s = max(s,0);
            tML = mean(s(J+1:end)); eML = s(1:J);
            [t,x,info] = solve_tau_mml(s,N,J);
            if info.ok, eM = t + x(:); reg(r) = true; else, t = tML; eM = eML; end
            a = [tML; eML]; b = [t; eM];
            thML(r,:) = a'; thM(r,:) = b';
            covML(r,:) = cover(a, [1; eta], m, N)';
            covM(r,:)  = cover(b, [1; eta], m, N)';
        end
        th0 = [1; eta]';
        [mML, vML, cML] = summ(thML, th0, sdl', N);
        [mM,  vM,  cM ] = summ(thM,  th0, sdl', N);
        d = sqrt(N)*(thM(:,1) - thML(:,1));
        rows(end+1,:) = {K, mat2str(rho'), N, 100*mean(reg), ...
            mML, min(vML), max(vML), cML, 100*mean(covML(:,1)), 100*min(mean(covML(:,2:end),1)), ...
            mM,  min(vM),  max(vM),  cM,  100*mean(covM(:,1)),  100*min(mean(covM(:,2:end),1)), ...
            mean(d), sqrt(mean(d.^2))}; %#ok<AGROW>
        fprintf(['K=%2d rho=%-9s N=%6d reg=%6.2f%% | ML: |mean|<=%.3f var %.3f-%.3f |corr|<=%.3f cov tau %.1f%% x>=%.1f%%' ...
                 ' | MML: |mean|<=%.3f var %.3f-%.3f |corr|<=%.3f cov tau %.1f%% x>=%.1f%% | sqrtN(MML-ML) mean %.4f rms %.4f\n'], ...
            K, mat2str(rho'), N, 100*mean(reg), mML, min(vML), max(vML), cML, 100*mean(covML(:,1)), ...
            100*min(mean(covML(:,2:end),1)), mM, min(vM), max(vM), cM, 100*mean(covM(:,1)), ...
            100*min(mean(covM(:,2:end),1)), mean(d), sqrt(mean(d.^2)));
    end
end
T = cell2table(rows,'VariableNames',{'K','rho','N','regular_pct', ...
    'ML_maxabsmean','ML_minvar','ML_maxvar','ML_maxabscorr','ML_cov_tau','ML_cov_x_min', ...
    'MML_maxabsmean','MML_minvar','MML_maxvar','MML_maxabscorr','MML_cov_tau','MML_cov_x_min', ...
    'diff_mean','diff_rms'});
end

function [mx, vr, cr] = summ(th, th0, sdl, N)
% Standardised errors Z: largest |mean|, variances, largest |correlation|.
Z = sqrt(N)*(th - th0)./sdl;
mx = max(abs(mean(Z,1)));
C = cov(Z); vr = diag(C)';
Rm = C./sqrt(diag(C)*diag(C)');
cr = max(abs(Rm(~eye(size(Rm)))));
if isempty(cr), cr = 0; end
end

function c = cover(th, th0, m, N)
% 95% plug-in Wald intervals from V of Theorem 2, for tau and x_j.
t = th(1); e = th(2:end); v = 2*t^2/m;
se = sqrt([v; 2*e.^2 + v]/N);
est = [t; e - t]; tru = [th0(1); th0(2:end) - th0(1)];
c = abs(est - tru) <= 1.959963984540054*se;
end
