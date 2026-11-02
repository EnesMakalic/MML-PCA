function T = bias_full(R)
%BIAS_FULL Comprehensive check of Theorem 1: empirical vs first-order bias
% of tau_hat for ML and MML87 (tau = 1, mean estimated, distinct spikes).
% Spikes are geometric: rho_j = rhoJ * 1.5^(J-j), so rho_J is the weakest.
% N*S ~ W_K(N-1, Sigma) by the Bartlett decomposition. Fits without a
% regular MML87 solution are completed by ML at the same J (Theorem 1).
% Control variate c = mean noise-block diagonal of S - (N-1)/N, E(c) = 0.
if nargin < 1, R = 1e5; end
Ks = [5 10 20 50]; Js = [1 2 4]; rhoJs = [2 5 20];
Ns = [50 100 200 500 1000 5000];
rows = {}; id = 0;
for K = Ks
for J = Js
    if J >= K, continue; end
    m = K - J;
for rhoJ = rhoJs
    rho = rhoJ*1.5.^(J-(1:J))';
    sd = sqrt([1 + rho; ones(m,1)]);
    bML  = -(J + 1 + sum(1./rho));
    bMML = (J^2 + sum(1./rho))/m;
    for N = Ns
        id = id + 1; nu = N - 1;
        eML = zeros(R,1); eMML = zeros(R,1); reg = false(R,1);
        parfor r = 1:R
            st = RandStream('Threefry','Seed',id); st.Substream = r;
            RandStream.setGlobalStream(st);
            A = tril(randn(st,K,K),-1);
            A(1:K+1:end) = sqrt(2*randg(max(nu - (0:K-1),0)/2));
            L = sd.*A; S = (L*L')/N;
            s = sort(eig((S+S')/2),'descend'); s = max(s,0);
            tML = mean(s(J+1:end));
            [t,~,info] = solve_tau_mml(s,N,J);
            if ~info.ok, t = tML; end
            reg(r) = info.ok;
            c = mean(diag(S(J+1:end,J+1:end))) - nu/N;
            eML(r) = N*(tML - 1 - c); eMML(r) = N*(t - 1 - c);
        end
        se = @(v) std(v)/sqrt(R);
        rows(end+1,:) = {K, J, rhoJ, mat2str(round(rho',3)), N, 100*mean(reg), ...
            bML, mean(eML), se(eML), bMML, mean(eMML), se(eMML)}; %#ok<AGROW>
        fprintf('K=%2d J=%d rhoJ=%2g N=%5d | ML %8.3f (%.3f) vs %8.3f | MML %8.3f (%.3f) vs %8.3f | reg %6.2f%%\n', ...
            K, J, rhoJ, N, mean(eML), se(eML), bML, mean(eMML), se(eMML), bMML, 100*mean(reg));
    end
end
end
end
T = cell2table(rows,'VariableNames',{'K','J','rhoJ','rho','N','regular_pct', ...
    'ML_theory','ML_emp','ML_se','MML_theory','MML_emp','MML_se'});
end
