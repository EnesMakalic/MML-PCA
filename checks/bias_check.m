function T = bias_check(R)
%BIAS_CHECK Empirical vs first-order (Theorem 1) bias of tau_hat, ML and MML87.
% Mean estimated; N*S ~ W_K(N-1, Sigma) drawn by the Bartlett decomposition
% (the estimators depend on the data only through the eigenvalues of S).
% MML87 fits without a regular solution are completed by ML (Theorem 1).
% Variance reduction: control variate c = mean of the noise-block diagonal of
% S minus tau(N-1)/N, which has expectation exactly zero, so
% mean(tau_hat - tau - c) is an unbiased estimate of the bias.
if nargin < 1, R = 1e5; end
K = 10; tau = 1;
cfg = {5, [10 5], [20 15 10 5], [4 2]};
Ns = [25 50 100 200 500 1000 5000];
rows = {};
for ic = 1:numel(cfg)
    rho = cfg{ic}(:); J = numel(rho); m = K - J;
    eta = tau*(1 + rho); sd = sqrt([eta; tau*ones(m,1)]);
    bML  = -(J + 1 + sum(1./rho));            % N*bias/tau, Theorem 1
    bMML = (J^2 + sum(1./rho))/m;
    for N = Ns
        nu = N - 1;
        eML = zeros(R,1); eMML = zeros(R,1); eMLc = eML; eMMLc = eML; reg = false(R,1);
        parfor r = 1:R
            st = RandStream('Threefry','Seed',ic*100 + find(Ns==N)); st.Substream = r; RandStream.setGlobalStream(st);
            A = tril(randn(st,K,K),-1);
            A(1:K+1:end) = sqrt(2*randg((nu - (0:K-1))/2));
            L = sd.*A;                         % diag(sd)*A
            S = (L*L')/N;
            s = sort(eig((S+S')/2),'descend');
            tML = mean(s(J+1:end));
            [t,~,info] = solve_tau_mml(s,N,J);
            if ~info.ok, t = tML; end
            reg(r) = info.ok;
            c = mean(diag(S(J+1:end,J+1:end))) - tau*nu/N;
            eML(r) = N*(tML - tau);  eMML(r) = N*(t - tau);
            eMLc(r) = N*(tML - tau - c); eMMLc(r) = N*(t - tau - c);
        end
        se = @(v) std(v)/sqrt(R);
        rows(end+1,:) = {mat2str(rho'), N, 100*mean(reg), bML, mean(eMLc), se(eMLc), ...
            bMML, mean(eMMLc), se(eMMLc), mean(eML), se(eML), mean(eMML), se(eMML)}; %#ok<AGROW>
        fprintf('rho=%s N=%5d  ML %7.3f (%.3f) vs %7.3f | MML %7.3f (%.3f) vs %7.3f | reg %.2f%%\n', ...
            mat2str(rho'), N, mean(eMLc), se(eMLc), bML, mean(eMMLc), se(eMMLc), bMML, 100*mean(reg));
    end
end
T = cell2table(rows,'VariableNames',{'rho','N','regular_pct','ML_theory','ML_emp','ML_se', ...
    'MML_theory','MML_emp','MML_se','ML_raw','ML_raw_se','MML_raw','MML_raw_se'});
end
