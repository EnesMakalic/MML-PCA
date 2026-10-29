%SIM_ASYMPTOTIC Simulation checks of Theorems 2 and 3 (supplementary Table S1).
% Both regimes use R replications at N = 100, 1000, 10000, with an estimated
% mean and the certified solver solve_tau_mml. Fits without a regular
% solution are completed by the maximum likelihood values.
%
% The estimates depend on the data only through the J largest sample
% eigenvalues and the trace of S. They are invariant to rotation of Sigma,
% so Sigma is taken diagonal. In the proportional regime the J largest
% eigenvalues are computed with svds, which avoids a full eigendecomposition
% when K = 5000.
%
% Writes results/final_asymptotic.csv. The N = 10000 proportional block
% (K = 5000) dominates the run time: about 5-6 hours on 14 workers.
here = fileparts(mfilename('fullpath'));
addpath(here);
if isempty(gcp('nocreate')), parpool; end
R  = 2e4;
Ns = [100 1000 10000];
zq = 1.959963984540054;
J  = 2;
out = {};

%% Fixed dimension: K = 6, (rho_1, rho_2) = (5, 2), tau = 1
K = 6; x0 = [5; 2]; m = K - J;
V = [2/m, -2/m*ones(1,J); -2/m*ones(J,1), diag(2*(1+x0).^2) + 2/m];
lim = [V(1,1) V(2,2) V(3,3) V(1,2) V(2,3) 0.95];
names = {'var_tau','var_x1','var_x2','cov_tau_x1','cov_x1_x2','coverage'};
for N = Ns
    Z = zeros(R,J+1); cover = false(R,1); ok = true(R,1);
    sd = sqrt([1+x0; ones(K-J,1)])';
    parfor r = 1:R
        rs = RandStream('Threefry','Seed',1); rs.Substream = r;
        X = randn(rs,N,K).*sd; X = X - mean(X,1);
        s = sort(eig(X'*X/N),'descend');
        [t,x,info] = solve_tau_mml(s,N,J); ok(r) = info.ok;
        if ~info.ok, t = mean(s(J+1:end)); x = s(1:J) - t; end
        Z(r,:) = sqrt(N)*[t-1, (x(:)-x0)'];
        se = sqrt(2/(N*m));
        cover(r) = t*exp(-zq*se) <= 1 && 1 <= t*exp(zq*se);
    end
    C = cov(Z);
    est = [C(1,1) C(2,2) C(3,3) C(1,2) C(2,3) mean(cover)];
    % Monte Carlo standard errors (delta method for the (co)variances)
    Zc = Z - mean(Z,1);
    mc = @(u) std(u)/sqrt(R);
    mcse = [mc(Zc(:,1).^2) mc(Zc(:,2).^2) mc(Zc(:,3).^2) mc(Zc(:,1).*Zc(:,2)) ...
            mc(Zc(:,2).*Zc(:,3)) sqrt(mean(cover)*(1-mean(cover))/R)];
    for i = 1:numel(names)
        out(end+1,:) = {'fixed', N, K, R, 100*mean(ok), names{i}, lim(i), est(i), mcse(i)}; %#ok<SAGROW>
    end
    fprintf('Fixed K=%d, N=%d done (regular %.2f%%)\n', K, N, 100*mean(ok));
end

%% Proportional regime: K = N/2, (rho_1, rho_2) = (8, 4), tau = 1
rho = [8; 4]; g = 0.5;
L = (1+rho).*(1+g./rho); b = L - 1 - 2*g; xi = sqrt(b.^2 - 4*g*(1+g));
z = (b + xi)/(2*(1+g)); beta = sum(1./z - 1./rho);
v = 2*(1+rho).^2.*(1-g./rho.^2);
lim = [beta, 2/g, (z.^2./xi.^2.*v)'];
names = {'mean_Ntau','var_Ntau','var_x1','var_x2'};
for N = Ns
    K = N/2;
    gN = K/N; bN = (1+rho).*(1+gN./rho) - 1 - 2*gN;
    zN = (bN + sqrt(bN.^2 - 4*gN*(1+gN)))/(2*(1+gN));
    sd = sqrt([1+rho; ones(K-J,1)])';
    nt = zeros(R,1); xz = zeros(R,J); ok = true(R,1);
    parfor r = 1:R
        rs = RandStream('Threefry','Seed',2); rs.Substream = r;
        X = randn(rs,N,K).*sd; X = X - mean(X,1);
        top = svds(X,J).^2/N; tr = sum(X(:).^2)/N;
        s = [top; repmat((tr - sum(top))/(K-J), K-J, 1)];   % same top J and trace
        [t,x,info] = solve_tau_mml(s,N,J); ok(r) = info.ok;
        if ~info.ok, t = mean(s(J+1:end)); x = s(1:J) - t; end
        nt(r) = N*(t - 1);
        xz(r,:) = sqrt(N)*(x(:) - zN)';
    end
    xc = xz - mean(xz,1); nc = nt - mean(nt);
    est  = [mean(nt) var(nt) var(xz)];
    mcse = [std(nt)/sqrt(R), std(nc.^2)/sqrt(R), std(xc.^2)/sqrt(R)];
    for i = 1:numel(names)
        out(end+1,:) = {'proportional', N, K, R, 100*mean(ok), names{i}, lim(i), est(i), mcse(i)}; %#ok<SAGROW>
    end
    fprintf('Proportional K=%d, N=%d done (regular %.2f%%)\n', K, N, 100*mean(ok));
end

T = cell2table(out, 'VariableNames', ...
    {'regime','N','K','R','regular_pct','quantity','limit','estimate','mcse'});
writetable(T, fullfile(here,'results','final_asymptotic.csv'));
disp(T)
