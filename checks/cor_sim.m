function T = cor_sim(R, Ns, cfg, seed0)
%COR_SIM Monte Carlo check of Corollary 1 (MML87 has smaller limiting Stein
% loss than ML) and of the entropy-loss remark after it (no reverse result),
% in the proportional regime K = gamma*N of Proposition 1.
% For each (gamma, rho) and N, reports the mean of L(ML) - L(MML87) for
% Stein and entropy loss with its standard error, its limit from
% Proposition 1, and the percentage of data sets in which MML87 has the
% smaller loss. tau = 1, Sigma = diag(1+rho, 1, ..., 1), mean estimated:
% N*S ~ W_K(N-1, Sigma) via the Bartlett decomposition. MML87 fits without
% a regular solution are completed by ML at the same J. Losses use the
% sample eigenvectors exactly. Defaults: R = 2000, Ns = [100 400 1600].
% CFG is a cell array of {gamma, rho} rows (default: one and two spikes);
% SEED0 is the base seed (default 1000).
% R = 5000 with Ns = [100 200 400 800 1600] takes about 23 minutes on 14
% workers.
if nargin < 1, R = 2000; end
if nargin < 2, Ns = [100 400 1600]; end
if nargin < 4, seed0 = 1000; end
if nargin < 3 || isempty(cfg)
    cfg = { 0.25, 1.5;  0.25, 2;  0.25, 5;  0.25, 10; ...
            0.5,  2.5;  0.5,  3;  0.5,  5;  0.5,  10; ...
            1,    4;    1,    6;  1,    10; ...
            0.25, [5 2]; 0.5, [8 4] };
end
rows = {}; id = 0;
for ic = 1:size(cfg,1)
    g = cfg{ic,1}; rho = cfg{ic,2}(:); J = numel(rho);
    [dSt, dEnt] = limits(g, rho);
    for N = Ns
        id = id + 1; K = round(g*N); nu = N - 1;
        sd = sqrt([1 + rho; ones(K-J,1)]);
        dS = zeros(R,1); dE = zeros(R,1); reg = false(R,1);
        LS = zeros(R,2); LE = zeros(R,2);
        parfor r = 1:R
            st = RandStream('Threefry','Seed',seed0+id); st.Substream = r;
            RandStream.setGlobalStream(st);
            A = tril(randn(st,K,K),-1);
            A(1:K+1:end) = sqrt(2*randg(max(nu - (0:K-1),0)/2));
            L = sd.*A; S = (L*L')/N; S = (S+S')/2;
            [V,E] = eig(S); [s,ix] = sort(diag(E),'descend'); s = max(s,0);
            Vj = V(:,ix(1:J));
            tML = mean(s(J+1:end)); yML = s(1:J);
            [t,x,info] = solve_tau_mml(s,N,J);
            if info.ok, yM = t + x(:); reg(r) = true; else, t = tML; yM = yML; end
            [sML,eML] = losses(tML,yML,Vj,rho,K);
            [sM, eM ] = losses(t,  yM, Vj,rho,K);
            LS(r,:) = [sML sM]; LE(r,:) = [eML eM];
            dS(r) = sML - sM; dE(r) = eML - eM;
        end
        se = @(v) std(v)/sqrt(R);
        rows(end+1,:) = {g, mat2str(rho'), N, K, 100*mean(reg), ...
            mean(LS(:,1)), mean(LS(:,2)), mean(dS), se(dS), dSt, 100*mean(dS>0), ...
            mean(LE(:,1)), mean(LE(:,2)), mean(dE), se(dE), dEnt, 100*mean(dE>0)}; %#ok<AGROW>
        fprintf(['g=%.2f rho=%-6s N=%4d reg=%6.2f%% | Stein ML-MML %7.4f (%.4f) lim %7.4f P(MML<ML)=%5.1f%%' ...
                 ' | Ent ML-MML %8.4f (%.4f) lim %8.4f P(MML<ML)=%5.1f%%\n'], ...
            g, mat2str(rho'), N, 100*mean(reg), mean(dS), se(dS), dSt, 100*mean(dS>0), ...
            mean(dE), se(dE), dEnt, 100*mean(dE>0));
    end
end
T = cell2table(rows,'VariableNames',{'gamma','rho','N','K','regular_pct', ...
    'Stein_ML','Stein_MML','Stein_diff','Stein_diff_se','Stein_diff_limit','Stein_PMMLbetter', ...
    'Ent_ML','Ent_MML','Ent_diff','Ent_diff_se','Ent_diff_limit','Ent_PMMLbetter'});
end

function [Ls, Le] = losses(t, y, Vj, rho, K)
% Stein and entropy losses of Sigma_hat = t*I + Vj*diag(y - t)*Vj'
% against Sigma = diag(1+rho, 1, ..., 1).
J = numel(rho);
w = [1./(1+rho); ones(K-J,1)];  d = [1+rho; ones(K-J,1)];
ldS = sum(log(1+rho));  ldH = (K-J)*log(t) + sum(log(y));
trSiH = t*sum(w) + sum((y - t)'.*sum(w.*Vj.^2,1));
trHiS = sum(d)/t + sum((1./y - 1/t)'.*sum(d.*Vj.^2,1));
Ls = trSiH - (ldH - ldS) - K;
Le = trHiS - (ldS - ldH) - K;
end

function [dSt, dEnt] = limits(g, rho)
% Limits of L(ML) - L(MML) for Stein and entropy loss (Corollary 1 and the
% entropy remark): sums over spikes of differences of per-spike excesses.
h = @(u) u - 1 - log(u);
b = rho - g + g./rho;
eM  = 1 + (b + sqrt(b.^2 - 4*g*(1+g)))/(2*(1+g));
lam = (1+rho).*(1+g./rho);
eSt = (1+rho).*(rho+g)./(rho*(1+g) + 2*g);
c2  = (1 - g./rho.^2)./(1 + g./rho);
eEn = 1 + rho.*c2;
dSt  = sum(h(lam./eSt) - h(eM./eSt));
dEnt = sum(h(eEn./lam) - h(eEn./eM));
end
