function T = sim_compare(niter, varargin)
%SIM_COMPARE Fixed-rank comparison of ML, MML87 and alternative estimators.
%
% T = SIM_COMPARE(NITER) compares, at a matched rank, the estimators
%   ML      maximum likelihood (Tipping & Bishop 1999)
%   MML     MML87 (solve_tau_mml: Algorithm 1 with existence certificate)
%   BC      first-order bias-corrected ML,
%           tau = tauML*[1 + (J + 1 + sum_j tauML/xML_j)/N], x = s_j - tau
%   PLY     Passemier, Li & Yao (2017) bias-corrected noise variance with
%           their spike estimates (estimate_tau_ply)
%   DGJent  Donoho, Gavish & Johnstone (2018) shrinker optimal for
%           entropy loss = 2 KL(Sigma0 || Sigmahat)   (dgj_shrink)
%   DGJst   DGJ shrinker optimal for Stein loss = 2 KL(Sigmahat || Sigma0)
%           (DGJent/DGJst use the DGJ median noise estimate, Section 9.2)
%   DGJentP, DGJstP  the same shrinkers with the PLY noise variance
%
% over N, per-component SNR and J (K fixed). In each data set MML87 is
% fitted at the true rank J if a regular solution exists, otherwise at the
% largest regular rank J' < J; every method is then fitted at that same
% rank. All methods use the sample eigenvectors. Data are centred.
%
% Metrics: S1 = log(sigmahat/sigma), S2 = S1^2, KL = KL(Sigma0||Sigmahat),
% rKL = KL(Sigmahat||Sigma0), Seta = mean_j {log(etahat_j/eta_j)/2}^2.
%
% Name-value options: 'N' ([25 50 100]), 'SNR' ([2 5 20]), 'J' ([1 2 4]),
% 'K' (10), 'Design' ('equal'), 'GeometricRatio' (0.75), 'Seed' (1).
% SNR is per component ('equal': rho_j = SNR; 'geometric': weakest
% component rho_J = SNR).
%
% Output T has one row per (N,SNR,J,method) with means, Monte Carlo
% standard errors, and the percentage of regular J-factor MML87 fits.

p = inputParser;
p.addParameter('N',      [25 50 100]);
p.addParameter('SNR',    [2 5 20]);
p.addParameter('J',      [1 2 4]);
p.addParameter('K',      10);
p.addParameter('Design', 'equal');
p.addParameter('GeometricRatio', 0.75);
p.addParameter('Seed',   1);
p.parse(varargin{:});
o = p.Results;
if nargin < 1 || isempty(niter), niter = 1e4; end

methods = {'ML','MML','BC','PLY','DGJent','DGJst','DGJentP','DGJstP'};
nm = numel(methods);
K = o.K; gr = o.GeometricRatio; design = o.Design;
rows = {};
cell_id = 0;
for N = o.N
    mpMed = mp_median(K/(N-1));
for snr = o.SNR
for J = o.J
    cell_id = cell_id + 1;
    base = o.Seed*1e6 + cell_id;
    if strcmpi(design,'geometric')
        snrGen = snr*sum(gr.^(0:J-1))/(K*gr^(J-1));
    else
        snrGen = snr*J/K;
    end
    M = NaN(niter, 5*nm);
    reg = false(niter,1); unres = false(niter,1);

    parfor i = 1:niter
        stream = RandStream('Threefry','Seed',base);
        stream.Substream = i;
        [s,U,tp] = generate_pca_data(N,K,'sigma2',1.0,'J',J,'SNR',snrGen, ...
            'LoadingDesign',design,'GeometricRatio',gr, ...
            'CenterData',true,'Stream',stream);
        Sigma0 = tp.Sigma; eta0 = tp.sigma2 + tp.lambda2(:);

        [tMML,xMML,Jr,ok,un] = local_mml_fallback(s,N,K,J);
        reg(i) = ok; unres(i) = un;

        tau = zeros(nm,1); X = zeros(J,nm);
        % ML
        tML = mean(s(Jr+1:K)); xML = max(s(1:Jr) - tML, 0);
        tau(1) = tML; X(1:Jr,1) = xML;
        % MML87
        tau(2) = tMML; X(1:Jr,2) = xMML;
        % first-order bias-corrected ML
        if Jr > 0
            tBC = tML*(1 + (Jr + 1 + sum(tML./max(xML,realmin)))/N);
        else
            tBC = tML*N/(N-1);
        end
        tau(3) = tBC; X(1:Jr,3) = max(s(1:Jr) - tBC, 0);
        % Passemier, Li & Yao
        [tau(4), a] = estimate_tau_ply(s,N,Jr); X(1:Jr,4) = a;
        % Donoho, Gavish & Johnstone
        [tau(5), X(1:Jr,5)] = dgj_shrink(s,N,Jr,'entropy',mpMed);
        [tau(6), X(1:Jr,6)] = dgj_shrink(s,N,Jr,'stein',mpMed);
        [tau(7), X(1:Jr,7)] = dgj_shrink(s,N,Jr,'entropy',mpMed,tau(4));
        [tau(8), X(1:Jr,8)] = dgj_shrink(s,N,Jr,'stein',mpMed,tau(4));

        row = zeros(1,5*nm);
        for m = 1:nm
            row(5*(m-1)+(1:5)) = local_metrics(U,K,J,tau(m),X(:,m),Sigma0,tp.sigma2,eta0);
        end
        M(i,:) = row;
    end

    mu = mean(M,1); se = std(M,0,1)/sqrt(niter);
    for m = 1:nm
        idx = 5*(m-1)+(1:5);
        rows(end+1,:) = [{N, snr, J, methods{m}}, num2cell(mu(idx)), ...
            num2cell(se(idx)), {100*mean(reg)}, {100*mean(unres)}]; %#ok<AGROW>
    end
    fprintf('done N=%d SNR=%g J=%d (regular %.1f%%)\n', N, snr, J, 100*mean(reg));
end
end
end

T = cell2table(rows, 'VariableNames', {'N','SNR','J','Method', ...
    'S1','S2','KL','rKL','Seta','se_S1','se_S2','se_KL','se_rKL','se_Seta', ...
    'regular_pct','unresolved_pct'});
end

%% -----------------------------------------------------------------------
function m = local_metrics(U,K,J,tau,x,Sigma0,tau0,eta0)
UJ = U(:,1:J);
Sh = UJ*diag(x)*UJ' + tau*eye(K);
z = zeros(K,1);
e = 0.5*log(tau/tau0);
m = [e, e^2, mvgkl(z,Sigma0,z,Sh), mvgkl(z,Sh,z,Sigma0), ...
     mean(0.25*log((tau + x(:))./eta0).^2)];
end

function [tau,x,Jr,okJ,unres] = local_mml_fallback(s,N,K,J)
% MML87 at rank J if a regular solution exists, otherwise at the largest
% rank J' < J that admits one. UNRES flags a rank whose existence could not
% be determined (it is then treated as having no regular solution).
okJ = false; unres = false;
for Jr = J:-1:1
    [tau,x,info] = solve_tau_mml(s,N,Jr);
    unres = unres || strcmp(info.status,'unresolved');
    if info.ok
        okJ = (Jr == J);
        x = x(:);
        return
    end
end
Jr = 0;
tau = N*sum(s)/(K*(N-1));
x = zeros(0,1);
end
