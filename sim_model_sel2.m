function T = sim_model_sel2(niter, varargin)
%SIM_MODEL_SEL2 Latent-dimension selection experiment for the revised MML87.
%
% T = SIM_MODEL_SEL2(NITER) compares rank selection by
%   MML      MML87 (mml_ppca_fp), using its own MML87 estimates
%   MML-ML   the MML87 rank with maximum likelihood estimates
%   BIC      Bayesian information criterion for probabilistic PCA; the
%            same criterion as Minka's bic_pca, but with J = 0 allowed
%   Bayes    Laplace approximation to the evidence (Minka 2000, laplace_pca)
%   GIC      generalized information criterion (pca_gic)
%   NML      normalized maximum likelihood (pca_nml)
%   SPA      signflip parallel analysis (signflippa)
% over candidate ranks 0..Jmax. All methods other than MML use maximum
% likelihood estimates at their selected rank. Bayes and NML require the
% External-code folders (Laplace_PCA, lightspeed) on the MATLAB path.
%
% Design: per-component signal-to-noise ratio (rho_j = lambda_j^2/tau),
% tau = 1, Haar-distributed signal directions, unknown mean (data are
% centred). 'equal' loadings give rho_j = SNR; 'geometric' gives the weakest
% component rho_J = SNR and rho_j = SNR/r^(J-j). J = 0 is pure noise.
%
% Output T has one row per (N, SNR, J, Method) with the percentages of
% under-, exact and over-selection, the mean selected rank, the mean
% forward KL(Sigma0||Sigmahat) and reverse KL(Sigmahat||Sigma0), and (for
% MML) the percentage of data sets in which the selected rank equals the
% largest rank admitting a regular MML87 solution.
%
% Name-value options: 'N' ([25 50 100]), 'SNR' ([2 5 20]), 'J' ([0 1 2 4]),
% 'K' (10), 'Jmax' (6), 'Design' ('equal'), 'GeometricRatio' (0.75),
% 'Seed' (1).

p = inputParser;
p.addParameter('N',[25 50 100]);
p.addParameter('SNR',[2 5 20]);
p.addParameter('J',[0 1 2 4]);
p.addParameter('K',10);
p.addParameter('Jmax',6);
p.addParameter('Design','equal');
p.addParameter('GeometricRatio',0.75);
p.addParameter('Seed',1);
p.parse(varargin{:});
o = p.Results;
if nargin < 1 || isempty(niter), niter = 2000; end

methods = {'MML','MML-ML','BIC','Bayes','GIC','NML','SPA'};
nm = numel(methods); K = o.K; Jmax = o.Jmax; gr = o.GeometricRatio;
rows = {}; cell_id = 0;
for N = o.N
for J = o.J
    snrList = o.SNR; if J == 0, snrList = NaN; end
    for snr = snrList
    cell_id = cell_id + 1; base = o.Seed*1e6 + cell_id;
    if J == 0
        snrGen = 1;            % unused
    elseif strcmpi(o.Design,'geometric')
        snrGen = snr*sum(gr.^(0:J-1))/(K*gr^(J-1));
    else
        snrGen = snr*J/K;
    end
    Jsel = zeros(niter,nm); KLf = zeros(niter,nm); KLr = zeros(niter,nm);
    thr = false(niter,1); un = false(niter,1);

    parfor i = 1:niter
        stream = RandStream('Threefry','Seed',base); stream.Substream = i;
        [s,U,tp,X] = generate_pca_data(N,K,'sigma2',1.0,'J',J,'SNR',snrGen, ...
            'LoadingDesign',o.Design,'GeometricRatio',gr, ...
            'CenterData',true,'Stream',stream);
        Xc = X - mean(X,1); Sigma0 = tp.Sigma; z = zeros(K,1);

        % MML87
        res = mml_ppca_fp(X,'MaxFactors',Jmax,'Warn',false);
        un(i) = ~isempty(res.unresolved);
        jm = res.J;
        okJ = res.table(~isnan(res.table(:,2)),1);
        thr(i) = (jm == max(okJ));

        % BIC and GIC over 0..Jmax with ML estimates
        bic = zeros(Jmax+1,1); gic = zeros(Jmax+1,1); nml = zeros(Jmax+1,1);
        for r = 0:Jmax
            tauML = mean(s(r+1:K));
            negLL = N/2*(sum(log(s(1:r))) + (K-r)*log(tauML) + K + K*log(2*pi));
            P = K + K*r - r*(r-1)/2 + 1;
            bic(r+1) = negLL + P/2*log(N);
            gic(r+1) = pca_gic(s,tauML,r,N);
            SigML = U(:,1:r)*diag(s(1:r)-tauML)*U(:,1:r)' + tauML*eye(K);
            nml(r+1) = pca_nml(Xc,r,[],s,SigML,tauML,1);
        end
        [~,jb] = min(bic); jb = jb-1;
        [~,jg] = min(gic); jg = jg-1;
        [~,jn] = min(nml); jn = jn-1;
        jl = min(Jmax, laplace_pca(Xc));
        % signflippa draws from the global generator; make it reproducible by
        % continuing this replication's own stream
        RandStream.setGlobalStream(stream);
        js = min(Jmax, signflippa(Xc));

        jj = [jm, jm, jb, jl, jg, jn, js];
        kf = zeros(1,nm); kr = zeros(1,nm);
        for m = 1:nm
            if m == 1
                Sh = res.Omega;
            else
                q = jj(m); tq = mean(s(q+1:K));
                Sh = U(:,1:q)*diag(max(s(1:q)-tq,0))*U(:,1:q)' + tq*eye(K);
            end
            kf(m) = mvgkl(z,Sigma0,z,Sh); kr(m) = mvgkl(z,Sh,z,Sigma0);
        end
        Jsel(i,:) = jj; KLf(i,:) = kf; KLr(i,:) = kr;
    end

    for m = 1:nm
        rows(end+1,:) = {N, snr, J, methods{m}, ...
            100*mean(Jsel(:,m)<J), 100*mean(Jsel(:,m)==J), 100*mean(Jsel(:,m)>J), ...
            mean(Jsel(:,m)), mean(KLf(:,m)), mean(KLr(:,m)), ...
            std(KLf(:,m))/sqrt(niter), std(KLr(:,m))/sqrt(niter), ...
            100*mean(thr), 100*mean(un)}; %#ok<AGROW>
    end
    fprintf('done N=%d J=%d SNR=%g\n',N,J,snr);
    end
end
end
T = cell2table(rows,'VariableNames',{'N','SNR','J','Method','under','exact','over', ...
    'meanJ','KL','rKL','se_KL','se_rKL','mml_is_threshold_pct','unresolved_pct'});
end
