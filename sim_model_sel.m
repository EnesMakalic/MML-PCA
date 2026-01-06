function [S1mean, S2mean, KLmean, model_mml, model_bic, model_ard, model_spa, model_gic, model_nml] ...
    = sim_model_sel(niter, snr, N, K, Jtrue, Jmax, display)
% SIM_MODEL_SEL Finite sample performance of MLE and MML tau estimators.
%
% Description:
% Code to reproduce results for a specific entry in Table II/III.
%
% IMPORTANT: To run this code you must have Minka's PCA toolbox installed
% and in MATLAB path. See: https://tminka.github.io/papers/pca/
%
% Inputs:
%   niter   - Number of simulation iterations (default: 1e4)
%   'SNR'   - Signal-to-noise ratio (default: 8)
%                  SNR = (1/(K*sigma^2)) * sum(alpha_j^2)
%   N       - Number of samples (default: 50)
%   K       - Dimensionality of data 
%   Jtrue   - Number of true latent factors
%   Jmax    - Maximum number of latent factors to estimate
%   display - Do we display the result summary? (default: 1)
%
% Reference:
% E. Makalic and Daniel F. Schmidt, MML Probabilistic Principal Component
% Analysis, arXiv:2209.14559 [stat.ME], 2026.
%
arguments
    niter (1,1) double {mustBePositive,mustBeInteger} = 1e3;
    snr (1,1) double {mustBePositive, mustBeNumeric} = 8;
    N (1,1) double {mustBePositive,mustBeInteger} = 50;
    K (1,1) double {mustBePositive,mustBeInteger} = 6;
    Jtrue (1,1) double {mustBePositive,mustBeInteger} = 2;
    Jmax (1,1) double {mustBePositive,mustBeInteger} = 3;
    display (1,1) double = 1;    
end

%% Performance metrics
% MML
model_mml = zeros(niter, 1); 
S1_mml = zeros(niter,1); 
S2_mml = zeros(niter,1);
kl_mml = zeros(niter,1);

% BIC
model_bic = zeros(niter, 1); 
S1_bic = zeros(niter,1); 
S2_bic = zeros(niter,1);
kl_bic = zeros(niter,1);

% GIC
model_gic = zeros(niter, 1); 
S1_gic = zeros(niter,1); 
S2_gic = zeros(niter,1);
kl_gic = zeros(niter,1);

% NML
model_nml = zeros(niter, 1); 
S1_nml = zeros(niter,1); 
S2_nml = zeros(niter,1);
kl_nml = zeros(niter,1);

% Bayes [to run this, must have Minka's PCA toolbox installed and in path]
model_ard = zeros(niter, 1); 
S1_ard = zeros(niter,1); 
S2_ard = zeros(niter,1);
kl_ard = zeros(niter,1);

% SPA
model_spa = zeros(niter, 1); 
S1_spa = zeros(niter,1); 
S2_spa = zeros(niter,1);
kl_spa = zeros(niter,1);

%% Experiment
tStart = tic;  
for i = 1:niter

    %% Experiment setup
    [delta, U, true_params, x] = generate_pca_data(N, K, ...
        'sigma2', 1.0, 'J', Jtrue, 'SNR', snr);

    %% For all J = 1...Jmax
    msglen = zeros(Jmax, 3); rgic = zeros(Jmax,1); rnml = zeros(Jmax,1);
    S1 = zeros(Jmax,1); S1ml = zeros(Jmax,1); klml = zeros(Jmax,1);
    S2 = zeros(Jmax,1); S2ml = zeros(Jmax,1); klmml = zeros(Jmax,1);
    for J = 1:Jmax                  

        %% ML estimates        
        [s2_ml, ~, Sigma_ml] = estimate_tau_mle(delta, U, J);

        %% MML Estimates
        [s2_mml, alpha_mml, Sigma_mml] = estimate_tau_mml(delta, U, N, J);

        % GIC
        rgic(J) = pca_gic(delta, s2_ml, J, N);

        % NML
        Sx = (x'*x)/N;
        rnml(J) = pca_nml(x, J, Sx, delta, Sigma_ml, s2_ml, 1);

        %% Estimation performance
        Sigma = true_params.Sigma;

        % S1, S2 for MML/MLE
        S1ml(J) = log(s2_ml)/2; S2ml(J) = (log(s2_ml)/2)^2;
        S1(J) = log(s2_mml)/2; S2(J) = (log(s2_mml)/2)^2;

        % KL for MML/MLE        
        klml(J) = mvgkl(zeros(K,1), Sigma, zeros(K,1), Sigma_ml);
        klmml(J) = mvgkl(zeros(K,1), Sigma, zeros(K,1), Sigma_mml);

        if(min(alpha_mml) > 0.1)
            [msglen(J,1), msglen(J,2), msglen(J,3)] = mml_pca_codelength(N, K, J, s2_mml, delta);
        else
            msglen(J,:) = inf(1,3);
        end
    end    
    
    %% Statistics for selected models
    [~, ind] = min(msglen(:,1));
    model_mml(i) = ind;
    S1_mml(i) = S1(ind); 
    S2_mml(i) = S2(ind);
    kl_mml(i) = klmml(ind);

    %% GIC
    [~, ind] = min(rgic(:,1));
    model_gic(i) = ind;
    S1_gic(i) = S1(ind); 
    S2_gic(i) = S2(ind);
    kl_gic(i) = klml(ind);

    %% BIC model
    [k,~,~] = bic_pca(x);
    k = min(Jmax,k);
    model_bic(i) = k;
    S1_bic(i) = S1ml(k);
    S2_bic(i) = S2ml(k);
    kl_bic(i) = klml(k);

    %% Bayes model
    k = laplace_pca(x);
    k = min(Jmax,k);
    k = max(1, k);
    model_ard(i) = k;
    S1_ard(i) = S1ml(k);
    S2_ard(i) = S2ml(k);
    kl_ard(i) = klml(k);    

    %% SignflipPA
    k = signflippa(x);
    k = min(Jmax,k);
    k = max(1, k);
    model_spa(i) = k;
    S1_spa(i) = S1ml(k);
    S2_spa(i) = S2ml(k);
    kl_spa(i) = klml(k);    

    %% NML
    [~, ind] = min(rnml(:,1));
    model_nml(i) = ind;
    S1_nml(i) = S1(ind); 
    S2_nml(i) = S2(ind);
    kl_nml(i) = klml(ind);

end
tElapsed = toc(tStart);

S1mean =  mean([S1_mml S1_bic S1_ard S1_spa S1_gic S1_nml]);
S2mean = mean([S2_mml S2_bic S2_ard S2_spa S2_gic S2_nml]);
KLmean =  mean([kl_mml kl_bic kl_ard kl_spa kl_gic kl_nml]);
Results = [S1mean; S2mean; KLmean];

if(display)
    % Convert elapsed seconds
    days    = floor(tElapsed / 86400);
    hours   = floor(mod(tElapsed, 86400) / 3600);
    minutes = floor(mod(tElapsed, 3600) / 60);
    seconds = mod(tElapsed, 60);
    parts = {};   
    if days >= 1
        parts{end+1} = sprintf('%d days', days);
    end
    if hours >= 1
        parts{end+1} = sprintf('%d hours', hours);
    end
    if minutes >= 1
        parts{end+1} = sprintf('%d minutes', minutes);
    end    
    % Seconds always printed
    parts{end+1} = sprintf('%.3f seconds', seconds);    
    % Join and print
    fprintf('Elapsed time: %s\n', strjoin(parts, ', '));

    tb1 = table(Results(:,1),Results(:,2),Results(:,3),Results(:,4),Results(:,5),Results(:,6), ...
         'variablenames',{'MML','BIC','BAYES','SPA','GIC','NML'},'RowNames',{'S1','S2','KL'});
        
    tb2 = table(mean([model_mml < Jtrue,model_mml == Jtrue,model_mml > Jtrue])'*100, ...
        mean([model_bic < Jtrue,model_bic == Jtrue,model_bic > Jtrue])'*100,...
        mean([model_ard < Jtrue,model_ard == Jtrue,model_ard > Jtrue])'*100,...        
        mean([model_spa < Jtrue,model_spa == Jtrue,model_spa > Jtrue])'*100,...
        mean([model_gic < Jtrue,model_gic == Jtrue,model_gic > Jtrue])'*100,...
        'variablenames',{'MML','BIC','BAYES','SPA','GIC'}, ...
        'RowNames',{'<J','=J','>J'});

    fprintf('MML vs MLE parameter estimation performance for selected models:\n');
    disp(tb1);
    fprintf('Model selection percentages:\n');
    disp(tb2);
end

end
