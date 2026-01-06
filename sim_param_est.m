function [AvgPerf, collapsed] = sim_param_est(niter, snr, N, K, J, display)
% SIM_PARAM_EST Finite sample performance of MLE and MML tau estimators.
%
% Description:
%  Code to reproduce results for a specific ML/MML entry in Table I.
%
% Inputs:
%   niter   - Number of simulation iterations (default: 1e4)
%   'SNR'   - Signal-to-noise ratio (default: 8)
%                  SNR = (1/(K*sigma^2)) * sum(alpha_j^2)
%   N       - Number of samples (default: 50)
%   K       - Dimensionality of data 
%   J       - Number of true latent factors
%   display - Do we display the result summary? (default: 1)
%
% Reference:
% E. Makalic and Daniel F. Schmidt, MML Probabilistic Principal Component
% Analysis, arXiv:2209.14559 [stat.ME], 2026.
%
arguments
    niter (1,1) double {mustBePositive,mustBeInteger} = 1e4;
    snr (1,1) double {mustBePositive, mustBeNumeric} = 8;
    N (1,1) double {mustBePositive,mustBeInteger} = 50;
    K (1,1) double {mustBePositive,mustBeInteger} = 5;
    J (1,1) double {mustBePositive,mustBeInteger} = 1;
    display (1,1) double = 1;
end

%% Performance metrics
S1_mml = zeros(niter,1); S2_mml = zeros(niter,1); 
S1_ml = zeros(niter,1); S2_ml = zeros(niter,1);
kl_ml = zeros(niter,1);
kl_mml = zeros(niter,1);

%% Run experiment niter times
collapsed = 0;
tStart = tic;   
parfor i = 1:niter

    %% Experiment setup
    [delta, U, true_params, x] = generate_pca_data(N, K, ...
        'sigma2', 1.0, 'J', J, 'SNR', snr);
   
    %% ML estimates        
    [s2_ml, ~, Sigma_ml] = estimate_tau_mle(delta, U, J);

    %% MML Estimates
    [s2_mml, alpha_mml, Sigma_mml] = estimate_tau_mml(delta, U, N, J);

    % MML will sometimes zero-out a factor. 
    if(any(alpha_mml < 1e-5))    
        collapsed = collapsed + 1;     % Collapse J and retry.  
        [s2_mml,Sigma_mml,~,s2_ml,Sigma_ml] = estimateCollapsed(U,x,N,K,J,delta);
    end

    %% Performance metrics    
    s2 = true_params.sigma2;
    Sigma = true_params.Sigma;

    % S1, S2
    S1_mml(i) = log(s2_mml/s2)/2; 
    S2_mml(i) = (log(s2_mml/s2)/2)^2;
    S1_ml(i) = log(s2_ml/s2)/2; 
    S2_ml(i) = (log(s2_ml/s2)/2)^2;

    % KL divergence
    kl_ml(i)  = mvgkl(zeros(K,1), Sigma, zeros(K,1), Sigma_ml);
    kl_mml(i) = mvgkl(zeros(K,1), Sigma, zeros(K,1), Sigma_mml);
end
tElapsed = toc(tStart);

%% compute averages of all performance metrics
AvgPerf = [ ...
    mean([S1_ml, S1_mml]); ...
    mean([S2_ml, S2_mml]); ...
    mean([kl_ml, kl_mml])
];

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

    fprintf('MML collapsed a factor in %4.2f%% of iterations. \n\n', collapsed/niter*100);
       
    tb = table(AvgPerf(:,1),AvgPerf(:,2), ...
        'variablenames',{'MLE','MML'},'RowNames',{'S1','S2','KL'});
    disp(tb);
end

end

% Helper function for parameter simulation experiments
% Factor length estimated as 0. Reduce J and re-try
function [s2_mml,Sigma_mml,J,s2_ml,Sigma_ml] = estimateCollapsed(U,x,N,K,Jmax,delta)

for J = (Jmax-1):-1:0
    if(J > 0)    
        %% MML Estimates
        [s2_mml, alpha_mml, Sigma_mml] = estimate_tau_mml(delta, U, N, J);
         
        %% ML estimates        
        [s2_ml, ~, Sigma_ml] = estimate_tau_mle(delta, U, J);      

    else % J == 0
        % MML
        s2_mml = var(x(:));        
        Sigma_mml = s2_mml*eye(K);

        % MLE
        s2_ml = var(x(:),1);
        Sigma_ml = s2_ml*eye(K);        
    end
 
    if(~any(alpha_mml < 1e-5)) 
        break;
    end
end

end
