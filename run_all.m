%RUN_ALL Reproduce all simulation results of the revised paper.
%
% Runs, with 1e5 replications each:
%   1. sim_compare, equal loadings      -> results/final_compare_equal.csv
%      (main Table 1; supplementary Tables S1-S3)
%   2. sim_compare, geometric loadings  -> results/final_compare_geometric.csv
%      (supplementary Tables S4-S6)
%   3. sim_compare, high-dimensional    -> results/final_compare_highdim.csv
%      (supplementary Tables S7-S9)
%   4. sim_model_sel2, equal loadings   -> results/final_modelsel.csv
%      (main Tables 2-3)
% The LaTeX tables are then produced by scripts/make_tables.py.
%
% Requirements: MATLAB with the Parallel Computing Toolbox and a C compiler
% for MEX, and two toolboxes by Tom Minka that are not distributed here:
%   - the PCA toolbox (laplace_pca), https://tminka.github.io/papers/pca/
%   - lightspeed (mvnormpdfln, used by pca_nml), https://github.com/tminka/lightspeed
% Put them in external/Laplace_PCA and external/lightspeed next to this file,
% or add them to the END of the MATLAB path yourself; lightspeed's normcdf.m
% would otherwise shadow MATLAB's. Lightspeed's solve_tril is compiled into
% build/ if no compiled version is found.
%
% Running everything took about 55 minutes on 12 workers.

here = fileparts(mfilename('fullpath'));
res  = fullfile(here,'results');
bld  = fullfile(here,'build');
if ~exist(res,'dir'), mkdir(res); end
addpath(here);

for d = {fullfile(here,'external'), fullfile(here,'..','External-code')}
    if exist(fullfile(d{1},'Laplace_PCA'),'dir'), addpath(fullfile(d{1},'Laplace_PCA'),'-end'); end
    if exist(fullfile(d{1},'lightspeed'),'dir'),  addpath(fullfile(d{1},'lightspeed'),'-end');  end
end
if exist('laplace_pca','file') ~= 2
    error('run_all:external','laplace_pca not found: install Minka''s PCA toolbox (see the header of run_all.m).');
end
if exist('mvnormpdfln','file') ~= 2
    error('run_all:external','mvnormpdfln not found: install Minka''s lightspeed toolbox (see the header of run_all.m).');
end

% Compile lightspeed's solve_tril (needed by pca_nml via mvnormpdfln).
if ~exist(bld,'dir'), mkdir(bld); end
addpath(bld);
if exist('solve_tril') ~= 3 %#ok<EXIST>   % 3 = compiled MEX file on the path
    lsdir = fileparts(which('mvnormpdfln'));
    mex('-outdir', bld, '-DBLAS64', fullfile(lsdir,'solve_tril.c'), '-lmwblas');
    rehash;
end

R = 1e5;
if isempty(gcp('nocreate')), parpool; end
t0 = tic;

T = sim_compare(R,'Design','equal');
writetable(T, fullfile(res,'final_compare_equal.csv'));
fprintf('[%.0fs] estimator comparison, equal loadings\n', toc(t0));

T = sim_compare(R,'Design','geometric');
writetable(T, fullfile(res,'final_compare_geometric.csv'));
fprintf('[%.0fs] estimator comparison, geometric loadings\n', toc(t0));

H = [];
for K = [20 50 100]
    Js = unique([1 ceil(K/10) ceil(K/4)]);
    for g = [10 2]                       % N = 10K (gamma = 0.1) and N = 2K (gamma = 0.5)
        N = g*K;
        if g == 10, snr = [5 20]; else, snr = [10 20]; end
        Ti = sim_compare(R,'K',K,'N',N,'SNR',snr,'J',Js,'Design','equal','Seed',K*10+g);
        Ti.K = repmat(K,height(Ti),1);
        H = [H; Ti]; %#ok<AGROW>
        fprintf('[%.0fs] high-dimensional comparison, K=%d N=%d\n', toc(t0), K, N);
    end
end
writetable(H, fullfile(res,'final_compare_highdim.csv'));

T = sim_model_sel2(R,'Design','equal','Seed',11);
writetable(T, fullfile(res,'final_modelsel.csv'));
fprintf('[%.0fs] model selection\n', toc(t0));

fprintf('Done. Now run: python3 %s\n', fullfile(here,'scripts','make_tables.py'));
