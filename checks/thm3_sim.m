function T = thm3_sim(R, Ns, cfg, seed0, direct)
%THM3_SIM Monte Carlo check of Theorem 3 (proportional regime, K = gamma*N)
% for MML87 and ML, and of the limiting mean squared errors quoted after it.
% Reports the mean, variance, skewness and excess kurtosis of N(tau_hat -
% tau)/tau against the limits (tau*beta_gamma for MML87, -tau(1+J+sum 1/rho)
% for ML, variance 2/gamma), the scaled mean squared error against 2/gamma +
% centring^2, and the mean and variance of sqrt(N)(x_hat_j - tau z_j)/tau
% against 0 and z_j^2 v_j/xi_j^2. Mean estimated: N*S ~ W_K(N-1, Sigma).
% MML87 fits without a regular solution are completed by ML at the same J.
%
% Sampler: only tr(S) and the top J eigenvalues of S are needed. With
% Sigma = tau(I + sum rho_j e_j e_j'), write W = H (B B') H' for the
% Wishart W_K(N-1, I), with B the Dumitriu-Edelman lower bidiagonal matrix
% and H Haar orthogonal, independent of B. The eigenvalues of N*S/tau are
% those of B'B + U U', U = B' H_J diag(sqrt(rho)), where H_J (first J rows
% of H, transposed) is a uniformly distributed K x J orthonormal frame. The
% top J eigenvalues are found by Lanczos (eigs) with O(K) products.
% DIRECT = true instead draws S by the Bartlett decomposition and uses eig
% (for validating the sampler at moderate K).
% CFG rows are {gamma, rho, tau}; N must make gamma*N an integer. SEED0 is
% the base seed (default 4000). The defaults (four designs, N = 200 to 1e5,
% R = 2e4) take about 4 minutes on 14 workers; R = 1e5 takes about 20.
if nargin < 1, R = 2e4; end
if nargin < 2, Ns = [200 1000 5000 20000 100000]; end
if nargin < 3 || isempty(cfg)
    cfg = {0.5, [8 4], 1; 0.25, 5, 1; 0.1, [3 2], 1; 0.75, [10 5], 2};
end
if nargin < 4, seed0 = 4000; end
if nargin < 5, direct = false; end
rows = {}; id = 0;
for ic = 1:size(cfg,1)
    g = cfg{ic,1}; rho = cfg{ic,2}(:); tau = cfg{ic,3}; J = numel(rho);
    [cM, cML, z, vx] = limits(g, rho);
    for N = Ns
        id = id + 1; K = round(g*N); m = K - J; nu = N - 1;
        eT = zeros(R,2); ex = zeros(R,J); reg = false(R,1);
        parfor r = 1:R
            st = RandStream('Threefry','Seed',seed0+id); st.Substream = r;
            RandStream.setGlobalStream(st);
            if direct
                sd = sqrt(tau*[1 + rho; ones(m,1)]);
                A = tril(randn(K,K),-1);
                A(1:K+1:end) = sqrt(2*randg((nu - (0:K-1))/2));
                L = sd.*A; S = (L*L')/N;
                e = sort(eig((S+S')/2),'descend');
                top = e(1:J); trS = sum(e);
            else
                [top, trS] = topeig(K, nu, rho, J);
                top = tau*top/N; trS = tau*trS/N;
            end
            rest = (trS - sum(top))/m;
            s = [top; rest*ones(m,1)];
            tML = rest;
            [t,x,info] = solve_tau_mml(s,N,J);
            if info.ok, reg(r) = true; else, t = tML; x = top - tML; end
            eT(r,:) = N*([t tML] - tau)/tau;
            ex(r,:) = sqrt(N)*(x(:)' - tau*z')/tau;
        end
        mom = @(v) [mean(v), var(v), skewness(v), kurtosis(v) - 3, mean(v.^2)];
        a = mom(eT(:,1)); b = mom(eT(:,2));
        se = @(v) std(v)/sqrt(R);
        rows(end+1,:) = {g, mat2str(rho'), tau, N, K, 100*mean(reg), ...
            cM, a(1), se(eT(:,1)), 2/g, a(2), a(3), a(4), 2/g + cM^2, a(5), ...
            cML, b(1), se(eT(:,2)), b(2), b(3), b(4), 2/g + cML^2, b(5), ...
            mat2str(round(mean(ex,1),3)), mat2str(round(vx',2)), mat2str(round(var(ex,0,1),2))}; %#ok<AGROW>
        fprintf(['g=%.2f rho=%-7s tau=%g N=%6d reg=%6.2f%% | MML mean %6.3f(%.3f) lim %6.3f var %5.2f lim %4.2f ' ...
                 'skew %5.2f exkurt %5.2f MSE %5.2f lim %5.2f | ML mean %6.3f(%.3f) lim %6.3f var %5.2f MSE %6.2f lim %6.2f' ...
                 ' | x mean %s var %s lim %s\n'], ...
            g, mat2str(rho'), tau, N, 100*mean(reg), a(1), se(eT(:,1)), cM, a(2), 2/g, a(3), a(4), a(5), 2/g + cM^2, ...
            b(1), se(eT(:,2)), cML, b(2), b(5), 2/g + cML^2, ...
            mat2str(round(mean(ex,1),3)), mat2str(round(var(ex,0,1),2)), mat2str(round(vx',2)));
    end
end
T = cell2table(rows,'VariableNames',{'gamma','rho','tau','N','K','regular_pct', ...
    'MML_mean_limit','MML_mean','MML_mean_se','var_limit','MML_var','MML_skew','MML_exkurt','MML_mse_limit','MML_mse', ...
    'ML_mean_limit','ML_mean','ML_mean_se','ML_var','ML_skew','ML_exkurt','ML_mse_limit','ML_mse', ...
    'x_mean','x_var_limit','x_var'});
end

function [top, trS] = topeig(K, nu, rho, J)
% Top J eigenvalues and trace of N*S/tau, N*S ~ W_K(nu, I + sum rho_j e_j e_j').
d = sqrt(2*randg((nu - (0:K-1)')/2));          % chi_{nu}, ..., chi_{nu-K+1}
o = sqrt(2*randg((K-1:-1:1)'/2));              % chi_{K-1}, ..., chi_1
B = spdiags([[o; 0] d], [-1 0], K, K);         % lower bidiagonal
[Q,~] = qr(randn(K,J),0);                      % uniform orthonormal K x J frame
U = (B'*Q).*sqrt(rho');
afun = @(v) B'*(B*v) + U*(U'*v);
opts = struct('issym', true, 'tol', 1e-13, 'maxit', 2000, 'p', min(K, max(2*J+20, 30)));
top = sort(eigs(afun, K, J, 'largestreal', opts), 'descend');
trS = sum(d.^2) + sum(o.^2) + sum(U(:).^2);
end

function [cM, cML, z, vx] = limits(g, rho)
% Theorem 3 limits for tau = 1.
b  = rho - g + g./rho;
xi = sqrt(b.^2 - 4*g*(1+g));
z  = (b + xi)/(2*(1+g));
cM  = sum(1./z - 1./rho);
cML = -(1 + numel(rho) + sum(1./rho));
v  = 2*(1+rho).^2.*(1 - g./rho.^2);
vx = z.^2.*v./xi.^2;
end
