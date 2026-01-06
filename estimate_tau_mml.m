function [tau_mml, alpha_mml, Sigma_mml] = estimate_tau_mml(eigenvalues, eigenvecs, N, J)
%ESTIMATE_TAU_MML  Compute MML estimate of residual variance
%
% Description:
% Implements Theorem 1 (eqs. (46)-(47)) and uses tau_ML from eq. (9).
% delta must be descending eigenvalues: delta(1) > ... > delta(K) > 0.
%
% Syntax:
%   tau_mml = estimate_tau_mml(eigenvalues, N, J);
%
% Inputs:
%   eigenvalues - K×1 vector of sample covariance eigenvalues (sorted descending)
%   eigenvecs   - KxJ matrix of eigenvectors of the sample covariance
%   N           - sample size
%   J           - number of true latent factors
%
% Outputs:
%   tau_mml   - MML estimate of residual variance sigma^2
%   alpha_mml - MML estimate of alpha
%   Sigma_mml - MML estimate of variance-covariance matrix Sigma
%
% Examples:
%   tau_mml = estimate_tau_mle(eigs, 100, 2);
%
% Reference:
% E. Makalic and Daniel F. Schmidt, MML Probabilistic Principal Component
% Analysis, arXiv:2209.14559 [stat.ME], 2026.

delta = eigenvalues(:);
K = numel(delta);

% ML residual variance tau_ML = mean of smallest K-J eigenvalues (eq. (9)) 
tauML = mean(delta(J+1:K));

% Top-J eigenvalues
d = delta(1:J);

% Elementary symmetric polynomials e_0..e_J of d(1:J)
% e(1)=e_0=1, e(2)=e_1, ..., e(J+1)=e_J
e = zeros(J+1,1);
e(1) = 1.0;
for i = 1:J
    for t = i:-1:1
        e(t+1) = e(t+1) + d(i)*e(t);
    end
end

% Safe elementary symmetric polynomial lookup:
% returns e_k for k in [0,J], otherwise 0 (Theorem 1 convention)
function val = eget(k)
    if k < 0 || k > J
        val = 0.0;
    else
        val = e(k+1);
    end
end

% Build coefficients a_m for m=0..J+1 (Theorem 1, eq. (47)) 
a = zeros(J+2,1);  % a(1)=a_0,...,a(J+2)=a_{J+1}
for m = 0:(J+1)
    e1 = eget(J-m);
    e2 = eget(J-m+1);

    % c_m term (from eq. (47))
    c_m = 1 - (K*J - m + 1)/(N*(K-J)) + (m-1)/N;

    % a_m = (-1)^(m+1) [ tauML*e_{J-m} + c_m*e_{J-m+1} ]
    a(m+1) = ((-1)^(m+1)) * (tauML*e1 + c_m*e2);
end

% MATLAB roots expects descending powers
polyDesc = flipud(a);
rts = roots(polyDesc);

% Filter real roots and admissible interval 0 < tau < delta_J (Theorem 1 domain) 
tolImag = 1e-8;
realRts = rts(abs(imag(rts)) < tolImag);
realRts = real(realRts);

deltaJ = d(end);
valid = realRts(realRts > 0 & realRts < deltaJ);

% MML estimate
tau_mml = deltaJ;
if ~isempty(valid)
    tau_mml = min(valid);
end

alpha_mml = sqrt(max(0, delta(1:J) - tau_mml));
R = eigenvecs(:,1:J);
A = R * diag(alpha_mml);    
Sigma_mml = A*A' + tau_mml*eye(K);      

end
