function [tau_ml, alpha_ml, Sigma_ml] = estimate_tau_mle(eigenvalues, eigenvecs, J)
% ESTIMATE_TAU_MLE Compute maximum likelihood estimate of residual variance
%
% Description:
%   Computes the maximum likelihood (ML) estimate of the residual variance
%   tau = sigma^2 for the probabilistic PCA model as described in equation (9)
%   of Makalic & Schmidt "MML Probabilistic Principal Component Analysis".
%
%   The ML estimate is the average of the (K-J) smallest eigenvalues:
%   tau_ML = (1/(K-J)) * sum(delta_{J+1}, ..., delta_K)
%
% Syntax:
%   tau_ML = estimate_tau_mle(eigenvalues)
%
% Inputs:
%   eigenvalues - K×1 vector of sample covariance eigenvalues (sorted descending)
%   eigenvecs   - KxJ matrix of eigenvectors of the sample covariance
%   J           - number of true latent factors
%
% Outputs:
%   tau_ml   - Maximum likelihood estimate of residual variance sigma^2
%   alpha_ml - Maximum likelihood estimate of alpha
%   Sigma_ml - Maximum likelihood estimate of variance-covariance matrix Sigma
%
% Examples:
%   tau_ML = estimate_tau_mle(eigs, 2);
%
% Reference:
% E. Makalic and Daniel F. Schmidt, MML Probabilistic Principal Component
% Analysis, arXiv:2209.14559 [stat.ME], 2026.
arguments
    eigenvalues (:,:) {mustBeNumeric}
    eigenvecs (:,:) {mustBeNumeric}
    J (1,1) {mustBePositive, mustBeInteger}    
end

delta = eigenvalues(:); % Ensure column vector
K = length(delta);

% MLE estimate of tau
tau_ml = mean(delta(J+1:end));   

% MLE estimates of alpha and Sigma
R = eigenvecs(:,1:J);
alpha_ml = sqrt(max(0, delta(1:J) - tau_ml));
A = R * diag(alpha_ml);
Sigma_ml = A*A' + tau_ml*eye(K);

end
