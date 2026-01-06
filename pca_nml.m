function [sc, pc, negll] = pca_nml(x, m, ~, delta, Sigma_ml, s2_ml, stype)

[N,K] = size(x);

%% Determine s
if(nargin < 7)
    stype = 0;
end

prec = 15;
xs = round( bsxfun(@times, x, 10^(prec)) );
Sint = (xs'*xs) / N;
switch stype
    case 0
        s = 64;
    case 1
        s = ceil( (log2(trace(Sint)) - log2(K))/2 );
    case 2
        [~,e,~] = svd(Sint); 
        s = ceil( log2( max(diag(e)) ) / 2 );
    otherwise
        error('invalid s estimate');
end

%% Compute stochastic complexity
% Negative log-likelihood
negll = -sum(mvnormpdfln(x', zeros(1,K)', [], Sigma_ml));

% Parametric complexity
%logIs = s*log(2)*m*(m-1)/2 + (m*log(s*log(2)) + m*(m-1)/4*log(2));
logIs = m*log(s*log(2)) + (2*s + 1)*m*(m-1)/4*log(2);
logVol = -(3/2)*m*log(2) - gammaln(m+1) + m*log(2) + m*(m+1)/4*log(pi) - sum( gammaln((1:m)/2) ) + logIs;
pc = m*(m+1)/4*log(N/2/pi) + logVol - (m+2)*m*(m-1)/(24*N);

% Stochastic complexity
sc = pc + negll;

end
