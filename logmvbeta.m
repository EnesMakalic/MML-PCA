%LOGMVBETA    Log of the multivariate beta function.
%   LOGMVBETA(p,a,b) computes log( Beta_p( a, b ) ).
%
%  The input arguments are:
%   p    - [1 x 1] 
%   a,b  - [1 x 1] 
%
%  Returns:
%   f    - [1 x 1] log( Gamma_p( a, b ) )
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2022-
function f = logmvbeta(p, a, b)

f = logmvgamma(p, a) + logmvgamma(p, b) - logmvgamma(p, a + b);

end