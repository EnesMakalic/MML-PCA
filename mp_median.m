function m = mp_median(gamma)
%MP_MEDIAN Median of the Marchenko-Pastur law with ratio 0 < gamma <= 1.
%
% M = MP_MEDIAN(GAMMA) returns the median of the Marchenko-Pastur
% distribution with unit scale, density
%   f(x) = sqrt((b-x)(x-a)) / (2 pi gamma x),  a = (1-sqrt(gamma))^2,
%   b = (1+sqrt(gamma))^2.

if ~(gamma > 0 && gamma <= 1)
    error('mp_median:gamma','Require 0 < gamma <= 1.');
end
a = (1 - sqrt(gamma))^2;
b = (1 + sqrt(gamma))^2;
f = @(x) sqrt(max((b - x).*(x - a), 0))./(2*pi*gamma*x);
F = @(t) integral(f, a, t, 'AbsTol', 1e-12, 'RelTol', 1e-10);
m = fzero(@(t) F(t) - 0.5, [a + eps, b]);
end
