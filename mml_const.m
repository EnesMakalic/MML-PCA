%% MML_CONST    Computes the WF87 quantisation constant.
% 
% Computes (k/2) (log(kappa_k) + 1), where kappa_k is the mean squared
% quantisation error of an optimal quantising lattice in k
% dimensions.
% 
%  The input arguments are: 
%   k    - number of free parameters (k > 0)
% 
%  Returns:
%   f    - WF87 quantisation constant for dimension k (>0)
%
% References:
% [1] Erik Agrell and Thomas Eriksson. 
% “Optimization of Lattices for Quantization.” IEEE Trans. Inf. Theory 44 (1998): 1814-1828.
%
% [2] Erik Agrell and Bruce Allen
% "On the best lattice quantizers",(2022) arXiv:2202.09605 [cs.IT]
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2019-
function f = mml_const(k)

% MSE
kappa_k = [1 / 12               % k = 1 [Z lattice]
     5/(36*sqrt(3))             % k = 2 [A2 lattice]
     19/(192*2^(1/3))           % k = 3 [A*3 lattice]
     13/120/sqrt(2)             % k = 4 [D4 lattice]
     2641/(23040*2^(3/5))       % k = 5 [D*5 lattice]
     12619/(68040*3^(5/6))      % k = 6 [E*6 lattice]
     21361/(161280*2^(6/7))     % k = 7 [E*7 lattice]
     929/12960                  % k = 8 [E8 lattice]
     0.071622594                % k = 9
     0.070813818                % k = 10
     0.070426259                % k = 11
     0.070095600                % k = 12
     0.071034583                % k = 13
     0.071455542                % k = 14
     0.071709124                % k = 15
     0.06830                    % k = 16
    ];

% WF87 constant
if(k <= 0)
    f = 0;  
elseif(k <= 16)
    f = (k/2)*(log(kappa_k(k)) + 1);
else
    f = -k*log(2*pi)/2 + log(k*pi)/2 + psi(1);
end

end