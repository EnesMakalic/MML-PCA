function k = signflippa(X, T, cutoff)
    arguments
        X (:,:) double {mustBeNumeric,mustBeReal}
        T (1,1) double {mustBePositive,mustBeInteger} = 10
        cutoff (1,1) double {mustBePositive,mustBeReal,mustBeInRange(cutoff,50,100)} = 95
    end    

% Compute data singular values
[n,p] = size(X);
S = svd(X);

% permutations
Snull = zeros(T, p);
for t = 1:T
    % flip signs of X at random
    R = ones(n,p);
    R(rand(n*p,1) < 0.5) = -1; 
    Snull(t,:) = svd(X .* R)';    
end

% Determine k by computing how many leading data singular values are
% greater than the cutoff percentiles of the null distribution singular
% values
k = 0;
for i = 1:p
    alpha = prctile(Snull(:,i), cutoff);

    if(S(i) > alpha)
        k = k + 1;
    else
        break;
    end
end

end