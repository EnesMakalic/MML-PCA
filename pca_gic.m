function [rgic, negll, dof] = pca_gic(lam, s2, r, n)

p = length(lam);

%% calculate DOF
sumterm = 0;
for j = 1:r
    for l = (r+1):p
        sumterm = sumterm + (lam(l) * (lam(j) - s2))/(s2*(lam(j) - lam(l)));
    end
end

ix = (r+1):p;
if(r < 2)
    dof = r + p + sum(lam(ix).^2)/(p-r) / ( sum(lam(ix))/(p-r) )^2 + sumterm;
else
    dof = nchoosek(r,2) + r + p + sum(lam(ix).^2)/(p-r) / ( sum(lam(ix))/(p-r) )^2 + sumterm;
end

%% negative log-likelihood
negll = (p-r)*log(s2) + sum(log(lam(1:r)));

%% GIC for this r
rgic = negll + (2/n)*dof;

end
