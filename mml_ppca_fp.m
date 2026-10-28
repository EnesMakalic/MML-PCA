function res = mml_ppca_fp(X, varargin)
%MML_PPCA_FP Minimum message length probabilistic PCA (fixed-point solver).
%
% RES = MML_PPCA_FP(X) fits the isotropic-noise PPCA model
%
%   x_i = mu + A*v_i + eps_i,
%   v_i ~ N(0,I_J),  eps_i ~ N(0,tau*I_K),
%
% for J = 0,...,Jmax and returns the regular model with the shortest
% Wallace-Freeman MML87 codelength.
%
% RES = MML_PPCA_FP(X,'NumFactors',J) fits a specified rank J.
%
% This is the reference implementation of Algorithm 1 of the paper. For
% each J > 0, the residual variance tau is computed by the damped
% fixed-point iteration
%
%   x_j^(t+1)   = xhat_j(tau^(t)),                      j = 1,...,J,
%   tau^(t+1)   = (1-w)*tau^(t) + w*g_J(tau^(t)),
%   g_J(tau)    = [B + C*tau^2*sum_j 1/xhat_j(tau)]/A,
%
% started at tau^(0) = min{tauML, tauMax/2}, where xhat_j(tau) is the
% larger root of the loading quadratic
%
%   (N+K-em)x^2 + {(N+2K-J+1-em)tau - N*delta_j}x + (K-J+1)tau^2 = 0,
%
% em = 1 when the mean is estimated and em = 0 otherwise, and
% A = N(K-J)-JK-em(K-J), B = N*sum_{k>J} delta_k, C = K-J+1. The regular
% loading profile exists on the admissible interval (0, tauMax), with
%
%   tauMax = N*delta_J / {(N+2K-J+1-em) + 2*sqrt((N+K-em)(K-J+1))}.
%
% If an update leaves the admissible interval, the damping parameter w is
% halved and the update repeated. The iteration stops when the undamped
% residual satisfies |g_J(tau) - tau| <= Tol*tau, which makes the result
% scale equivariant. If the iteration fails, existence is decided by the
% concave scaled score H(t) = 2 t^2 dI/dt: a regular interior minimum exists
% if and only if max H > 0, and it is then unique (see SOLVE_TAU_MML). A rank
% is reported as having no regular solution only when this test proves it.
% For J = 0 the estimate is tau = B/A in closed form.
%
% Name-value options
%   'NumFactors'    scalar J, or [] to scan 0:Jmax            (default [])
%   'MaxFactors'    Jmax when scanning                        (default K-1)
%   'EstimateMean'  treat mu as a free parameter              (default true)
%   'SignQuotient'  quotient independent column signs         (default true)
%   'Tol'           relative convergence tolerance            (default 1e-12)
%   'MaxIter'       fixed-point iteration cap                 (default 10000)
%   'Damping'       initial damping w in (0,1]                (default 0.5)
%   'MinDamping'    smallest damping before giving up         (default 1e-8)
%   'Verbose'       print the per-J table                     (default false)
%   'Warn'          issue diagnostic warnings                 (default true)
%
% Output RES has the same fields as MML_PPCA: J, mu, tau, alpha, b, P, A,
% B, Omega, codelength, converged, methodUsed, tauRatio, table, collapsed,
% warnings and diagnostics, plus RES.unresolved (ranks whose existence could
% not be determined numerically). The columns of RES.table are
% [J, codelength, tau, iterations, tau/tauMax, status], where status is
% 1 = fixed point, 2 = existence-certified fallback, 0 = no regular solution
% (proved), -1 = unresolved numerical failure.

%% Options
p = inputParser;
p.addParameter('NumFactors',   [],    @(v) isempty(v) || (isscalar(v) && v >= 0));
p.addParameter('MaxFactors',   [],    @(v) isempty(v) || (isscalar(v) && v >= 0));
p.addParameter('EstimateMean', true,  @islogical);
p.addParameter('SignQuotient', true,  @islogical);
p.addParameter('Tol',          1e-12, @(v) isscalar(v) && v > 0);
p.addParameter('MaxIter',      10000, @(v) isscalar(v) && v >= 1);
p.addParameter('Damping',      0.5,   @(v) isscalar(v) && v > 0 && v <= 1);
p.addParameter('MinDamping',   1e-8,  @(v) isscalar(v) && v > 0 && v <= 1);
p.addParameter('Verbose',      false, @islogical);
p.addParameter('Warn',         true,  @islogical);
p.parse(varargin{:});
o = p.Results;

[N,K] = size(X);
if N < 2, error('mml_ppca_fp:data','Need at least two observations.'); end

%% Centre and eigendecompose once
if o.EstimateMean
    mu = mean(X,1);
    Xc = bsxfun(@minus,X,mu);
else
    mu = [];
    Xc = X;
end
em = double(o.EstimateMean);

S = (Xc'*Xc)/N;
S = (S+S')/2;
[U,Dg] = eig(S);
[delta,ord] = sort(real(diag(Dg)),'descend');
U = U(:,ord);
delta = max(delta,0);

%% Candidate ranks
if isempty(o.MaxFactors)
    Jmax = K-1;
else
    Jmax = min(floor(o.MaxFactors),K-1);
end
if isempty(o.NumFactors)
    Jlist = 0:Jmax;
else
    Jlist = floor(o.NumFactors);
    if Jlist > K-1
        error('mml_ppca_fp:rank','NumFactors must be at most K-1.');
    end
end

% Columns: [J,codelength,tau,iterations,tau/tauMax,status]
tbl = nan(numel(Jlist),6);
statusCode = containers.Map({'fixedpoint','fallback','none','tangent','unresolved'}, {1,2,0,0,-1});
best = struct('codelength',inf);

for ii = 1:numel(Jlist)
    J = Jlist(ii);
    [tau,x,sinfo] = solve_tau_mml(delta,N,J,'EstimateMean',o.EstimateMean, ...
        'Tol',o.Tol,'MaxIter',o.MaxIter,'Damping',o.Damping,'MinDamping',o.MinDamping);
    tauRatio = sinfo.tauRatio;

    tbl(ii,:) = [J,NaN,NaN,sinfo.iterations,tauRatio,statusCode(sinfo.status)];
    if ~sinfo.ok, continue; end

    alpha = sqrt(x(:))';
    I = local_codelength(delta,N,K,J,tau,alpha,em,o.SignQuotient);
    tbl(ii,2:3) = [I,tau];

    if I < best.codelength
        P = U(:,1:J);
        best = struct('J',J,'mu',mu,'tau',tau, ...
            'alpha',alpha,'b',alpha/sqrt(tau),'P',P, ...
            'A',P*diag(alpha),'B',P*diag(alpha)/sqrt(tau), ...
            'Omega',P*diag(x)*P'+tau*eye(K), ...
            'codelength',I,'converged',true, ...
            'methodUsed',sinfo.status,'tauRatio',tauRatio);
    end
end

if ~isfinite(best.codelength)
    best = struct('J',Jlist(1),'mu',mu,'tau',NaN,'alpha',[],'b',[], ...
        'P',zeros(K,0),'A',zeros(K,0),'B',zeros(K,0), ...
        'Omega',[],'codelength',inf,'converged',false, ...
        'methodUsed','none','tauRatio',NaN,'collapsed',true);
else
    best.collapsed = false;
end

res = best;
res.table = tbl;
res.solver = 'fixedpoint';
res.unresolved = tbl(tbl(:,6) == -1, 1)';

%% Diagnostics and warnings
res.diagnostics = local_diagnostics(res,tbl,delta,N,K,Jmax,numel(Jlist)>1);
res.warnings = cell(0,2);
if ~isempty(res.unresolved)
    res.warnings(end+1,:) = {'mml_ppca:unresolved',sprintf([ ...
        'Existence of a regular solution could not be determined for J = %s; ' ...
        'these ranks were excluded from the comparison.'],mat2str(res.unresolved))};
end
if ~res.collapsed
    d = res.diagnostics;
    if d.saturated
        res.warnings(end+1,:) = {'mml_ppca:saturated',sprintf([ ...
            'Selected J=K-1=%d. This is the largest candidate rank and ' ...
            'the selected J may carry little information.'],K-1)};
    end
    if d.boundary && ~d.saturated
        res.warnings(end+1,:) = {'mml_ppca:boundary',sprintf([ ...
            'The codelength was still decreasing at MaxFactors=%d. ' ...
            'Increase MaxFactors if scientifically appropriate.'],Jmax)};
    end
    if d.sphericityDf>0 && d.sphericityZ>3 && d.sphericityAtJ==res.J
        res.warnings(end+1,:) = {'mml_ppca:sphericity',sprintf([ ...
            'Residual eigenvalues are far from equal (AM/GM=%.2f, z=%.1f). ' ...
            'The isotropic residual assumption may be inadequate.'], ...
            d.residHeterogeneity,d.sphericityZ)};
    end
end

if o.Warn
    for ii=1:size(res.warnings,1)
        warning(res.warnings{ii,1},'%s',res.warnings{ii,2});
    end
end

if o.Verbose
    fprintf('   J   codelength        tau        iters   tau/tauMax\n');
    for ii=1:size(tbl,1)
        if tbl(ii,6) == -1
            fprintf('%4d      UNRESOLVED                %5d      %8.4f\n', ...
                tbl(ii,1),tbl(ii,4),tbl(ii,5));
        elseif isnan(tbl(ii,2))
            fprintf('%4d      no regular solution       %5d      %8.4f\n', ...
                tbl(ii,1),tbl(ii,4),tbl(ii,5));
        else
            mark='';
            if tbl(ii,1)==res.J, mark='  <-- MML'; end
            fprintf('%4d %12.4f %12.6f %8d   %8.4f%s\n', ...
                tbl(ii,1),tbl(ii,2),tbl(ii,3),tbl(ii,4),tbl(ii,5),mark);
        end
    end
    d = res.diagnostics;
    fprintf(['\n  residual heterogeneity (AM/GM) = %.3f   sphericity z = %s ' ...
        '(at J=%d)   tau/lambda_K = %.3f\n'],d.residHeterogeneity, ...
        local_num2str(d.sphericityZ),d.sphericityAtJ,d.tauRatio);
    for ii=1:size(res.warnings,1)
        fprintf('  [!] %s\n',res.warnings{ii,2});
    end
end
end

%% -----------------------------------------------------------------------
% Diagnostics and codelength helpers (shared with mml_ppca.m).
function d=local_diagnostics(res,tbl,delta,N,K,Jmax,scanned)
d=struct('saturated',false,'boundary',false,'interiorMinimum',false, ...
    'residHeterogeneity',NaN,'sphericityStat',NaN,'sphericityDf',0, ...
    'sphericityZ',NaN,'sphericityAtJ',NaN,'tauRatio',NaN, ...
    'sphericitySingular',false);
if res.collapsed, return; end
J=res.J;
d.saturated=(J==K-1)&&(K>1);
if scanned
    valid=isfinite(tbl(:,2)); jv=tbl(valid,1); vv=tbl(valid,2);
    if numel(vv)>1
        [~,im]=min(vv);
        d.boundary=(jv(im)==Jmax)&&(jv(end)==Jmax)&&(vv(end)<vv(end-1));
        d.interiorMinimum=(im>1)&&(im<numel(vv));
    end
end
if delta(end)>0, d.tauRatio=res.tau/delta(end); end
Jref=max(0,min(J,K-2)); d.sphericityAtJ=Jref;
tr=delta(Jref+1:end); m=numel(tr);
if any(tr<=0)
    % singular residual spectrum: AM/GM is infinite and the chi-square
    % calibration of the sphericity statistic is unavailable
    d.residHeterogeneity=Inf; d.sphericityZ=NaN; d.sphericitySingular=true;
elseif m>=2
    d.residHeterogeneity=mean(tr)/exp(mean(log(tr)));
    d.sphericityDf=m*(m+1)/2-1;
    d.sphericityStat=N*m*log(d.residHeterogeneity);
    df=d.sphericityDf;
    if df>0&&d.sphericityStat>0
        d.sphericityZ=((d.sphericityStat/df)^(1/3)-(1-2/(9*df)))/sqrt(2/(9*df));
    else
        d.sphericityZ=-Inf;
    end
else
    d.residHeterogeneity=1; d.sphericityZ=-Inf;
end
end

function s=local_num2str(z)
if ~isfinite(z), s='n/a'; else, s=sprintf('%.1f',z); end
end

%% -----------------------------------------------------------------------
function I=local_codelength(delta,N,K,J,tau,alpha,em,signquot)
y=alpha(:).^2+tau;
params=1+J+J*K-J*(J+1)/2+em*K;
h=0.5*log(tau);
if J>0
    h=h+K*J/2*log(pi)-local_lmvgamma(J,K/2);
    h=h-J*log(2)-J^2/2*log(pi)-J^2/2*log(tau) ...
        +local_lmvgamma(J,J/2)+local_lmvbeta(J,K/2,J/2) ...
        -(K-J)*sum(log(alpha))+(K+J)/2*sum(log(y));
    h=h+J*log(2);
    if signquot, h=h-J*log(2); end
end
F=params/2*log(N)+(J+1)/2*log(2)+0.5*log(K-J) ...
    -(J*(K-J)+1)/2*log(tau);
if J>0
    F=F+(2*(K-J)+1)*sum(log(alpha))-(K+1)/2*sum(log(y));
end
if em
    F=F-0.5*((K-J)*log(tau)+sum(log(y)));
end
L=N*K/2*log(2*pi)+N/2*((K-J)*log(tau)+sum(log(y))) ...
    +N/2*(sum(delta(J+1:end))/tau+sum(delta(1:J)./y));
I=h+F+L+local_mmlconst(params);
end

function c=local_mmlconst(D)
kap=[1/12,5/(36*sqrt(3)),19/(192*2^(1/3)),13/120/sqrt(2), ...
    2641/(23040*2^(3/5)),12619/(68040*3^(5/6)), ...
    21361/(161280*2^(6/7)),929/12960,0.071622594,0.070813818, ...
    0.070426259,0.070095600,0.071034583,0.071455542,0.071709124,0.06830];
if D<=0, c=0;
elseif D<=16, c=(D/2)*(log(kap(D))+1);
else, c=-D/2*log(2*pi)+0.5*log(D*pi)-0.5772156649015329;
end
end

function v=local_lmvgamma(p,a)
if p<=0, v=0; return; end
j=(1:p)';
v=p*(p-1)/4*log(pi)+sum(gammaln(a+(1-j)/2));
end

function v=local_lmvbeta(p,a,b)
v=local_lmvgamma(p,a)+local_lmvgamma(p,b)-local_lmvgamma(p,a+b);
end
