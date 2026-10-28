function [delta,U,true_params,X] = generate_pca_data(N,K,varargin)
%GENERATE_PCA_DATA Generate data from a probabilistic PCA model.
%
% [delta,U,true_params,X] = generate_pca_data(N,K,Name,Value)
%
% Model
%   x_n ~ N_K(mu, R*diag(lambda2)*R' + sigma2*I_K).
%
% Name-value options
%   'sigma2'          residual variance; [] samples sigma from 1/sigma
%                     on SigmaRange                         (default [])
%   'SigmaRange'      [sigmaMin sigmaMax] for log-uniform sigma
%                     sampling when sigma2=[]             (default [0.25 4])
%   'J'               number of latent factors                  (default 1)
%   'SNR'             sum(lambda2)/(K*sigma2)                   (default 1)
%   'LoadingDesign'   'equal', 'geometric', or 'halfcauchy'     (default halfcauchy)
%   'GeometricRatio'  rho_(j+1)/rho_j for geometric design      (default 0.75)
%   'Lambda2'         explicit J-vector of loading strengths     (default [])
%   'Mu'              1-by-K population mean                    (default zeros)
%   'CenterData'      use sample-centred covariance for delta,U  (default false)
%   'Stream'          RandStream used for all random draws       (default [])
%
% The component signal-to-noise ratios rho_j=lambda_j^2/sigma2 are
% ordered decreasingly and normalized so that sum(rho_j)/K=SNR. If Lambda2
% is supplied, LoadingDesign and SNR are ignored for the loading strengths.
%
% Outputs
%   delta       ordered eigenvalues of Xwork'*Xwork/N
%   U           corresponding ordered sample eigenvectors
%   true_params structure containing mu, sigma2, rho, lambda2, alpha, R,
%               A, and Sigma
%   X           raw N-by-K observations (not sample-centred)

p=inputParser;
p.addRequired('N',@(x) isscalar(x)&&x>=2&&x==floor(x));
p.addRequired('K',@(x) isscalar(x)&&x>=1&&x==floor(x));
p.addParameter('sigma2',[],@(x) isempty(x)||(isscalar(x)&&isfinite(x)&&x>0));
p.addParameter('SigmaRange',[0.25 4],@(x) isnumeric(x)&&numel(x)==2&& ...
    all(isfinite(x))&&all(x>0)&&x(1)<x(2));
p.addParameter('J',1,@(x) isscalar(x)&&x>=0&&x==floor(x));
p.addParameter('SNR',1,@(x) isscalar(x)&&isfinite(x)&&x>=0);
p.addParameter('LoadingDesign','halfcauchy',@(x) ischar(x)||isstring(x));
p.addParameter('GeometricRatio',0.75,@(x) isscalar(x)&&x>0&&x<=1);
p.addParameter('Lambda2',[],@(x) isempty(x)||(isnumeric(x)&&isvector(x)));
p.addParameter('Mu',[],@(x) isempty(x)||(isnumeric(x)&&isvector(x)));
p.addParameter('CenterData',false,@islogical);
p.addParameter('Stream',[],@(x) isempty(x)||isa(x,'RandStream'));
p.parse(N,K,varargin{:});
o=p.Results;
J=o.J;
if J>=K, error('generate_pca_data:rank','Require 0 <= J < K.'); end

if isempty(o.Mu), mu=zeros(1,K); else, mu=reshape(o.Mu,1,[]); end
if numel(mu)~=K, error('generate_pca_data:mu','Mu must contain K elements.'); end
stream=o.Stream;

% The bounded density pi(sigma) proportional to 1/sigma is uniform on the
% log scale. If sigma2 is supplied, no scale sampling is performed.
if isempty(o.sigma2)
    sigmaRange=reshape(o.SigmaRange,1,[]);
    if isempty(stream), usigma=rand; else, usigma=rand(stream,1,1); end
    logSigma=log(sigmaRange(1))+usigma*(log(sigmaRange(2))-log(sigmaRange(1)));
    sigma=exp(logSigma);
    sigma2=sigma^2;
    sigmaSampled=true;
else
    sigma2=o.sigma2;
    sigma=sqrt(sigma2);
    sigmaRange=reshape(o.SigmaRange,1,[]);
    sigmaSampled=false;
end

if ~isempty(o.Lambda2)
    lambda2=sort(reshape(o.Lambda2,[],1),'descend');
    if numel(lambda2)~=J || any(~isfinite(lambda2)) || any(lambda2<=0)
        error('generate_pca_data:lambda2','Lambda2 must contain J positive finite values.');
    end
    rho=lambda2/sigma2;
else
    design=lower(string(o.LoadingDesign));
    switch design
        case "equal"
            weights=ones(J,1);
        case "geometric"
            weights=o.GeometricRatio.^(0:J-1)';
        case "halfcauchy"
            if isempty(stream), u=rand(J,1); else, u=rand(stream,J,1); end
            z=abs(tan(pi*(u-0.5)));
            weights=max(z.^2,realmin('double'));
        otherwise
            error('generate_pca_data:design', ...
                'LoadingDesign must be equal, geometric, or halfcauchy.');
    end
    if J==0
        rho=zeros(0,1);
    elseif o.SNR==0
        error('generate_pca_data:snr','SNR must be positive when J > 0.');
    else
        rho=K*o.SNR*weights/sum(weights);
        rho=sort(rho,'descend');
    end
    lambda2=sigma2*rho;
end

if J>0
    if isempty(stream), Z=randn(K,J); else, Z=randn(stream,K,J); end
    [R,~]=qr(Z,0);
    % Standardize QR signs for reproducibility across equivalent outputs.
    sg=sign(diag(R(1:J,1:J))); sg(sg==0)=1;
    R=R*diag(sg);
else
    R=zeros(K,0);
end
A=R*diag(sqrt(lambda2));
Sigma=A*A'+sigma2*eye(K);
L=chol((Sigma+Sigma')/2,'lower');
if isempty(stream), E=randn(N,K); else, E=randn(stream,N,K); end
X=bsxfun(@plus,E*L',mu);

if o.CenterData
    Xwork=bsxfun(@minus,X,mean(X,1));
else
    Xwork=X;
end
S=(Xwork'*Xwork)/N;
S=(S+S')/2;
[U,D]=eig(S,'vector');
[delta,ord]=sort(real(D),'descend');
U=U(:,ord);
delta=max(delta,0);

true_params=struct();
true_params.mu=mu;
true_params.sigma2=sigma2;
true_params.sigma=sigma;
true_params.sigmaSampled=sigmaSampled;
true_params.SigmaRange=sigmaRange;
true_params.rho=rho;
true_params.lambda2=lambda2;
true_params.alpha=sqrt(lambda2);
true_params.R=R;
true_params.A=A;
true_params.Sigma=Sigma;
true_params.SNR=sum(lambda2)/(K*sigma2);
true_params.LoadingDesign=char(lower(string(o.LoadingDesign)));
true_params.GeometricRatio=o.GeometricRatio;
end
