function [prices,diagnostics] = matrixAuditPrices(blocks,tau,K,theta,S0,Ym,Yn,settings)
% Block Wishart pricing using Riccati ODEs and Lewis half-moment inversion.
% am=0, with rate and FX blocks disjoint, matches thesis Matrix (2,2).
% The half-moment contour avoids the legacy pricer's extreme damping shifts.
if nargin<8, settings=struct; end
if ~isfield(settings,'allowFormal') || ~settings.allowFormal
    validateMatrixDiffusion(blocks);
end
if ~isfield(settings,'nodes'), settings.nodes=160; end
if ~isfield(settings,'cutoff'), settings.cutoff=600; end
if ~isfield(settings,'tolerance'), settings.tolerance=2e-7; end
[u,w] = quadrature(settings.nodes,settings.cutoff);
[t,~,ti] = unique(tau(:)); nt=numel(t);
xi = [0 -1i reshape(u-0.5i,1,[])];
L = zeros(nt,numel(xi));
for b=1:numel(blocks)
    L = L + matrixAuditTransform(blocks(b),xi,t,settings.tolerance,'sde');
end
% Anchor both discount and forward moments. For disjoint rate/FX blocks
% these coincide with the thesis's individual currency yield corrections.
ym = Ym(:); yn = Yn(:); tm = tau(:);
cn = real(L(:,1)); cm = real(L(:,2));
correction = -cn + 1i*xi.*(cn-cm);
normalized = L+correction;
if max(abs(imag(L(:,1:2))),[],'all')>1e-6
    error('Riccati:Moments','Complex discount/forward moment.');
end
lp = normalized(ti,3:end) + (-yn+1i*(u.'-0.5i).*(yn-ym)).*tm;
phi = exp(lp);
integral = real(exp(-1i*log(K(:)/S0).*u.').*phi) * (w./(u.^2+0.25));
call = S0*exp(-ym.*tm)-sqrt(S0*K(:)).*integral/pi;
prices = call-(theta(:)<0).*(S0*exp(-ym.*tm)-K(:).*exp(-yn.*tm));
prices = reshape(prices,size(K));
diagnostics.maxTailTransform = max(abs(phi(:,end)));
diagnostics.discountMomentError = max(abs(normalized(:,1)));
diagnostics.forwardMomentError = max(abs(normalized(:,2)));
diagnostics.quadratureNodes = numel(u);
diagnostics.cutoff = settings.cutoff;
diagnostics.odeTolerance = settings.tolerance;
end

function [u,w] = quadrature(n,cutoff)
persistent savedN savedCutoff savedU savedW
if isequal(n,savedN) && isequal(cutoff,savedCutoff)
    u=savedU; w=savedW; return
end
% Composite Gauss-Legendre grid resolves the kernel near zero and the tail.
edges = [0 1 4 12 35 100 250 cutoff];
edges = unique(edges(edges<=cutoff));
m = ceil(n/(numel(edges)-1)); j=(1:m-1)';
beta=j./sqrt(4*j.^2-1); [V,D]=eig(diag(beta,1)+diag(beta,-1));
[x,ix]=sort(diag(D)); weights=2*V(1,ix)'.^2;
u=[]; w=[];
for k=1:numel(edges)-1
    h=(edges(k+1)-edges(k))/2; c=(edges(k+1)+edges(k))/2;
    u=[u; c+h*x]; w=[w; h*weights]; %#ok<AGROW>
end
savedN=n; savedCutoff=cutoff; savedU=u; savedW=w;
end
