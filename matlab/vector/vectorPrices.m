function prices=vectorPrices(p,tau,K,theta,S0,Ym,Yn,settings)
% Lewis half-moment inversion with domestic discount and FX forward matching.
if nargin<8,settings=struct;end
if ~isfield(settings,'step'),settings.step=.05;end
if ~isfield(settings,'cutoff'),settings.cutoff=900;end
validateattributes(settings.step,{'numeric'},{'scalar','positive','finite'});
validateattributes(settings.cutoff,{'numeric'},{'scalar','positive','finite'});
validateattributes(S0,{'numeric'},{'scalar','positive','finite'});
tau=tau+zeros(size(K));theta=theta+zeros(size(K));
Ym=Ym+zeros(size(K));Yn=Yn+zeros(size(K));
assert(all(K>0 & isfinite(K) & tau>0 & isfinite(tau),'all'));
assert(all(abs(theta)==1,'all') && all(isfinite(Ym) & isfinite(Yn),'all'));
validateVectorParameters(p,max(tau,[],'all'));
n=round(settings.cutoff/settings.step);
assert(mod(n,2)==0 && abs(n*settings.step-settings.cutoff)<1e-8,'Simpson grid must have an even integer interval count.');
u=(0:n)*settings.step;z=u-.5i;
w=2*ones(1,n+1);w(2:2:n)=4;w([1 end])=1;
prices=zeros(size(K));
for T=reshape(unique(tau(:)),1,[])
 ix=find(tau==T);strikes=reshape(K(ix),[],1);
 yi=reshape(Ym(ix),[],1);yj=reshape(Yn(ix),[],1);signs=reshape(theta(ix),[],1);
 raw=vectorLogTransform(p,[0 -1i z],T);
 y0n=yj+real(raw(1))/T;y0m=yi+real(raw(2))/T;
 phi=exp(raw(3:end)+(-y0n+1i*z.*(y0n-y0m))*T);
 integral=real(exp(-1i*log(strikes/S0)*u).*phi)*(w./(u.^2+.25)).'*settings.step/3;
 base=(signs>0).*S0.*exp(-yi*T)+(signs<0).*strikes.*exp(-yj*T);
 prices(ix)=base-sqrt(S0*strikes).*integral/pi;
end
assert(all(isfinite(prices),'all'),'Vector:Nonfinite','Nonfinite prices.');
end
