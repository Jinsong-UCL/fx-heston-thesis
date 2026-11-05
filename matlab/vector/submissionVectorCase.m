function r=submissionVectorCase(model,cfg)
% Vector-only validation: full-truncation Euler or quadratic-exponential variance.
% Both market discount/forward moments are matched; parameters are not refitted.
T=model.T; K=model.strikes; th=model.theta; xi=model.xi; S0=model.S0;
assert(any(strcmp(cfg.scheme,{'EM','QE'})),'Scheme must be EM or QE.');
validateattributes(cfg.steps,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.blocks,{'numeric'},{'scalar','integer','>=',2});
validateattributes(cfg.paths,{'numeric'},{'scalar','integer','positive'});
n=cfg.paths/cfg.blocks; assert(n==floor(n)); dt=T/cfg.steps;
assert(strcmp(model.kind,'vector'),'This engine supports vector models only.');
    P=reshape(model.p,model.N,9); kap=P(:,1)'; vb=P(:,2)'; sig=P(:,3)'; rho=P(:,4)';
    v0=P(:,5)'; a=(P(:,7)-P(:,6)); rm=P(:,8); rn=P(:,9);
    assert(all(P(:,6)==0),'Only calibrated am=0 models are supported.');
    L=vectorLog(model.p,model.N,[0 -1i xi],T);
    % Domestic discount and foreign forward shifts from the same transform.
    y0n=model.Yn+real(L(1))/T;
    y0m=model.Ym+real(L(2))/T;
    r.reference=model.referencePrices;
    r.cf=exp(L(3:end)+(-y0n+1i*xi*(y0n-y0m))*T); r.sdeReference=r.reference;
    r.beta=kap.*vb./sig.^2;
bp=zeros(cfg.blocks,numel(K)); bsp=bp; becf=zeros(cfg.blocks,numel(xi)); bsecf=becf;
bdiag=zeros(cfg.blocks,7); bvar=cell(cfg.blocks,1); bcount=zeros(cfg.blocks,numel(r.beta));
tic;
for block=1:cfg.blocks
    rng(cfg.seed+block,'twister');
    x=zeros(n,1); directS=S0*ones(n,1); integ=zeros(n,1); badS=0;
    count=zeros(1,numel(r.beta));
    v=repmat(v0,n,1);
    for step=1:cfg.steps
        vp=max(v,0);
            if strcmp(cfg.scheme,'QE')
                vn=qe(vp,kap,vb,sig,dt);
                av=(vp+vn)/2;
                % Trapezoidal integrated variance in BOTH innovation and residual noise.
                dw=((vn-vp-(vb-av).*kap*dt).*(rho./sig) ...
                    +sqrt(av*dt).*sqrt(1-rho.^2).*randn(n,model.N))*a;
                mu=(y0n-y0m)+(av*(rn-rm));
                dx=mu*dt-0.5*(av*(a.^2))*dt+dw;
                integ=integ+(y0n+av*rn)*dt;
            else
                z=randn(n,model.N); w=z.*rho+randn(n,model.N).*sqrt(1-rho.^2);
                vn=v+(vb-vp).*kap*dt+sqrt(vp*dt).*sig.*z;
                count=count+sum(vn<0,1);
                dw=(sqrt(vp*dt).*w)*a;
                mu=(y0n-y0m)+vp*(rn-rm);
                dx=mu*dt-0.5*(vp*(a.^2))*dt+dw;
                integ=integ+(y0n+vp*rn)*dt;
                factor=1+mu*dt+dw; badS=badS+sum(factor<=0);
                directS=directS.*factor;
            end
            v=vn;
        x=x+dx;
    end
    discount=exp(-integ); terminal=S0*exp(x);
    bp(block,:)=mean(discount.*max(th.*(terminal-K),0),1);
    bsp(block,:)=mean(discount.*max(th.*(directS-K),0),1);
    becf(block,:)=mean(discount.*exp(1i*x*xi),1);
    if all(directS>0), bsecf(block,:)=mean(discount.*exp(1i*log(directS/S0)*xi),1); else, bsecf(block,:)=NaN; end
    bdiag(block,:)=[mean(discount) mean(discount.*terminal) badS/(n*cfg.steps) ...
        mean(abs(terminal-directS)) mean(discount.^2) mean(terminal) mean(x)];
    bvar{block}=mean(max(v,0),1);
    bcount(block,:)=count/(n*cfg.steps);
end
r.seconds=toc; r.model=model; r.cfg=cfg;
r.price=mean(bp); r.se=std(bp)/sqrt(cfg.blocks); r.ci95=[r.price-1.96*r.se;r.price+1.96*r.se];
r.directPrice=mean(bsp); r.directSE=std(bsp)/sqrt(cfg.blocks);
r.pairedDifference=mean(bsp-bp); r.pairedDifferenceSE=std(bsp-bp)/sqrt(cfg.blocks);
r.ecf=mean(becf); r.ecfSE=std(becf)/sqrt(cfg.blocks); r.directECF=mean(bsecf);
r.ecfRealSE=std(real(becf))/sqrt(cfg.blocks); r.ecfImagSE=std(imag(becf))/sqrt(cfg.blocks);
r.diagnostics=mean(bdiag); r.diagnosticsSE=std(bdiag)/sqrt(cfg.blocks);
r.boundaryFraction=mean(bcount); r.terminalVariance=mean(vertcat(bvar{:}),1);
r.terminalVarianceSE=std(vertcat(bvar{:}),0,1)/sqrt(cfg.blocks);
r.blockPrices=bp; r.blockDirectPrices=bsp; r.blockECF=becf; r.blockDiagnostics=bdiag;
if strcmp(cfg.scheme,'QE')
    r.directPrice(:)=NaN; r.directSE(:)=NaN; r.directECF(:)=NaN;
    r.pairedDifference(:)=NaN; r.pairedDifferenceSE(:)=NaN; r.diagnostics(3:4)=NaN;
end
end

function out=vectorLog(p,N,u,T)
P=reshape(p,N,9); k=P(:,1); v=P(:,2); s=P(:,3); rho=P(:,4); v0=P(:,5); am=P(:,6); an=P(:,7); rm=P(:,8); rn=P(:,9);
c=k-1i*u.*(rho.*s.*(an-am)); h=(an-am).^2.*u.^2+1i*u.*(an.^2-am.^2)-2i*u.*(rn-rm)+2*rn;
d=sqrt(c.^2+s.^2.*h); g=(c-d)./(c+d);
out=(k.*v./s.^2).' *((c-d)*T+2*log((1-g)./(1-g.*exp(-d*T)))) ...
    +(v0./s.^2).'*((c-d).*(1-exp(-d*T))./(1-g.*exp(-d*T)));
end
function vn=qe(v,k,b,s,dt)
e=exp(-k*dt); m=b+(v-b).*e;
s2=v.*s.^2.*e./k.*(1-e)+b.*s.^2./(2*k).*(1-e).^2;
psi=s2./max(m.^2,realmin); U=rand(size(v)); vn=zeros(size(v)); low=psi<=1.5;
b2=2./psi(low)-1+sqrt(2./psi(low)).*sqrt(2./psi(low)-1);
vn(low)=m(low)./(1+b2).*(sqrt(b2)+sqrt(2)*erfinv(2*U(low)-1)).^2;
high=~low; prob=(psi(high)-1)./(psi(high)+1); z=U(high); values=zeros(size(z)); pos=z>prob;
mh=m(high); values(pos)=log((1-prob(pos))./(1-z(pos))).*mh(pos)./(1-prob(pos)); vn(high)=values;
end
