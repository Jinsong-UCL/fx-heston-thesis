function r=simulateCorrectedMatrix(p,m,steps,T,paths)
% Exact OU-factor variance transitions for beta=1/2, no eigenvalue projection.
% V=X X', dX=-kappa X dt/2 + sigma' dB/2, with two independent columns.
% Asset Brownian increment is coupled to the SAME dB as the OU transition.
if nargin<4,T=1;end
if nargin<5,paths=200000;end
validateattributes(steps,{'numeric'},{'scalar','integer','positive'});
validateattributes(paths,{'numeric'},{'scalar','integer','positive'});
assert(mod(paths,40)==0 && paths>=80,'Use at least 80 paths, divisible by 40.');
validateMatrixDiffusion(p);assert(all(abs([p.beta]-.5)<1e-12));
blocks=40;n=paths/blocks;dt=T/steps;xi=0:2:100;nf=numel(xi);
ix=find(abs(m.tau-T)<1e-10,1);assert(~isempty(ix));Yn=m.usdYield(ix);Ym=m.eurYield(ix);
K=m.strikes(ix,[1 3 5 7 9]);th=[-1 -1 1 1 1];
L=zeros(1,nf+2);for b=1:2,L=L+matrixAuditTransform(p(b),[0 -1i xi],T,1e-11,'sde');end
y0n=Yn+real(L(1))/T;y0m=Ym+real(L(2))/T;
cf=exp(L(3:end)+(-y0n+1i*xi*(y0n-y0m))*T);
prices=matrixAuditPrices(p,T*ones(size(K)),K,th,m.spot,Ym*ones(size(K)),Yn*ones(size(K)),struct('nodes',560,'cutoff',1200,'tolerance',1e-10));
expectedV=[];
for b=1:2
 q=p(b);F=-q.kappa/2;G=q.sigma.'/2;
 M=expm([F G*G.';zeros(2) -F.']*dt);E=M(1:2,1:2);C=M(1:2,3:4)*E.';C=(C+C.')/2;
 J=expm([F G;zeros(2,4)]*dt);J=J(1:2,3:4);
 residual=C-J*J.'/dt;residual=(residual+residual.')/2;[U,D]=eig(residual);
 assert(min(diag(D))>-1e-10*max(norm(C),realmin),'OU innovation covariance failure');
 p(b).E=E;p(b).J=J/dt;p(b).R=U*diag(sqrt(max(diag(D),0)));
 p(b).X0=chol(q.v0,'lower');p(b).independent=sqrtm(eye(2)-q.rho.'*q.rho);
 E=expm(F*T);source=G*G.';
 stationary=reshape(-(kron(eye(2),F)+kron(F,eye(2)))\source(:),2,2);
 CT=stationary-E*stationary*E.';
 ev=E*q.v0*E.'+2*CT;expectedV=[expectedV ev(:).']; %#ok<AGROW>
end
becf=zeros(blocks,nf);bp=zeros(blocks,5);bd=zeros(blocks,3);bv=zeros(blocks,8);tic;
for block=1:blocks
 rng(922761+block,'twister');X={repmat(p(1).X0,1,1,n),repmat(p(2).X0,1,1,n)};
 xr=zeros(n,1);xf=xr;integ=xr;
 for step=1:steps
  for b=1:2
   q=p(b);old=X{b};B=randn(2,2,n)*sqrt(dt);
   X{b}=pagemtimes(q.E,old)+pagemtimes(q.J,B)+pagemtimes(q.R,randn(2,2,n));
   if b==1
    V=.5*(pagemtimes(old,pagetranspose(old))+pagemtimes(X{b},pagetranspose(X{b})));
    xr=xr+contract(q.rn-q.rm,V)*dt;integ=integ+contract(q.rn,V)*dt;
   else
    AX=pagemtimes(q.an,old);corr=pagemtimes(q.rho,AX);ind=pagemtimes(q.independent,AX);
    variance=reshape(sum(sum(AX.^2,1),2),n,1);
    dw=reshape(sum(sum(corr.*B,1),2),n,1)+sqrt(reshape(sum(sum(ind.^2,1),2),n,1)*dt).*randn(n,1);
    xf=xf-.5*variance*dt+dw;
   end
  end
 end
 x=xr+xf+(y0n-y0m)*T;discount=exp(-integ-y0n*T);S=m.spot*exp(x);
 becf(block,:)=mean(discount.*exp(1i*x*xi));bp(block,:)=mean(discount.*max(th.*(S-K),0));
 bd(block,:)=[mean(discount) mean(discount.*S) mean(discount.*x)];
 vv=zeros(1,8);
 for b=1:2,V=pagemtimes(X{b},pagetranspose(X{b}));vv(4*(b-1)+(1:4))=reshape(mean(V,3),1,4);end
 bv(block,:)=vv;
end
r=struct('date',m.date,'T',T,'steps',steps,'paths',paths,'blocks',blocks,'seconds',toc,...
 'xi',xi,'cfReal',real(cf),'cfImag',imag(cf),'ecfReal',real(mean(becf)),'ecfImag',imag(mean(becf)),...
 'realSE',std(real(becf))/sqrt(blocks),'imagSE',std(imag(becf))/sqrt(blocks),...
 'strikes',K,'theta',th,'referencePrices',prices,'prices',mean(bp),'priceSE',std(bp)/sqrt(blocks),...
 'discountForwardLogMoments',mean(bd),'momentSE',std(bd)/sqrt(blocks),...
 'expectedDiscount',exp(-Yn*T),'expectedDiscountedSpot',m.spot*exp(-Ym*T),...
 'varianceMean',mean(bv),'varianceSE',std(bv)/sqrt(blocks),'expectedVarianceMean',expectedV,...
 'method','Exact two-column OU Wishart transitions, correlated log-price Euler, trapezoidal rate integral; no spectral projection.');
r.maxRealZ=max(abs(r.ecfReal-r.cfReal)./r.realSE);
r.maxImagZ=max(abs(r.ecfImag(2:end)-r.cfImag(2:end))./r.imagSE(2:end));
r.priceZ=(r.prices-r.referencePrices)./r.priceSE;
r.varianceZ=(r.varianceMean-r.expectedVarianceMean)./r.varianceSE;
fprintf('OU MC %s T=%g steps=%d realZ %.3f imagZ %.3f maxPriceZ %.3f\n',m.date,T,steps,r.maxRealZ,r.maxImagZ,max(abs(r.priceZ)));
end
function x=contract(A,V),x=reshape(sum(sum(A.'.*V,1),2),[],1);end
