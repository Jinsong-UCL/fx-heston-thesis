function verifyModels(runWorkflows)
% Deterministic price checks; optionally run short fits and simulations.
if nargin<1,runWorkflows=false;end
for N=1:2
 d=loadExample(sprintf('vector_N%d',N));row=find(d.tau==1,1);
 v=vectorPrices(d.parameters,1,d.strikes(row,:),d.theta.',d.spot,d.eurYield(row),d.usdYield(row),struct('step',.025,'cutoff',1800));
 assert(max(abs(v(:)-d.validation.prices(:)))<2e-12,'Vector reference mismatch.');
 K=d.strikes(row,6);c=vectorPrices(d.parameters,1,K,1,d.spot,d.eurYield(row),d.usdYield(row));
 p=vectorPrices(d.parameters,1,K,-1,d.spot,d.eurYield(row),d.usdYield(row));
 assert(abs(c-p-d.spot*exp(-d.eurYield(row))+K*exp(-d.usdYield(row)))<1e-12);
end
d=loadExample('matrix');m=d.market;th=repmat([-1 -1 -1 -1 -1 1 1 1 1],numel(m.tau),1);
T=repmat(m.tau(:),1,9);yi=repmat(m.eurYield(:),1,9);yj=repmat(m.usdYield(:),1,9);
v=matrixAuditPrices(d.parameters,T,m.strikes,th,m.spot,yi,yj,struct('nodes',560,'cutoff',1200,'tolerance',1e-10));
assert(max(abs(v-d.validation.prices),[],'all')<1e-9,'Matrix reference mismatch.');
assert(checkCorrectedLinearisation(d.parameters)<1e-7);
price=computeBSPrices(1.2,1.3,.2,.01,.03,2,1);
assert(abs(fxImpliedVolatility(price,1.2,1.3,.01,.03,2,1)-.2)<1e-10);
if runWorkflows
 for N=1:2
  fit=calibrateVector(N,1,struct('MaxFunctionEvaluations',100));
  assert(all(isfinite(fit.prices),'all'));
  r=runVectorSimulation(N,struct('Parameters',fit.parameters,'Paths',2000,'Blocks',20,'Steps',100));
  assert(all(isfinite(r.price)) && all(r.se>0));
 end
 r=runVectorSimulation(2,struct('Scheme','EM','Paths',2000,'Blocks',20,'Steps',100));
 assert(all(isfinite(r.price)));
 fit=calibrateMatrix(1,struct('MaxFunctionEvaluations',100));
 assert(all(isfinite(fit.prices),'all'));
 r=runMatrixSimulation(struct('Parameters',fit.parameters,'Paths',2000,'Steps',100));
 assert(all(isfinite(r.prices)) && all(r.priceSE>0));
end
disp('All numerical and requested workflow checks passed.');
end
