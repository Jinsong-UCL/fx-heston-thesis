function r=calibrateMatrix(maxIterations,options)
% January 2013 Matrix (2,2), vector-style relative-price elastic-net loss.
% Penalise physical matrices, never scaled optimiser/Cholesky coordinates.
if nargin<1,maxIterations=150;end
if nargin<2,options=struct;end
validateattributes(maxIterations,{'numeric'},{'scalar','integer','positive'});
root=hestonRoot();
if ~isfield(options,'OutputDir'),options.OutputDir=fullfile(root,'results','matrix');end
if ~isfield(options,'UseParallel'),options.UseParallel=false;end
if ~isfield(options,'MaxFunctionEvaluations'),options.MaxFunctionEvaluations=16000;end
out=options.OutputDir;if ~exist(out,'dir'),mkdir(out);end
input=loadExample('matrix');
baseline=struct('market',input.market,'parameters',input.ivBaselineParameters);
m=baseline.market;p=input.initialParameters;
if isfield(options,'InitialParameters'),p=options.InitialParameters;end
[x,lb,ub,scale]=encode(p);alpha=1e-4;lambda=.5;
assert(all(x>=lb & x<=ub),'Initial point outside optimiser bounds');
assert(max(abs(physical(decode(x,scale))-physical(p)))<1e-10,'Parameter encoding is not price-preserving');
T=repmat(m.tau(:),1,9);ym=repmat(m.eurYield(:),1,9);yn=repmat(m.usdYield(:),1,9);
th=repmat([-1 -1 -1 -1 -1 1 1 1 1],numel(m.tau),1);
settings=struct('nodes',210,'cutoff',900,'tolerance',1e-10);
if options.UseParallel && isempty(gcp('nocreate')),parpool;end
best=x;bestf=objective(x);history=[];timer=tic;
fprintf('Initial relative-price elastic-net objective %.12g\n',bestf);
opt=optimoptions('fmincon','Algorithm','sqp','Display','iter','UseParallel',options.UseParallel,...
 'MaxIterations',maxIterations,'MaxFunctionEvaluations',options.MaxFunctionEvaluations,'FiniteDifferenceStepSize',1e-4,...
 'OptimalityTolerance',1e-7,'StepTolerance',1e-9,'ConstraintTolerance',1e-8,'OutputFcn',@record);
[xx,f,flag,info]=fmincon(@objective,x,[],[],[],[],lb,ub,@constraints,opt);
if f<bestf && max(constraints(xx))<=1e-7,best=xx;bestf=f;end
r=struct('parameters',decode(best,scale),'market',m,'alpha',alpha,'lambda',lambda,...
 'objective',bestf,'history',history,'seconds',toc(timer),'exitflag',flag,'output',info,...
 'settings',settings,'penaltyConvention','Concatenate physical kappa,sigma,rho,v0,am,an,rm,rn,beta for each block. Full matrix entries, including symmetric duplicates and fixed entries; no optimiser scaling.');
% Converged repricing on two grids, using identical settings for both fits.
for which=1:2
 if which==1,q=baseline.parameters;else,q=r.parameters;end
 validateMatrixDiffusion(q);
 vals=cell(1,2);ivs=vals;
 for g=1:2
  sett=struct('nodes',280*g,'cutoff',900+300*(g-1),'tolerance',2e-9/20^(g-1));
  vals{g}=matrixAuditPrices(q,T,m.strikes,th,m.spot,ym,yn,sett);
  ivs{g}=fxImpliedVolatility(vals{g},m.spot,m.strikes,ym,yn,T,th);
 end
 err=100*ivs{2}-m.marketIV;rel=(vals{2}-m.marketPrice)./m.marketPrice;
 z=physical(q);penalty=alpha*(lambda*norm(z,1)+(1-lambda)*sum(z.^2));
 met=struct('relativePriceMSE',mean(rel.^2,'all'),'relativePriceRMSE',sqrt(mean(rel.^2,'all')),...
 'elasticNetPenalty',penalty,'totalObjective',mean(rel.^2,'all')+penalty,...
 'ivRMSEPP',sqrt(mean(err.^2,'all')),'ivMAEPP',mean(abs(err),'all'),...
 'maxIVErrorPP',max(abs(err),[],'all'),'priceRMSE',sqrt(mean((vals{2}-m.marketPrice).^2,'all')),...
 'gridPriceChange',max(abs(vals{2}-vals{1}),[],'all'),'gridIVChangePP',100*max(abs(ivs{2}-ivs{1}),[],'all'));
 assert(all(isfinite(ivs{2}),'all') && met.gridIVChangePP<1e-4,'Repricing did not converge');
 if which==1,r.baseline=met;else,r.metrics=met;r.prices=vals{2};r.impliedVolatility=ivs{2};end
end
r.note='New relative-price elastic-net fit; old IV fit retained separately. Same January matrix quotes/curves. Monte Carlo has not yet been rerun for these new parameters.';
writeJSON(fullfile(out,'calibration.json'),r);disp(r.baseline);disp(r.metrics);
 function f=objective(z)
  try
   q=decode(z,scale);v=matrixAuditPrices(q,T,m.strikes,th,m.spot,ym,yn,settings);
   rel=(v-m.marketPrice)./m.marketPrice;t=physical(q);
   f=mean(rel.^2,'all')+alpha*(lambda*norm(t,1)+(1-lambda)*sum(t.^2));
   if ~isfinite(f) || any(v<=0,'all'),f=1e6;end
  catch,f=1e6;end
 end
 function [c,ceq]=constraints(z)
  q=decode(z,scale);c=[];
  for b=1:2,c=[c;1e-5-trace(q(b).kappa);1e-6-det(q(b).kappa);norm(q(b).rho,2)-.995];end
  ceq=[];
 end
 function stop=record(z,vals,state)
  stop=false;
  if strcmp(state,'iter') && vals.fval<bestf && max(constraints(z))<=1e-7
   best=z;bestf=vals.fval;history=[history;vals.iteration bestf toc(timer)];
   writeJSON(fullfile(out,'checkpoint.json'),struct('parameters',decode(z,scale),'objective',bestf,'history',history));
   fprintf('SAVED iter %d objective %.12g elapsed %.1fs\n',vals.iteration,bestf,toc(timer));
  end
 end
end
function z=physical(p)
z=[];names={'kappa','sigma','rho','v0','am','an','rm','rn','beta'};
for b=1:numel(p),for j=1:numel(names),a=p(b).(names{j});z=[z;a(:)];end,end
end
function [x,lb,ub,scale]=encode(p)
x=[];lb=[];ub=[];scale=[];
for b=1:2
 L=chol(p(b).v0,'lower');s=p(b).sigma;
 v=[p(b).kappa(:);s(1,1);s(1,2);s(2,2);L(1,1);L(2,1);L(2,2);p(b).beta];
 sc=[ones(4,1);.2*ones(3,1);.1*ones(3,1);1];
 low=[-50*ones(4,1);1e-4;-2;1e-4;1e-5;-1;1e-5;.5];
 high=[50*ones(4,1);2;2;2;1;1;1;.5];
 if b==1
  v=[v;p(b).rm(1,1);p(b).rm(1,2);p(b).rm(2,2);p(b).rn(1,1);p(b).rn(1,2);p(b).rn(2,2)];
  sc=[sc;ones(6,1)];low=[low;-3*ones(6,1)];high=[high;3*ones(6,1)];
 else
  rr=p(b).rho/.995;raw=rr/sqrt(1-norm(rr,2)^2);
  v=[v;raw(:);p(b).an(1,1);p(b).an(1,2);p(b).an(2,2)];
  sc=[sc;max(1,norm(raw,'fro'))*ones(4,1);ones(3,1)];low=[low;-inf(4,1);-3*ones(3,1)];high=[high;inf(4,1);3*ones(3,1)];
 end
 x=[x;v./sc];lb=[lb;low./sc];ub=[ub;high./sc];scale=[scale;sc];
end
end
function p=decode(x,scale)
x=x(:).*scale;off=0;
for b=1:2
 z=x(off+1:end);p(b).kappa=reshape(z(1:4),2,2);p(b).sigma=[z(5) z(6);0 z(7)];
 L=[z(8) 0;z(9) z(10)];p(b).v0=L*L.';p(b).beta=z(11);p(b).am=zeros(2);
 if b==1
  p(b).rho=zeros(2);p(b).an=zeros(2);p(b).rm=[z(12) z(13);z(13) z(14)];p(b).rn=[z(15) z(16);z(16) z(17)];off=17;
 else
  raw=reshape(z(12:15),2,2);p(b).rho=.995*raw/sqrt(1+norm(raw,2)^2);p(b).an=[z(16) z(17);z(17) z(18)];p(b).rm=zeros(2);p(b).rn=zeros(2);
 end
end
end
function writeJSON(path,r)
fid=fopen(path,'w');fprintf(fid,'%s',jsonencode(r,'PrettyPrint',true));fclose(fid);
end
