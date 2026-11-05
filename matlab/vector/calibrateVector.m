function r=calibrateVector(N,maxIterations,options)
% New local fit using the corrected curve-matched pricer and unique quotes.
% Objective: relative-price MSE + physical-parameter elastic-net penalty.
if nargin<1,N=1;end
if nargin<2,maxIterations=150;end
if nargin<3,options=struct;end
assert(any(N==[1 2]));
validateattributes(maxIterations,{'numeric'},{'scalar','integer','positive'});
if ~isfield(options,'OutputDir'),options.OutputDir=fullfile(hestonRoot(),'results',sprintf('vector_N%d',N));end
if ~isfield(options,'MaxFunctionEvaluations'),options.MaxFunctionEvaluations=16000;end
if ~isfield(options,'Alpha'),options.Alpha=1e-4;end
if ~isfield(options,'Lambda'),options.Lambda=.5;end
validateattributes(options.Alpha,{'numeric'},{'scalar','nonnegative','finite'});
validateattributes(options.Lambda,{'numeric'},{'scalar','>=',0,'<=',1,'finite'});
d=loadExample(sprintf('vector_N%d',N));p=d.parameters;
if isfield(options,'InitialParameters'),p=options.InitialParameters;end
assert(isequal(size(p),[N 9]));validateVectorParameters(p,max(d.tau));
maxvar=max(d.marketIVDecimal.^2,[],'all');
lo=repmat([.01 0 .01 -1 0 0 0 -1 -1],N,1);
hi=repmat([2 maxvar 1 1 maxvar 0 2 1 1],N,1);
lo(1,7)=1;hi(1,7)=1;
assert(all(p>=lo & p<=hi,'all'),'Initial parameters are outside the stated bounds.');
scale=ones(N,9);scale(:,[2 5])=.01;free=find(lo<hi);
x0=p(free)./scale(free);lb=lo(free)./scale(free);ub=hi(free)./scale(free);
settings=struct('step',.05,'cutoff',900);alpha=options.Alpha;lambda=options.Lambda;
initialObjective=objective(x0);assert(initialObjective<1e6,'Invalid initial prices.');
opts=optimoptions('fmincon','Algorithm','sqp','Display','iter', ...
 'MaxIterations',maxIterations,'MaxFunctionEvaluations',options.MaxFunctionEvaluations, ...
 'FiniteDifferenceStepSize',1e-4,'OptimalityTolerance',1e-7,'StepTolerance',1e-9);
timer=tic;[x,value,flag,info]=fmincon(@objective,x0,[],[],[],[],lb,ub,[],opts);
if ~isfinite(value) || value>initialObjective,x=x0;value=initialObjective;end
fitted=decode(x);coarse=price(fitted,settings);
fine=price(fitted,struct('step',.025,'cutoff',1800));
gridChange=max(abs(fine-coarse),[],'all');
assert(gridChange<1e-7,'Vector:Quadrature','Final price grid did not converge.');
iv=fxImpliedVolatility(fine,d.spot,d.strikes,d.eurYield,d.usdYield,d.tau,d.theta.');
r=struct('N',N,'parameters',fitted,'initialParameters',p,'initialObjective',initialObjective, ...
 'objective',value,'alpha',alpha,'lambda',lambda,'exitflag',flag,'output',info, ...
 'seconds',toc(timer),'settings',settings,'gridPriceChange',gridChange,'prices',fine, ...
 'impliedVolatilityDecimal',iv,'priceRMSE',sqrt(mean((fine-d.marketPrices).^2,'all')), ...
 'ivRMSEDecimal',sqrt(mean((iv-d.marketIVDecimal).^2,'all')), ...
 'note','New bounded local optimisation on 187 unique quotes; not a reconstruction of the historical optimiser trajectory.');
writeResult(fullfile(options.OutputDir,'calibration.json'),r);
 function q=decode(z)
  q=p;q(free)=z.*scale(free);
 end
 function values=price(q,grid)
  values=vectorPrices(q,d.tau,d.strikes,d.theta.',d.spot,d.eurYield,d.usdYield,grid);
 end
 function f=objective(z)
  try
   q=decode(z);v=price(q,settings);relative=(v-d.marketPrices)./d.marketPrices;
   f=mean(relative.^2,'all')+alpha*(lambda*sum(abs(q),'all')+(1-lambda)*sum(q.^2,'all'));
   if any(v<=0,'all') || ~isfinite(f),f=1e6;end
  catch,f=1e6;end
 end
end
