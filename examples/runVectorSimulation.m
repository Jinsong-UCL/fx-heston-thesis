function r=runVectorSimulation(N,options)
% Simulate a saved or newly fitted vector model; EM or QE variance stepping.
if nargin<1,N=1;end
if nargin<2,options=struct;end
assert(any(N==[1 2]));
if ~isfield(options,'T'),options.T=1;end
if ~isfield(options,'Steps'),options.Steps=250;end
if ~isfield(options,'Paths'),options.Paths=20000;end
if ~isfield(options,'Blocks'),options.Blocks=40;end
if ~isfield(options,'Scheme'),options.Scheme='QE';end
if ~isfield(options,'Seed'),options.Seed=81473+10000*N;end
if ~isfield(options,'OutputDir'),options.OutputDir=fullfile(hestonRoot(),'results',sprintf('vector_N%d',N));end
d=loadExample(sprintf('vector_N%d',N));p=d.parameters;
if isfield(options,'Parameters'),p=options.Parameters;end
assert(isequal(size(p),[N 9]));validateVectorParameters(p,options.T);
row=find(abs(d.tau-options.T)<1e-10,1);assert(~isempty(row),'Choose a maturity present in the example.');
cols=[1 3 6 9 11];K=d.strikes(row,cols);th=d.theta(cols).';
reference=vectorPrices(p,options.T,K,th,d.spot,d.eurYield(row),d.usdYield(row),struct('step',.025,'cutoff',1800));
model=struct('kind','vector','p',p(:),'N',N,'T',options.T,'strikes',K,'theta',th, ...
 'S0',d.spot,'Ym',d.eurYield(row),'Yn',d.usdYield(row),'xi',0:2:100,'referencePrices',reference);
cfg=struct('paths',options.Paths,'blocks',options.Blocks,'steps',options.Steps,'scheme',options.Scheme,'seed',options.Seed);
r=submissionVectorCase(model,cfg);
summary=struct('N',N,'parameters',p,'T',options.T,'settings',cfg, ...
 'strikes',K,'theta',th,'referencePrices',reference,'prices',r.price,'priceSE',r.se, ...
 'cfReal',real(r.cf),'cfImag',imag(r.cf),'ecfReal',real(r.ecf),'ecfImag',imag(r.ecf), ...
 'cfRealSE',r.ecfRealSE,'cfImagSE',r.ecfImagSE,'xi',model.xi, ...
 'expectedDiscount',exp(-model.Yn*model.T),'expectedDiscountedSpot',model.S0*exp(-model.Ym*model.T), ...
 'discountMean',r.diagnostics(1),'discountedSpotMean',r.diagnostics(2),'seconds',r.seconds);
writeResult(fullfile(options.OutputDir,['simulation_' cfg.scheme '.json']),summary);
if ~exist(options.OutputDir,'dir'),mkdir(options.OutputDir);end
save(fullfile(options.OutputDir,['simulation_' cfg.scheme '.mat']),'r');
fprintf('Vector N=%d %s: ATM reference %.9f, MC %.9f, sampling SE %.9f\n',N,cfg.scheme,reference(3),r.price(3),r.se(3));
end
