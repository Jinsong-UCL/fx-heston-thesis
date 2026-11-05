function r=runMatrixSimulation(options)
% Simulate the saved or newly calibrated two-block Wishart model.
if nargin<1,options=struct;end
if ~isfield(options,'T'),options.T=1;end
if ~isfield(options,'Steps'),options.Steps=250;end
if ~isfield(options,'Paths'),options.Paths=20000;end
if ~isfield(options,'OutputDir'),options.OutputDir=fullfile(hestonRoot(),'results','matrix');end
d=loadExample('matrix');p=d.parameters;
if isfield(options,'Parameters'),p=options.Parameters;end
r=simulateCorrectedMatrix(p,d.market,options.Steps,options.T,options.Paths);
writeResult(fullfile(options.OutputDir,'simulation.json'),r);
end
