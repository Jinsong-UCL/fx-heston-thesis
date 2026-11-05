function exportLanguageParity(outputFile, runSimulations)
% Export independently computed MATLAB values for verify_language_parity.py.
% Results are generated locally, not committed as a second set of inputs.
if nargin<1,outputFile=fullfile(hestonRoot(),'results','language_parity','matlab.json');end
if nargin<2,runSimulations=true;end
folder=fileparts(outputFile);
u=[0 -1i .4-.5i 2 5];times=[.25 1 5];
r=struct('runtime',version,'frequenciesReal',real(u),'frequenciesImag',imag(u),'maturities',times);
for N=1:2
 d=loadExample(sprintf('vector_N%d',N));
 T=repmat(d.tau(:),1,size(d.strikes,2));ym=repmat(d.eurYield(:),1,size(T,2));yn=repmat(d.usdYield(:),1,size(T,2));
 th=repmat(d.theta(:).',size(T,1),1);
 v=vectorPrices(d.parameters,T,d.strikes,th,d.spot,ym,yn,struct('step',.025,'cutoff',1800));
 coarse=vectorPrices(d.parameters,T,d.strikes,th,d.spot,ym,yn,struct('step',.05,'cutoff',900));
 L=complex(zeros(numel(times),numel(u)));for j=1:numel(times),L(j,:)=vectorLogTransform(d.parameters,u,times(j));end
 q=struct('parameters',d.parameters,'prices',v,'logTransformReal',real(L),'logTransformImag',imag(L), ...
  'initialObjective',mean(((coarse-d.marketPrices)./d.marketPrices).^2,'all')+1e-4*(.5*sum(abs(d.parameters),'all')+.5*sum(d.parameters.^2,'all')));
 if runSimulations
  for scheme={'EM','QE'}
   s=runVectorSimulation(N,struct('Scheme',scheme{1},'Paths',20000,'Steps',250,'Blocks',40,'OutputDir',fullfile(folder,sprintf('matlab_vector_N%d',N))));
   summary=struct('prices',s.price,'priceSE',s.se,'referencePrices',s.reference,'cfReal',real(s.cf),'cfImag',imag(s.cf), ...
    'ecfReal',real(s.ecf),'ecfImag',imag(s.ecf),'cfRealSE',s.ecfRealSE,'cfImagSE',s.ecfImagSE, ...
    'diagnostics',s.diagnostics,'diagnosticsSE',s.diagnosticsSE,'terminalVariance',s.terminalVariance,'terminalVarianceSE',s.terminalVarianceSE);
   q.(scheme{1})=summary;
  end
 end
 r.(sprintf('vector_N%d',N))=q;
end
d=loadExample('matrix');m=d.market;T=repmat(m.tau(:),1,9);ym=repmat(m.eurYield(:),1,9);yn=repmat(m.usdYield(:),1,9);th=repmat([-1 -1 -1 -1 -1 1 1 1 1],numel(m.tau),1);
settings=struct('nodes',560,'cutoff',1200,'tolerance',1e-10);
v=matrixAuditPrices(d.parameters,T,m.strikes,th,m.spot,ym,yn,settings);
L=zeros(numel(times),numel(u));for b=1:numel(d.parameters),L=L+matrixAuditTransform(d.parameters(b),u,times,1e-10,'sde');end
coarse=matrixAuditPrices(d.initialParameters,T,m.strikes,th,m.spot,ym,yn,struct('nodes',210,'cutoff',900,'tolerance',1e-10));
physical=[];fields={'kappa','sigma','v0','rho','am','an','rm','rn','beta'};
for b=1:numel(d.initialParameters)
 for j=1:numel(fields),z=d.initialParameters(b).(fields{j});physical=[physical;z(:)];end %#ok<AGROW>
end
r.matrix=struct('parameters',d.parameters,'prices',v,'logTransformReal',real(L),'logTransformImag',imag(L), ...
 'initialObjective',mean(((coarse-m.marketPrice)./m.marketPrice).^2,'all')+1e-4*(.5*sum(abs(physical))+.5*sum(physical.^2)));
if runSimulations
 r.matrix.simulation=runMatrixSimulation(struct('Paths',20000,'Steps',250,'OutputDir',fullfile(folder,'matlab_matrix')));
end
writeResult(outputFile,r);
fprintf('Exported MATLAB language parity values to %s\n',outputFile);
end
