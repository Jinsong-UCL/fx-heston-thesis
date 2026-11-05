function validateMatrixDiffusion(blocks)
% Guard for future SDE-order calibration/simulation. Formal diagnostics bypass
% this guard explicitly; projected paths are not validation of rejected fits.
for b=1:numel(blocks)
 p=blocks(b);d=size(p.v0,1);
 assert(norm(p.v0-p.v0.','fro')<1e-12 && min(eig(p.v0))>=-1e-12,'MatrixAudit:InitialState','Initial variance must be positive semidefinite.');
 assert(rank(p.sigma)==d,'MatrixAudit:Rank','This guard assumes full-rank sigma.');
 assert(p.beta>=(d-1)/4,'MatrixAudit:Beta','Block %d beta=%g violates beta >= (d-1)/4 for this noise scaling.',b,p.beta);
 assert(norm(p.rho,2)<=1+1e-12,'MatrixAudit:Correlation','Block %d rho is not a Brownian contraction.',b);
 assert(norm(p.am,'fro')==0,'MatrixAudit:Numeraire','This implementation assumes am=0.');
 assert(norm(p.an-p.an.','fro')<1e-12,'MatrixAudit:LoadingSymmetry','FX loading an must be symmetric.');
end
end
