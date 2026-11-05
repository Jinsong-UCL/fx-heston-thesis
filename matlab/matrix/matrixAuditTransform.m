function logPhi = matrixAuditTransform(p, xi, maturities, tolerance, order)
% Affine log contribution for the declared SDE; saved order is diagnostic only.
% Frequencies are integrated together with an exact sparse Jacobian.
if nargin < 4, tolerance = 2e-7; end
if nargin<5, order='sde'; end
xi = reshape(xi,1,1,[]); nf = numel(xi);
A = p.an-p.am; Q = p.sigma.'*p.sigma;
if strcmp(order,'sde')
    coupling=p.sigma.'*p.rho*A;
elseif strcmp(order,'saved')
    coupling=p.rho*p.sigma.'*A;
else
    error('Unknown covariance order');
end
C = p.kappa - 1i*xi.*coupling;
H = (xi.^2+1i*xi).*(A*A) ...
    - 2i*xi.*(p.rn-p.rm) + 2*p.rn;
% Symmetrization leaves the contraction with symmetric variance unchanged.
H = (H+pagetranspose(H))/2;
times = unique([0; maturities(:)]);
opts = odeset('RelTol',tolerance,'AbsTol',tolerance*0.01, ...
    'Jacobian',@jacobian);
[tt,yy] = ode15s(@rhs,times,complex(zeros(5*nf,1)),opts);
if tt(end) < times(end), error('Riccati:Explosion','Riccati integration failed.'); end
[~,ix] = ismember(maturities(:),tt);
if any(ix==0), error('Riccati:Times','Missing maturity.'); end
state = reshape(yy(ix,:).',5,nf,[]);
logPhi = reshape(state(5,:,:) + sum(state(1:4,:,:).*p.v0(:),1),nf,[]).';
if any(~isfinite(logPhi),'all'), error('Riccati:Nonfinite','Nonfinite transform.'); end

    function dy = rhs(~,y)
        z = reshape(y,5,nf); B = reshape(z(1:4,:),2,2,nf);
        F = -0.5*(pagemtimes(B,C)+pagemtimes(pagetranspose(C),B) ...
            -pagemtimes(pagemtimes(B,Q),B)+H);
        dy = [reshape(F,4,nf); p.beta*sum(Q(:).*z(1:4,:),1)];
        dy = dy(:);
    end
    function J = jacobian(~,y)
        z = reshape(y,5,nf); B = reshape(z(1:4,:),2,2,nf);
        W = pagetranspose(C)-pagemtimes(B,Q);
        vals = complex(zeros(5,5,nf));
        for k=1:4
            D = zeros(2); D(k)=1;
            dF = -0.5*(pagemtimes(D,pagetranspose(W))+pagemtimes(W,D));
            vals(1:4,k,:) = reshape(dF,4,1,nf);
        end
        vals(5,1:4,:) = repmat(p.beta*Q(:).',1,1,nf);
        [rr,cc] = ndgrid(1:5,1:5);
        rr = rr + reshape(5*(0:nf-1),1,1,[]);
        cc = cc + reshape(5*(0:nf-1),1,1,[]);
        J = sparse(rr(:),cc(:),vals(:),5*nf,5*nf);
    end
end
