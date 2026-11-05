function maxError=checkCorrectedLinearisation(p)
% Independent matrix-exponential check at noncommuting fitted parameters.
% Short maturities avoid ill-conditioning; beta=.5 makes exp(log det) single-valued.
maxError=0;
for b=1:2
 q=p(b);assert(abs(q.beta-.5)<1e-12);A=q.an;Q=q.sigma.'*q.sigma;
 for T=[.01 .1 .25]
  for xi=[0 .4 2 5]
   C=q.kappa-1i*xi*q.sigma.'*q.rho*A;
   H=(xi^2+1i*xi)*(A*A)-2i*xi*(q.rn-q.rm)+2*q.rn;
   M=expm([-C.' -H;-Q C]*T/2);U=M(3:4,3:4);V=M(1:2,3:4);F=V/U;
   direct=exp(q.beta*trace(C)*T+trace(F*q.v0))/det(U);
   ode=exp(matrixAuditTransform(q,xi,T,1e-11));
   maxError=max(maxError,abs(direct-ode));
  end
 end
end
assert(maxError<1e-7,'Riccati linearisation mismatch');
end
