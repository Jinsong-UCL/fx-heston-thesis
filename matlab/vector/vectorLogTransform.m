function out=vectorLogTransform(p,u,T)
% Raw discounted log transform, before deterministic curve matching.
% Parameter columns: kappa, vbar, sigma, rho, v0, am, an, rm, rn.
u=reshape(u,1,[]);
k=p(:,1);v=p(:,2);s=p(:,3);rho=p(:,4);v0=p(:,5);
am=p(:,6);an=p(:,7);rm=p(:,8);rn=p(:,9);
c=k-1i*u.*(rho.*s.*(an-am));
h=(an-am).^2.*u.^2+1i*u.*(an.^2-am.^2)-2i*u.*(rn-rm)+2*rn;
d=sqrt(c.^2+s.^2.*h);g=(c-d)./(c+d);e=exp(-d*T);
out=(k.*v./s.^2).' *((c-d)*T+2*log((1-g)./(1-g.*e))) ...
    +(v0./s.^2).'*((c-d).*(1-e)./(1-g.*e));
assert(all(isfinite(out)),'Vector:Nonfinite','Nonfinite transform.');
end
