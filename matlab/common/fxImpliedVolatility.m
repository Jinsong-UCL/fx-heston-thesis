function vol = fxImpliedVolatility(prices,S0,K,Yb,Yt,tau,theta)
% Vectorized bracketed inversion; invalid prices are rejected, never clipped.
lower=max(theta.*(S0*exp(-Yb.*tau)-K.*exp(-Yt.*tau)),0);
upper=(theta>0).*S0.*exp(-Yb.*tau)+(theta<0).*K.*exp(-Yt.*tau);
if any(~isfinite(prices) | prices<=lower | prices>=upper,'all')
    error('Calibration:InvalidPrice','Prices outside strict Black-Scholes bounds.');
end
lo=zeros(size(K)); hi=3*ones(size(K));
for j=1:40
    vol=(lo+hi)/2;
    v=computeBSPrices(S0,K,vol,Yb,Yt,tau,theta);
    ix=v<prices; lo(ix)=vol(ix); hi(~ix)=vol(~ix);
end
vol=(lo+hi)/2;
end
