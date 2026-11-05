function V = computeBSPrices(S0, K, vol, Yb_mat, Yt_mat, tau_mat, theta_mat)
% COMPUTEBSPRICES Compute Black-Scholes (Garman-Kohlhagen) option prices
%
% Inputs:
%   S0        - spot FX rate (scalar)
%   K         - strike matrix
%   vol       - volatility matrix (same size as K)
%   Yb_mat    - base currency yields (same size as K)
%   Yt_mat    - term currency yields (same size as K)
%   tau_mat   - times to maturity (same size as K)
%   theta_mat - call/put indicators +1/-1 (same size as K)
%
% Output:
%   V - option price matrix (same size as K)

d1 = (log(S0./K)+(Yt_mat-Yb_mat+0.5*vol.^2).*tau_mat)./(vol.*sqrt(tau_mat));
d2 = d1 - vol.*sqrt(tau_mat);
V = theta_mat.*(S0*exp(-Yb_mat.*tau_mat).*(0.5*erfc(-theta_mat.*d1/sqrt(2))) ...
               -K.*exp(-Yt_mat.*tau_mat).*(0.5*erfc(-theta_mat.*d2/sqrt(2))));

end
