"""Independent standard-library CIR transform and curve-matched FX pricing."""
import cmath
import math


def log_transform(parameters, u, maturity):
    result = 0j
    for k, mean, sigma, rho, v0, am, an, rm, rn in parameters:
        c = k - 1j*u*rho*sigma*(an-am)
        h = (an-am)**2*u*u + 1j*u*(an*an-am*am) - 2j*u*(rn-rm) + 2*rn
        d = cmath.sqrt(c*c + sigma*sigma*h)
        g = (c-d)/(c+d)
        e = cmath.exp(-d*maturity)
        result += k*mean/sigma**2*((c-d)*maturity + 2*cmath.log((1-g)/(1-g*e)))
        result += v0/sigma**2*(c-d)*(1-e)/(1-g*e)
    return result


def half_moment_prices(parameters, maturity, spot, ym, yn, strikes, theta, dx=.05, cutoff=900):
    if not all(math.isfinite(v) and v > 0 for v in (maturity, spot, dx, cutoff)):
        raise ValueError("Maturity, spot, grid step and cutoff must be positive and finite.")
    if len(strikes) != len(theta) or not strikes:
        raise ValueError("Provide matching nonempty strike and option-sign arrays.")
    if any(not math.isfinite(k) or k <= 0 for k in strikes) or any(s not in (-1, 1) for s in theta):
        raise ValueError("Strikes must be positive; option signs must be -1 or +1.")
    if not parameters or any(len(row) != 9 or not all(math.isfinite(v) for v in row) for row in parameters):
        raise ValueError("Each factor needs nine finite parameter values.")
    for row in parameters:
        k, mean, sigma, rho, v0, am, an, rm, rn = row
        if k <= 0 or mean < 0 or sigma <= 0 or abs(rho) > 1 or v0 < 0 or am != 0:
            raise ValueError("Invalid CIR parameters or unsupported nonzero am.")
        for moment in (0, .5, 1):
            c = k-moment*rho*sigma*an
            b = (-moment*moment+moment)*an*an-2*moment*(rn-rm)+2*rn
            delta = c*c+sigma*sigma*b
            explosion = math.inf
            if b < 0:
                if delta < 0:
                    d = math.sqrt(-delta)
                    explosion = 2*(math.pi/2+math.atan(c/d))/d
                elif c < 0:
                    if delta == 0:
                        explosion = -2/c
                    else:
                        d = math.sqrt(delta)
                        explosion = math.log((c-d)/(c+d))/d
            if maturity >= explosion:
                raise ValueError("A required discounted moment explodes before maturity.")
    y0n = yn + log_transform(parameters, 0, maturity).real/maturity
    y0m = ym + log_transform(parameters,-1j,maturity).real/maturity

    def phi(u):
        return cmath.exp(log_transform(parameters, u, maturity)
                         + (-y0n + 1j*u*(y0n-y0m))*maturity)

    n = round(cutoff/dx)
    if n < 2 or n % 2 or abs(n*dx-cutoff) > 1e-8:
        raise ValueError("Simpson grid requires an even integer number of intervals.")
    phi0, phi1 = phi(0).real, phi(-1j).real
    values = [phi(j*dx - .5j)/(j*j*dx*dx+.25) for j in range(n+1)]
    result = []
    for strike, sign in zip(strikes, theta):
        lm = math.log(spot/strike)
        terms = [(cmath.exp(1j*j*dx*lm)*v).real
                 * (1 if j == 0 or j == n else (4 if j % 2 else 2))
                 for j, v in enumerate(values)]
        integral = math.fsum(terms)*dx/3
        result.append((spot*phi1 if sign == 1 else strike*phi0)
                      - math.sqrt(spot*strike)*integral/math.pi)
    return result


def black_price(spot, strike, maturity, ym, yn, vol, sign):
    sd = vol*math.sqrt(maturity)
    d1 = (math.log(spot/strike)+(yn-ym)*maturity)/sd + .5*sd
    d2 = d1-sd
    normal = lambda x: .5*(1+math.erf(x/math.sqrt(2)))
    return sign*(spot*math.exp(-ym*maturity)*normal(sign*d1)
                 - strike*math.exp(-yn*maturity)*normal(sign*d2))


def implied_vol(price, spot, strike, maturity, ym, yn, sign):
    lower = max(sign*(spot*math.exp(-ym*maturity)-strike*math.exp(-yn*maturity)), 0)
    upper = spot*math.exp(-ym*maturity) if sign == 1 else strike*math.exp(-yn*maturity)
    if not math.isfinite(price) or not lower < price < upper:
        raise ValueError("Price is outside strict Black-Scholes bounds.")
    lo, hi = 1e-8, 5.
    if black_price(spot, strike, maturity, ym, yn, hi, sign) < price:
        raise ValueError("Implied volatility exceeds the inversion bracket.")
    for _ in range(80):
        mid = .5*(lo+hi)
        if black_price(spot,strike,maturity,ym,yn,mid,sign) > price:
            hi=mid
        else:
            lo=mid
    return .5*(lo+hi)
