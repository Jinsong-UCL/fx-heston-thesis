"""Independent deterministic checks of the terms-currency thesis framework.

Uses only the Python standard library; all products are bilinear, never
conjugate-linear. It does not change calibration parameters or saved results.
"""
import cmath
import json
import math
from pathlib import Path


def transpose(a):
    return [list(row) for row in zip(*a)]


def product(a, b):
    return [[sum(x*y for x, y in zip(row, col))
             for col in zip(*b)] for row in a]


def add(a, b):
    return [[x+y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, s):
    return [[s*x for x in row] for row in a]


def eye(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def dot(a, b):
    return sum(x*y for ar, br in zip(a, b) for x, y in zip(ar, br))


def error(a, b):
    return max(abs(x-y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def sqrt2(a):
    root_det = math.sqrt(a[0][0]*a[1][1]-a[0][1]*a[1][0])
    denom = math.sqrt(trace(a)+2*root_det)
    return scale(add(a, scale(eye(2), root_det)), 1/denom)


def inverse2(a):
    d = a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return scale([[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]], 1/d)


def expm(a):
    n = len(a)
    norm = max(sum(abs(x) for x in row) for row in a)
    squarings = max(0, math.ceil(math.log2(max(norm, 1))))
    small = scale(a, 2**(-squarings))
    out = eye(n)
    term = eye(n)
    for k in range(1, 100):
        term = scale(product(term, small), 1/k)
        out = add(out, term)
        if max(abs(x) for row in term for x in row) < 1e-17:
            break
    else:
        raise AssertionError("Matrix exponential series did not converge")
    for _ in range(squarings):
        out = product(out, out)
    return out


variance = [[.4, .07], [.07, .15]]
an = [[.7, .1], [.1, 1.2]]
am = [[.08, .01], [.01, -.02]]
loading = add(an, scale(am, -1))
sigma = [[.5, .04], [-.08, .6]]
rho = [[-.3, .12], [-.07, .2]]
kappa = [[1.5, .12], [-.04, .9]]
rn = [[.08, .01], [.01, .03]]
rm = [[.03, .002], [.002, .02]]
root = sqrt2(variance)
complement = sqrt2(add(eye(2), scale(product(rho, transpose(rho)), -1)))
q = product(transpose(sigma), sigma)
indices = [(0, 0), (0, 1), (1, 1)]
bad_basis = []
good_basis = []
for k in range(8):
    w = [[0., 0.], [0., 0.]]
    b = [[0., 0.], [0., 0.]]
    target, j = (w, k) if k < 4 else (b, k-4)
    target[j//2][j % 2] = 1
    z = add(product(w, transpose(rho)), product(b, complement))
    noise = product(product(root, z), sigma)
    dv = scale(add(noise, transpose(noise)), .5)
    component = [dv[i][j] for i, j in indices]
    bad_basis.append(component + [dot(product(loading, root), w)])
    good_basis.append(component + [dot(product(root, loading), w)])
actual = product(transpose(good_basis), good_basis)
wrong = product(transpose(bad_basis), bad_basis)
expected = [[0.]*4 for _ in range(4)]
for u, (i, j) in enumerate(indices):
    for v, (k, l) in enumerate(indices):
        expected[u][v] = .25*(variance[i][k]*q[j][l]
                              + variance[i][l]*q[j][k]
                              + variance[j][k]*q[i][l]
                              + variance[j][l]*q[i][k])
mixed = product(product(product(transpose(sigma), rho), loading), variance)
mixed = scale(add(mixed, transpose(mixed)), .5)
for u, (i, j) in enumerate(indices):
    expected[u][3] = expected[3][u] = mixed[i][j]
expected[3][3] = trace(product(product(loading, loading), variance))
covariance_error = error(actual, expected)
wrong_loading_error = error(wrong, expected)
assert covariance_error < 1e-14
assert wrong_loading_error > 1e-4

# Check the matrix-exponential solution against direct integration of Riccati.
xi = 1.7
tau = .7
beta = .5
c = add(kappa, scale(product(product(transpose(sigma), rho), loading), -1j*xi))
h = add(scale(product(loading, loading), xi*xi+1j*xi),
        add(scale(add(rn, scale(rm, -1)), -2j*xi), scale(rn, 2)))
block = [scale(transpose(c), -1)[i] + scale(h, -1)[i] for i in range(2)]
block += [scale(q, -1)[i] + c[i] for i in range(2)]
e = expm(scale(block, tau/2))
v = [row[2:] for row in e[:2]]
u = [row[2:] for row in e[2:]]
from_blocks = product(v, inverse2(u))


def rhs(phi):
    return scale(add(add(product(phi, c), product(transpose(c), phi)),
                     add(scale(product(product(phi, q), phi), -1), h)), -.5)


phi = [[0j, 0j], [0j, 0j]]
psi = 0j
steps = 1400
dt = tau/steps
for _ in range(steps):
    k1 = rhs(phi)
    p2 = add(phi, scale(k1, dt/2)); k2 = rhs(p2)
    p3 = add(phi, scale(k2, dt/2)); k3 = rhs(p3)
    p4 = add(phi, scale(k3, dt)); k4 = rhs(p4)
    psi += beta*dt/6*(trace(product(q, phi))+2*trace(product(q, p2))
                     + 2*trace(product(q, p3))+trace(product(q, p4)))
    phi = add(phi, scale(add(add(k1, scale(k2, 2)),
                             add(scale(k3, 2), k4)), dt/6))
riccati_error = error(from_blocks, phi)
det_u = u[0][0]*u[1][1]-u[0][1]*u[1][0]
psi_blocks = beta*(tau*trace(c)-2*cmath.log(det_u))
psi_error = abs(psi-psi_blocks)
assert riccati_error < 1e-12
assert psi_error < 1e-12

# A quotient must be formed under ONE pricing measure. Test generic
# non-collinear loadings, including the drift change from the numeraire.
li=[.2,-.1,.07];lj=[-.03,.18,.12];ri=.014;rj=.027;r1=.005
normi=sum(x*x for x in li);normj=sum(x*x for x in lj)
cross=sum(x*y for x,y in zip(li,lj));normdiff=sum((y-x)**2 for x,y in zip(li,lj))
mui=ri-r1+normi-cross;muj=rj-r1
quotient_drift=muj-mui+normi-cross
terms_drift_error=abs(quotient_drift-(rj-ri))
base_drift=rj-ri+normdiff
inverse_drift=-base_drift+normdiff
inverse_drift_error=abs(inverse_drift-(ri-rj))
assert max(terms_drift_error,inverse_drift_error)<1e-15

# The xi=-i discounted FX moment under Q^j is the foreign bond under Q^i.
# Its C coefficient changes to kappa_i=kappa_j-sigma^T rho A and H=2 R_i.
def coefficients(frequency):
    cc=add(kappa,scale(product(product(transpose(sigma),rho),loading),-1j*frequency))
    hh=add(scale(product(loading,loading),frequency*frequency+1j*frequency),
           add(scale(add(rn,scale(rm,-1)),-2j*frequency),scale(rn,2)))
    return cc,hh
ci,hi=coefficients(-1j)
base_kappa=add(kappa,scale(product(product(transpose(sigma),rho),loading),-1))
foreign_moment_error=max(error(ci,base_kappa),error(hi,scale(rm,2)))
assert foreign_moment_error<1e-14

# The thesis uses A2 = v0 * A2_Cui; its rescaled variance term is -v0 A1/A2.
scalar_c = 1.4-1j*xi*(-.3)*.5
scalar_h = xi*xi+1j*xi
scalar_f = cmath.sqrt(scalar_c**2+.5**2*scalar_h)
a1 = scalar_h*cmath.sinh(scalar_f*tau/2)
a2 = scalar_f*cmath.cosh(scalar_f*tau/2)+scalar_c*cmath.sinh(scalar_f*tau/2)
scalar_e = expm(scale([[-scalar_c, -scalar_h], [-.5**2, scalar_c]], tau/2))
scalar_error = abs(scalar_e[0][1]/scalar_e[1][1]+a1/a2)
assert scalar_error < 1e-14

report = dict(covariance_error=covariance_error,
              original_wrong_dot_loading_error=wrong_loading_error,
              riccati_matrix_exponential_error=riccati_error,
              scalar_affine_log_error=psi_error,
              terms_currency_quotient_drift_error=terms_drift_error,
              base_currency_inverse_drift_error=inverse_drift_error,
              foreign_moment_generator_error=foreign_moment_error,
              scalar_heston_limit_error=scalar_error)
print(json.dumps(report, indent=2))
