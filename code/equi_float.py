"""Float exploration: equilateral triangle of area S=1, lowest Robin eigenvalue (KLV / McCartin).
Parameter M>0: t = tanh M, K = M + artanh(tanh(M)/2), beta = -alpha = t K / 3^{1/4},
lambda0 = -4 K^2/sqrt3 = -4 beta^2 / tanh(M)^2."""
import numpy as np
from scipy.optimize import brentq
h = 3 ** 0.25
def K_of_M(M): return M + np.arctanh(np.tanh(M) / 2)
def beta_of_M(M): return np.tanh(M) * K_of_M(M) / h
def M_of_beta(beta):
    hi = 1.0
    while beta_of_M(hi) < beta: hi *= 2
    return brentq(lambda M: beta_of_M(M) - beta, 0, hi, xtol=1e-300, rtol=1e-15)
def lam0(beta):
    M = M_of_beta(beta); return -4 * K_of_M(M) ** 2 / np.sqrt(3)
def lam0_and_deriv(beta):
    """lambda0 and dlambda0/dalpha (alpha=-beta)."""
    M = M_of_beta(beta); t = np.tanh(M); K = K_of_M(M)
    dt = 1 - t * t
    dK = 1 + 0.5 * dt / (1 - t * t / 4)
    dl = -8 * K * dK / np.sqrt(3)
    dalpha = -(dt * K + t * dK) / h
    return -4 * K ** 2 / np.sqrt(3), dl / dalpha
if __name__ == '__main__':
    for b in (0.01, 0.1, 1, 2, 5, 10, 20):
        l, d = lam0_and_deriv(b)
        print(b, l, d, l + 4 * b * b)
