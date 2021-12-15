import numpy as np
import scipy.linalg as spla


ex = np.array([1, 0])
exex = np.outer(ex, ex)
ey = np.array([0, 1])
eyey = np.outer(ey, ey)
I = np.eye(2)
I4 = np.eye(4, dtype=np.complex128)


def eigvalsM(eps, b=0.):
    n = np.sqrt(eps - b ** 2)
    return np.array([-n, -n, n, n])


def eigvecsM(eps, b=0.):
    sqrtepsb = np.sqrt(eps - b ** 2)
    c1 = sqrtepsb
    c2 = eps * sqrtepsb / (b ** 2 - eps)
    c3 = sqrtepsb / (2 * eps)
    c4 = 0.5 / sqrtepsb
    return np.array([[[0, -c1, 0, c1],
                     [c2, 0, -c2, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0]],
                    [[0, -c3, 0, 0.5],
                     [-c4, 0, 0.5, 0],
                     [0, c3, 0, 0.5],
                     [c4, 0, 0.5, 0]]])


def propagator_layer(f, eps, d, b=0):
    """Propagator, or evolution operator, or transfer matrix of
    a uniform layer with permittivity eps and thickness d
    at frequency f.
    """
    scalar_input = np.isscalar(f)
    freqs = np.atleast_1d(f)
    N = freqs.shape[0]
    res = np.empty((N, 4, 4), dtype=np.complex_)
    w = eigvalsM(eps, b)
    vr = eigvecsM(eps, b)
    for i in range(N):
        ik0d = 2j * np.pi * freqs[i] * d
        res[i, :, :] = np.linalg.multi_dot(
            [vr[0], np.diag(np.exp(ik0d * w)), vr[1]]
        )
    if scalar_input:
        return res[0]
    return res


def propagator_grating(f, eps1, eps2, D, b=0):
    scalar_input = np.isscalar(f)
    N = D.shape[0]
    if scalar_input:
        propagator = propagator_layer(f, eps1, D[0], b)
        for n in range(1, N):
            eps = eps1 if n % 2 == 0 else eps2
            propagator = np.dot(propagator_layer(f, eps, D[n], b), propagator)
        return propagator
    propagator = propagator_layer(f, eps1, D[0], b)
    for n in range(1, N):
        eps = eps1 if n % 2 == 0 else eps2
        prop_layer = propagator_layer(f, eps, D[n], b)
        for i in range(propagator.shape[0]):
            propagator[i] = np.dot(prop_layer[i], propagator[i])
    return propagator


def gamma(eps, b=0):
    return np.diag([1 / np.sqrt(eps - b ** 2),
                    np.sqrt(eps - b ** 2) / eps])


def operator_r(propagator, eps_left=1, eps_right=None, b=0):
    """Reflection operator of a multilayer characterized by propagator
    surrounded by media with eps_left at left and eps_right at right.
    """
    if not eps_right:
        eps_right = eps_left
    gamma_left = gamma(eps_left, b)
    gamma_right = gamma(eps_right, b)
    factor1 = (np.bmat([gamma_right, -I]).
               dot(propagator).
               dot(np.bmat([I, -gamma_left]).T))
    factor2 = (np.bmat([gamma_right, -I]).
               dot(propagator).
               dot(np.bmat([I, gamma_left]).T))
    return spla.inv(factor1).dot(factor2)


def operator_t(propagator, eps_left=1, eps_right=None, b=0):
    """Transmission operator of a multilayer characterized by propagator
    surrounded by media with eps_left at left and eps_right at right.
    """
    if not eps_right:
        eps_right = eps_left
    gamma_left = gamma(eps_left, b)
    gamma_right = gamma(eps_right, b)
    return 2 * spla.inv((np.bmat([gamma_left, I]).
                dot(spla.inv(propagator)).
                dot(np.bmat([I, gamma_right]).T))).dot(gamma_left)
