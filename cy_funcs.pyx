# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython
import scipy.linalg as spla
from libc.math cimport sqrt, exp


ex = np.array([1, 0])
exex = np.outer(ex, ex)
ey = np.array([0, 1])
eyey = np.outer(ey, ey)
I = np.eye(2)
I4 = np.eye(4, dtype=np.complex128)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef matrix_M(double eps, double b=0):
    M = np.empty((4, 4), dtype=np.double)
    cdef double[:, ::1] M_view = M
    M_view[:, :] = 0
    M_view[0, 2] = eps - b ** 2
    M_view[1, 3] = eps
    M_view[2, 0] = 1
    M_view[3, 1] = 1 - (b ** 2 / eps)
    return M


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef eigvalsM(double eps, double b=0.):
    cdef double n = sqrt(eps - b ** 2)
    return np.array([-n, -n, n, n], dtype=np.double)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef eigvecsM(double eps, double b=0.):
    cdef double sqrtepsb = sqrt(eps - b ** 2)
    cdef double c1 = sqrtepsb
    cdef double c2 = eps * sqrtepsb / (b ** 2 - eps)
    cdef double c3 = sqrtepsb / (2 * eps)
    cdef double c4 = 0.5 / sqrtepsb
    return np.array([[[0, -c1, 0, c1],
                     [c2, 0, -c2, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0]],
                    [[0, -c3, 0, 0.5],
                     [-c4, 0, 0.5, 0],
                     [0, c3, 0, 0.5],
                     [c4, 0, 0.5, 0]]], dtype=np.double)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef propagator_layer(f, double eps, double d, double b=0):
    """Propagator, or evolution operator, or transfer matrix of
    a uniform layer with permittivity eps and thickness d
    at frequency f.
    """
    scalar_input = np.isscalar(f)
    cdef double[::1] freqs = np.atleast_1d(f)
    cdef Py_ssize_t i, N = freqs.shape[0]
    cdef double complex ik0d
    cdef np.ndarray[dtype=np.complex_t, ndim=3] res
    res = np.empty((N, 4, 4), dtype=np.complex_)
    cdef np.ndarray[dtype=np.double_t, ndim=1] w = eigvalsM(eps, b)
    cdef np.ndarray[dtype=np.double_t, ndim=3] vr = eigvecsM(eps, b)
    for i in range(N):
        ik0d = 2j * np.pi * freqs[i] * d
        res[i, :, :] = np.linalg.multi_dot(
            [vr[0], np.diag(np.exp(ik0d * w)), vr[1]]
        )
    if scalar_input:
        return res[0]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def propagator_grating(f, double eps1, double eps2, double[:] D, double b=0):
    scalar_input = np.isscalar(f)
    cdef size_t n, N = D.shape[0]
    if scalar_input:
        propagator = propagator_layer(f, eps1, D[0], b)
        for n in range(1, N):
            eps = eps1 if n % 2 == 0 else eps2
            propagator = np.dot(propagator_layer(f, eps, D[n], b), propagator)
        return propagator
    cdef np.ndarray[np.complex128_t, ndim=3] prop_layer
    propagator = propagator_layer(f, eps1, D[0], b)
    for n in range(1, N):
        eps = eps1 if n % 2 == 0 else eps2
        prop_layer = propagator_layer(f, eps, D[n], b)
        for i in range(propagator.shape[0]):
            propagator[i] = np.dot(prop_layer[i], propagator[i])
    return propagator


cpdef gamma(eps, b=0):
    return np.diag([1 / np.sqrt(eps - b ** 2),
                    np.sqrt(eps - b ** 2) / eps])


cpdef operator_r(propagator, eps_left=1, eps_right=None, b=0):
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


cpdef operator_t(propagator, eps_left=1, eps_right=None, b=0):
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
