import numpy as np


import pyximport
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    language_level=3,
    reload_support=True
)
from cy_funcs import *


class Layer(object):
    def __init__(self, eps, width):
        self.eps = eps
        self.width = width


class Grating(object):
    def __init__(self, eps1, eps2, widths):
        self.eps1 = eps1
        self.eps2 = eps2
        self.widths = widths
        self.layers = []
        for i, d in enumerate(widths):
            eps = self.eps1 if i % 2 == 0 else self.eps2
            self.layers.append(Layer(eps, d))

    def props_layers(self, f, b=0):
        for i, d in enumerate(self.widths):
            eps = self.eps1 if i % 2 == 0 else self.eps2
            yield propagator_layer(f, eps, d, b)

    def propagator(self, f, b=0):
        return propagator_grating(f, self.eps1, self.eps2, self.widths, b)

    def transmittivity(self, f, b=0., pol='x'):
        propagators = self.propagator(f, b)
        op_t = np.array([operator_t(p, 1., 1., b) for p in propagators])
        if pol == 'x':
            return np.abs(op_t[:, 0, 0]) ** 2
        if pol == 'y':
            return np.abs(op_t[:, 1, 1]) ** 2
        if pol == 'xy':
            return np.abs(op_t[:, 0, 1]) ** 2
        if pol == 'yx':
            return np.abs(op_t[:, 1, 0]) ** 2
