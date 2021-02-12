import numpy as np
import matplotlib.pyplot as plt


import pyximport
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    language_level=3,
    reload_support=True
)
from grating import Grating, operator_t


if __name__ == '__main__':
    n_grating_layers = 15
    n_freqs = 200
    epsilon_Si = 13.491
    epsilon_SiO2 = 2.085136
    np.random.seed(42)
    D = np.random.random_sample(15)
    gr = Grating(epsilon_Si, epsilon_SiO2, D)
    freqs = np.linspace(0.15, 0.25, n_freqs)
    ps = gr.propagator(freqs)
    ts = list(map(operator_t, ps))
    plt.plot(freqs, [np.abs(t[0, 0]) ** 2 for t in ts])
    plt.show()