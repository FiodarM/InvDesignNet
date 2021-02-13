import numpy as np
import matplotlib.pyplot as plt
from grating import Grating


if __name__ == '__main__':
    n_grating_layers = 15
    n_freqs = 200
    epsilon_Si = 13.491
    epsilon_SiO2 = 2.085136
    np.random.seed(42)
    D = np.random.random_sample(15)
    gr = Grating(epsilon_Si, epsilon_SiO2, D)
    freqs = np.linspace(0.15, 0.25, n_freqs)
    ts = gr.transmittivity(freqs)
    plt.plot(freqs, ts)
    plt.show()