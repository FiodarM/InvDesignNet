import gc

import numpy as np
import pyximport
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    language_level=3,
    reload_support=True
)
from grating import Grating, operator_t
import os
from tqdm import tqdm


batch_size = 1000
n_grating_layers = 15
n_freqs = 200
a = 1.
freqs = np.linspace(0.15 / a, 0.25 / a, n_freqs)
epsilon_Si = 13.491
epsilon_SiO2 = 2.085136


def save_samples(fname, samples):
    n_samples = len(samples['R'])
    if os.path.exists(fname):
        with np.load(fname) as existing:
            for k in samples.keys():
                samples[k] = np.vstack((existing[k], samples[k]))
    try:
        np.savez(fname, **samples)
    except KeyboardInterrupt:
        pass
    print("Saved {0} samples to {1}. The dataset contains {2} samples."
          .format(n_samples, fname, samples['R'].shape[0]))


fname = 'dataset.npz'


if __name__ == '__main__':
    samples = dict(D=[], R=[])
    try:
        pbar = tqdm(total=batch_size, leave=False)
        i = 0
        while True:
            D = np.random.random_sample(n_grating_layers)
            gr = Grating(epsilon_Si, epsilon_SiO2, D)
            ps = gr.propagator(freqs)
            ts = np.array(list(map(operator_t, ps)))
            i += 1
            pbar.update(1)
            samples['D'].append(D)
            samples['R'].append(np.abs(ts[:, 0, 0]) ** 2)
            if i == batch_size:
                save_samples(fname, samples)
                samples = dict(D=[], R=[])
                gc.collect()
                pbar.reset()
                i = 0
    except KeyboardInterrupt:
        print('Interrupting calculation...')
        answer = input('Do you want to save calculated data? y/[n]: ')
        if answer == 'y':
            save_samples(fname, samples)
    finally:
        pbar.close()
        print('Exiting script')
        exit(0)
