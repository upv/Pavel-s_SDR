import numpy as np

rng = np.random.default_rng(seed=42)

time_samples = 10000
sample_rate = 10e6
fft_size = 1024

n_sym = time_samples // fft_size
timeline = np.arange(0, fft_size / sample_rate, 1 / sample_rate)


modulation = 'QPSK'

def generate_bits(modulation, n_fft):

    if modulation == 'QPSK':
        n_bits_mod = 2
    else:
        n_bits_mod = 1

    n_bits = n_bits_mod * n_fft

    bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    return bits

bits = generate_bits(modulation, fft_size)




















