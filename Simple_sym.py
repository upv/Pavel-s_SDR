import numpy as np

time_samples = 10000
sample_rate = 10e6
fft_size = 1024

n_sym = time_samples // fft_size
timeline = np.arange(0, fft_size / sample_rate, 1 / sample_rate)

pass




















