import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

URI_TX = "ip:192.168.2.1"
LO = int(2.19e9)

N_FFT = 1024
SUBCARRIER_SPACING_HZ = 30e3
FS = int(20e6)

sdr = adi.ad9361(URI_TX)
sdr.sample_rate = FS
sdr.tx_lo = LO
sdr.tx_rf_bandwidth = int(FS)
sdr.tx_enabled_channels = [0, 1]
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.tx_hardwaregain_chan0 = int(-10)
sdr.tx_hardwaregain_chan1 = int(-10)
sdr.tx_cyclic_buffer = True



delta_f = 30e3
frac_guard = 0.5

N_sc_av = int(FS / delta_f)
N_fft = int(2 ** np.ceil(np.log2(N_sc_av)))
N_sc_use = int(np.ceil(N_fft * frac_guard))
guard_length = int(0.5 * N_fft)

CP_len = int(np.trunc(N_fft * 0.2))
num_bits_sym = 1
num_bits_sym_d = 1
do_dc_offset = 0

BlockSize = N_sc_use * num_bits_sym
BlockSize_d = N_sc_use * num_bits_sym_d

N_repeat = 10
num_rep_pilot = 1
signal_scale = 1.0

 
preamble = 1 - 2 * rng.integers(0, 2, size=(int(N_fft / 2), 1))
preamble = np.complex64(preamble)
preamble_full = np.matlib.repmat(preamble, 2, 1)
preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))

pream_len = preamble.shape[0]


data_stream = rng.integers(0, 2, size=(int(BlockSize), 1))
mod_sym_pilot = 1.0 - 2.0 * np.complex64(data_stream)

tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64);
tx_ofdm_sym[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset] = mod_sym_pilot[int(N_sc_use/2):];
tx_ofdm_sym[-int(N_sc_use/2):] = mod_sym_pilot[0:int(N_sc_use/2)];

#time_ofdm_sym_pilot = ifft(tx_ofdm_sym) * sqrt(N_fft) * signal_scale;
#time_ofdm_sym_cp_pilot = [time_ofdm_sym_pilot(end - CP_len + 1:end); time_ofdm_sym_pilot_rep];

time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")
time_ofdm_sym_cp_pilot = np.concatenate(( time_ofdm_sym_pilot[-CP_len:] , time_ofdm_sym_pilot))


# data frame
#data_stream = randi([0, 1], BlockSize_d, 1);
data_stream = rng.integers(0, 2, size=(BlockSize_d, 1))
mod_sym = 1 - 2 * data_stream
mod_sym = np.complex64(mod_sym)


#tx_ofdm_sym = zeros(N_fft, 1);
tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64)
tx_ofdm_sym[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset] = mod_sym[int(N_sc_use/2):]
tx_ofdm_sym[-int(N_sc_use/2):] = mod_sym[0:int(N_sc_use/2)]

time_ofdm_sym = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")

time_ofdm_sym_cp = np.concatenate((time_ofdm_sym[-CP_len:], time_ofdm_sym))

guard_frame = np.zeros((guard_length, 1), dtype=np.complex64)

#full_frame = [preamble_full_cp; time_ofdm_sym_cp_pilot; time_ofdm_sym_cp; guard_frame];
full_frame = np.concatenate((preamble_full_cp, time_ofdm_sym_cp_pilot, time_ofdm_sym_cp, guard_frame))

frame_len = full_frame.shape[0]
print(f'FrameLen={frame_len}')

num_rep = 3
full_frame_rep = np.matlib.repmat(full_frame, num_rep, 1)

FrameSize = full_frame_rep.shape[0];
sig_len = full_frame_rep.shape[0];

plt.figure(1)
x_arr = np.arange(0, FrameSize)
plt.figure(1)
plt.plot(x_arr, np.real(full_frame_rep))
plt.grid()
#plt.show()



























