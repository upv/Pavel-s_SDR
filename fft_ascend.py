import sys
# sys.path.append('/home/HwHiAiUser/pyadi-iio/')
# sys.path.append('/home/HwHiAiUser/libiio')

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import adi
from scipy.signal import get_window

# === SDR CONFIGURATION ===
sdr = adi.ad9361(uri="ip:192.168.2.2")  # Change if needed
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(1e6)
sdr.rx_lo = int(1.19e9)
sdr.rx_buffer_size = int(4096)
sdr.rx_rf_bandwidth = int(1e6)

# === FFT CONFIGURATION ===
N = sdr.rx_buffer_size
fs = sdr.sample_rate
freqs = np.fft.fftfreq(N, 1/fs)
freqs = np.fft.fftshift(freqs)

window = get_window("hann", N)

# ===== MATPLOTLIB SETUP =====
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

line0, = ax0.plot(freqs / 1e6, np.zeros(N), label='RX0')
line1, = ax1.plot(freqs / 1e6, np.zeros(N), label='RX1')

for ax in (ax0, ax1):
    ax.set_ylim(-120, 0)
    ax.set_xlim(freqs[0]/1e6, freqs[-1]/1e6)
    ax.set_ylabel("Power (dB)")
    ax.grid(True)
ax1.set_xlabel("Frequency (MHz)")
ax0.set_title("AntSDR Live FFT - Dual Channel")

ax0.legend(loc="upper right")
ax1.legend(loc="upper right")

# ===== UPDATE FUNCTION =====
def update(frame):
    samples = sdr.rx()   # returns list: [ch0, ch1]
    ch0, ch1 = samples[0], samples[1]

    # Apply window
    ch0_win = ch0 * window
    ch1_win = ch1 * window

    fft0 = np.fft.fftshift(np.fft.fft(ch0_win))
    fft1 = np.fft.fftshift(np.fft.fft(ch1_win))

    psd0 = 20 * np.log10(np.abs(fft0) + 1e-8)
    psd1 = 20 * np.log10(np.abs(fft1) + 1e-8)

    # normalize relative to each channel's peak
    psd0 -= np.max(psd0)
    psd1 -= np.max(psd1)

    line0.set_ydata(psd0)
    line1.set_ydata(psd1)
    return (line0, line1)

# ===== ANIMATION =====
ani = animation.FuncAnimation(
    fig,
    update,
    interval=100,           # refresh every 100 ms
    blit=True,
    cache_frame_data=False  # suppress unbounded cache warning
)

plt.tight_layout()
plt.show()
