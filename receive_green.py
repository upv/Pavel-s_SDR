import sys
sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

# rx_2ch_realtime.py
import adi
import numpy as np
import matplotlib.pyplot as plt

URI_RX = "ip:192.168.2.2"   # IP приемника
FS = int(1e6)
LO = int(2.19e9)
N = 4096
N_fft = 4096

sdr = adi.ad9363(URI_RX)

sdr.sample_rate = FS
sdr.rx_lo = LO
sdr.rx_rf_bandwidth = int(FS)
sdr.rx_buffer_size = N

# Включаем оба RX канала
sdr.rx_enabled_channels = [0, 1]

# AGC
# sdr.gain_control_mode_chan0 = "slow_attack"
# sdr.gain_control_mode_chan1 = "slow_attack"

# Если нужен ручной режим:
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = int(40)
sdr.rx_hardwaregain_chan1 = int(40)

window = np.hanning(N_fft)
freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1/FS)) / 1e3

plt.ion()
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8))

line0, = ax0.plot([], [])
line1, = ax1.plot([], [])

for ax, title in [(ax0, "RX0 Spectrum"), (ax1, "RX1 Spectrum")]:
    ax.set_xlim(-FS/2/1e3, FS/2/1e3)
    ax.set_ylim(-140, 10)
    ax.set_xlabel("Frequency relative to LO, kHz")
    ax.set_ylabel("Amplitude, dBFS")
    ax.set_title(title)
    ax.grid(True)

print(f"RX LO = {LO/1e9:.6f} GHz, FS = {FS/1e6:.3f} MSPS")

while True:
    data = sdr.rx()

    # Для 2 каналов data = [rx0, rx1]
    rx0 = data[0]
    rx1 = data[1]

    rx0 = rx0 - np.mean(rx0)
    rx1 = rx1 - np.mean(rx1)

    spec0 = np.fft.fftshift(np.fft.fft(rx0 * window))
    spec1 = np.fft.fftshift(np.fft.fft(rx1 * window))

    mag0 = 20 * np.log10(np.abs(spec0) / N + 1e-12)
    mag1 = 20 * np.log10(np.abs(spec1) / N + 1e-12)

    line0.set_data(freq, mag0)
    line1.set_data(freq, mag1)

    fig.canvas.draw()
    fig.canvas.flush_events()