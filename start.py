#!/usr/bin/env python3
"""
AntSDR Pluto: generate a simple signal and view it in real time.
Switch RX_ONLY: True = receiver + plot only; False = transmit tone + receiver + plot (use loopback cable).
"""
import sys
sys.path.append('/home/pavel/pyadi-iio/')
sys.path.append('/home/pavel/libiio')

import numpy as np
import matplotlib.pyplot as plt
import adi

# --- Config (change for your setup) ---
URI = "ip:192.168.2.1"  # or "ip:pluto.local" or "usb:1.2.3"
SAMPLE_RATE = int(2.4e6)  # Hz
CENTER_FREQ = int(915e6)  # Hz (e.g. 915 MHz ISM)
TONE_FREQ = 10e3          # Hz (offset from center; tone at center + 10 kHz)

# Switch: True = receiver + plot only; False = transmit tone + receiver + plot (loopback)
RX_ONLY = True

RX_GAIN = 30               # dB
TX_GAIN = -20              # dB (keep low if loopback cable is short)

# Buffer / plot
RX_BUFFER_SIZE = 4096
PLOT_FFT_SIZE = 2048
UPDATE_MS = 50             # plot update interval


def main():
    print("Connecting to Pluto at", URI)
    sdr = adi.ad9363(URI)

    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_enabled_channels = [0]
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_lo = CENTER_FREQ
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN

    if not RX_ONLY:
        # Generate simple complex tone: e^(j*2*pi*f*t)
        num_samps = 4096
        t = np.arange(num_samps) / SAMPLE_RATE
        tone = np.exp(2j * np.pi * TONE_FREQ * t).astype(np.complex64)
        # Pluto expects scale roughly [-2^14, 2^14]
        tone *= 0.3 * (2**14)
        sdr.tx_enabled_channels = [0]
        sdr.tx_lo = CENTER_FREQ
        sdr.tx_rf_bandwidth = SAMPLE_RATE
        sdr.tx_cyclic_buffer = True
        sdr.gain_control_mode_chan0 = "manual"
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.tx(tone)
        print(f"TX: tone at {TONE_FREQ/1e3:.1f} kHz offset, {CENTER_FREQ/1e6:.1f} MHz center")

    # Real-time receiver + plot (time domain + FFT)
    mode_str = "RX only" if RX_ONLY else "TX loopback + RX"
    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f"Pluto real-time — {mode_str} (close window to stop)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlabel("Sample")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.set_xlabel("Frequency (kHz)")
    ax_fft.set_xlim(-SAMPLE_RATE / 2e3, SAMPLE_RATE / 2e3)

    freqs = np.fft.fftshift(np.fft.fftfreq(PLOT_FFT_SIZE, 1 / SAMPLE_RATE)) / 1e3  # kHz

    try:
        while plt.fignum_exists(fig.number):
            rx_result = sdr.rx()
            # rx() returns list of arrays per channel, or single array when 1 channel
            data = np.asarray(rx_result[0] if isinstance(rx_result, (list, tuple)) else rx_result)
            data = data[:PLOT_FFT_SIZE]

            # Time domain
            ax_time.clear()
            ax_time.plot(np.real(data), label="I", alpha=0.8)
            ax_time.plot(np.imag(data), label="Q", alpha=0.8)
            ax_time.set_ylabel("Amplitude")
            ax_time.set_xlabel("Sample")
            ax_time.legend(loc="upper right", fontsize=8)
            ax_time.grid(True, alpha=0.3)

            # FFT
            win = np.hanning(len(data))
            spec = np.fft.fftshift(np.fft.fft(data * win))
            mag_db = 20 * np.log10(np.abs(spec) + 1e-12)
            ax_fft.clear()
            ax_fft.plot(freqs, mag_db)
            ax_fft.set_ylabel("Magnitude (dB)")
            ax_fft.set_xlabel("Frequency (kHz)")
            ax_fft.set_xlim(-SAMPLE_RATE / 2e3, SAMPLE_RATE / 2e3)
            ax_fft.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(UPDATE_MS / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not RX_ONLY:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
        plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
