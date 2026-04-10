import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create radio
    sdr = adi.ad9361("ip:192.168.2.2")
    samp_rate = 1e6

    # Configure Rx properties
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(1e6)
    sdr.rx_lo = int(2.19e9)
    sdr.rx_buffer_size = 8192
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(50)
    sdr.rx_hardwaregain_chan1 = int(50)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax_time0, ax_spec0 = axes[0]
    ax_time1, ax_spec1 = axes[1]

    time0_line, = ax_time0.plot([], [], lw=1)
    spec0_line, = ax_spec0.plot([], [], lw=1)
    time1_line, = ax_time1.plot([], [], lw=1)
    spec1_line, = ax_spec1.plot([], [], lw=1)

    ax_time0.set_title("RX0 time domain")
    ax_time0.set_xlabel("Time (µs)")
    ax_time0.set_ylabel("Amplitude")
    ax_time0.grid(True, alpha=0.3)

    ax_time1.set_title("RX1 time domain")
    ax_time1.set_xlabel("Time (µs)")
    ax_time1.set_ylabel("Amplitude")
    ax_time1.grid(True, alpha=0.3)

    for ax, title in zip(
        (ax_spec0, ax_spec1), ("RX0 spectrum", "RX1 spectrum")
    ):
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    try:
        while True:
            # Receive complex samples from both channels
            x = sdr.rx()  # x[0] -> RX0, x[1] -> RX1

            # Channel 0
            sig0 = np.asarray(x[0])
            N0 = len(sig0)
            t0 = np.arange(N0) / samp_rate * 1e6  # µs
            time0_line.set_data(t0, np.real(sig0))
            ax_time0.set_xlim(t0[0], t0[-1])
            ax_time0.set_ylim(
                1.1 * np.min(np.real(sig0)), 1.1 * np.max(np.real(sig0))
            )

            window0 = np.hanning(N0)
            spec0 = np.fft.fftshift(np.fft.fft(sig0 * window0))
            freq0 = (
                np.fft.fftshift(np.fft.fftfreq(N0, 1.0 / samp_rate)) / 1e6
            )  # MHz
            psd0 = 20 * np.log10(np.abs(spec0) + 1e-12)
            spec0_line.set_data(freq0, psd0)
            ax_spec0.set_xlim(freq0[0], freq0[-1])
            ax_spec0.set_ylim(psd0.min() - 10, psd0.max() + 10)

            # Channel 1
            sig1 = np.asarray(x[1])
            N1 = len(sig1)
            t1 = np.arange(N1) / samp_rate * 1e6  # µs
            time1_line.set_data(t1, np.real(sig1))
            ax_time1.set_xlim(t1[0], t1[-1])
            ax_time1.set_ylim(
                1.1 * np.min(np.real(sig1)), 1.1 * np.max(np.real(sig1))
            )

            window1 = np.hanning(N1)
            spec1 = np.fft.fftshift(np.fft.fft(sig1 * window1))
            freq1 = (
                np.fft.fftshift(np.fft.fftfreq(N1, 1.0 / samp_rate)) / 1e6
            )  # MHz
            psd1 = 20 * np.log10(np.abs(spec1) + 1e-12)
            spec1_line.set_data(freq1, psd1)
            ax_spec1.set_xlim(freq1[0], freq1[-1])
            ax_spec1.set_ylim(psd1.min() - 10, psd1.max() + 10)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()