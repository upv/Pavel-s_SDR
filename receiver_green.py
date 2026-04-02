import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import sys
import numpy as np
import adi
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# =========================
# Настройки SDR
# =========================
URI = "ip:192.168.2.2"      # поменяй на свой адрес
FS = 10_000_000              # sample rate
FC = 1900_000_000            # center frequency
BW = 10_000_000              # RF bandwidth
BUF = 2048                  # rx buffer size
USE_TWO_CHANNELS = True     # False -> только RX0

# =========================
# Инициализация SDR
# =========================
# Для 2-канального режима обычно используют adi.ad9361(...)
sdr = adi.ad9361(URI)

sdr.sample_rate = FS
sdr.rx_lo = FC
sdr.rx_rf_bandwidth = BW
sdr.rx_buffer_size = BUF

if USE_TWO_CHANNELS:
    sdr.rx_enabled_channels = [0, 1]
else:
    sdr.rx_enabled_channels = [0]

# Лучше manual, чтобы картинка не "плавала"
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 30

if USE_TWO_CHANNELS:
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan1 = 30

# =========================
# Qt / PyQtGraph GUI
# =========================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=False)

win = pg.GraphicsLayoutWidget(show=True, title="Pluto / ANTSDR RX realtime")
win.resize(1400, 900)

# ---- Time plots
p_time0 = win.addPlot(row=0, col=0, title="RX0 amplitude |x[n]|")
curve_time0 = p_time0.plot()
p_time0.setLabel("left", "Amplitude")
p_time0.setLabel("bottom", "Sample")

p_iq0 = win.addPlot(row=1, col=0, title="RX0 I/Q")
curve_i0 = p_iq0.plot()
curve_q0 = p_iq0.plot()
p_iq0.setLabel("left", "Value")
p_iq0.setLabel("bottom", "Sample")
p_iq0.addLegend()


# ---- Frequency plots
p_fft0 = win.addPlot(row=2, col=0, title="RX0 spectrum")
curve_fft0 = p_fft0.plot()
p_fft0.setLabel("left", "dB")
p_fft0.setLabel("bottom", "Frequency offset", "Hz")

if USE_TWO_CHANNELS:
    p_time1 = win.addPlot(row=0, col=1, title="RX1 amplitude |x[n]|")
    curve_time1 = p_time1.plot()
    p_time1.setLabel("left", "Amplitude")
    p_time1.setLabel("bottom", "Sample")

    p_iq1 = win.addPlot(row=1, col=1, title="RX1 I/Q")
    curve_i1 = p_iq1.plot()
    curve_q1 = p_iq1.plot()
    p_iq1.setLabel("left", "Value")
    p_iq1.setLabel("bottom", "Sample")
    p_iq1.addLegend()


    p_fft1 = win.addPlot(row=2, col=1, title="RX1 spectrum")
    curve_fft1 = p_fft1.plot()
    p_fft1.setLabel("left", "dB")
    p_fft1.setLabel("bottom", "Frequency offset", "Hz")

status_label = QtWidgets.QLabel()
status_label.setText("Running...")
proxy = QtWidgets.QGraphicsProxyWidget()
proxy.setWidget(status_label)
win.addItem(proxy, row=3, col=0, colspan=2 if USE_TWO_CHANNELS else 1)

# =========================
# FFT helper
# =========================
def calc_fft_db(x, fs):
    w = np.hanning(len(x))
    X = np.fft.fftshift(np.fft.fft(x * w))
    psd = 20 * np.log10(np.abs(X) + 1e-12)
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1/fs))
    return f, psd

# =========================
# Update loop
# =========================
def update():
    try:
        data = sdr.rx()

        if USE_TWO_CHANNELS:
            rx0 = data[0]
            rx1 = data[1]
        else:
            rx0 = data
            rx1 = None

        # -------- RX0 time
        amp0 = np.abs(rx0)
        curve_time0.setData(amp0)

        curve_i0.setData(np.real(rx0))
        curve_q0.setData(np.imag(rx0))

        # -------- RX0 FFT
        f0, psd0 = calc_fft_db(rx0, FS)
        curve_fft0.setData(f0, psd0)

        # -------- RX1
        if USE_TWO_CHANNELS and rx1 is not None:
            amp1 = np.abs(rx1)
            curve_time1.setData(amp1)

            curve_i1.setData(np.real(rx1))
            curve_q1.setData(np.imag(rx1))

            f1, psd1 = calc_fft_db(rx1, FS)
            curve_fft1.setData(f1, psd1)

            pwr0 = 20 * np.log10(np.sqrt(np.mean(np.abs(rx0) ** 2)) + 1e-12)
            pwr1 = 20 * np.log10(np.sqrt(np.mean(np.abs(rx1) ** 2)) + 1e-12)
            status_label.setText(
                f"RX0 RMS: {pwr0:.1f} dBFS   |   RX1 RMS: {pwr1:.1f} dBFS"
            )
        else:
            pwr0 = 20 * np.log10(np.sqrt(np.mean(np.abs(rx0) ** 2)) + 1e-12)
            status_label.setText(f"RX0 RMS: {pwr0:.1f} dBFS")

    except Exception as e:
        status_label.setText(f"Error: {e}")

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)   # ~20 FPS

if __name__ == "__main__":
    sys.exit(app.exec())












