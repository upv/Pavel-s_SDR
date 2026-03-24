#!/usr/bin/env python3
"""
Simple FM radio with Pluto SDR — receive and play broadcast FM (88–108 MHz).
Usage: python fmradio.py [frequency_mhz]
Example: python fmradio.py 98.5
"""
import sys
sys.path.append('/home/pavel/pyadi-iio/')
sys.path.append('/home/pavel/libiio')

import threading
import queue
import numpy as np
import sounddevice as sd

# Optional: add pyadi path if not installed system-wide
# sys.path.append("/home/pavel/pyadi-iio/")

import adi

# --- Config ---
URI = "ip:192.168.2.1"   # or "ip:pluto.local" or "usb:..."
SAMPLE_RATE = int(2.4e6)  # Hz
RX_GAIN = 40              # dB (adjust for your antenna)
BUFFER_SIZE = 32768       # larger = more audio per rx(), fewer underflows
AUDIO_RATE = 48000
AUDIO_CHUNK = 1024        # samples per audio frame
PREBUFFER_CHUNKS = 32     # fill queue before starting playback

# FM demod: resample from SAMPLE_RATE to QUAD_RATE, then to AUDIO_RATE
QUAD_RATE = 240000        # Hz after first decimation (2.4M / 10)
DECIM_IQ = 10             # 2.4M -> 240k
DECIM_AUDIO = 5           # 240k -> 48k (240k/5)


def fm_demod(iq):
    """Wideband FM: phase discriminator (diff of angle)."""
    angle = np.unwrap(np.angle(iq))
    demod = np.diff(angle)
    # pad to same length
    demod = np.concatenate([[demod[0]], demod])
    return demod.astype(np.float32)


def run_sdr(sdr, freq_hz, out_queue):
    """Capture IQ, channel-filter, FM demod, resample to audio rate, push to queue."""
    from scipy import signal as scipy_signal
    sdr.rx_lo = int(freq_hz)
    # Lowpass for ~200 kHz channel (single FM station)
    channel_taps = scipy_signal.firwin(81, 100e3, fs=SAMPLE_RATE)
    # Decimation filters
    audio_taps = scipy_signal.firwin(61, 15e3, fs=QUAD_RATE)
    leftover = np.array([], dtype=np.complex64)
    audio_leftover = np.array([], dtype=np.float32)

    while True:
        try:
            rx = sdr.rx()
            data = np.asarray(rx[0] if isinstance(rx, (list, tuple)) else rx, dtype=np.complex64)
        except Exception as e:
            print("RX error:", e)
            break
        data = np.concatenate([leftover, data])
        n = (len(data) // DECIM_IQ) * DECIM_IQ
        if n == 0:
            leftover = data
            continue
        leftover = data[n:]
        data = data[:n]
        # Channel filter and decimate
        filtered = scipy_signal.lfilter(channel_taps, 1.0, data)
        iq_240k = filtered[::DECIM_IQ]
        # FM demod
        audio_240k = fm_demod(iq_240k)
        # Lowpass and decimate to 48k
        audio_lp = scipy_signal.lfilter(audio_taps, 1.0, audio_240k)
        audio_48k = audio_lp[::DECIM_AUDIO]
        audio_48k = np.concatenate([audio_leftover, audio_48k])
        # Normalize to roughly [-1, 1]; WFM deviation ~75 kHz, adjust gain
        gain = 0.5 / (np.std(audio_48k) + 1e-9)
        audio_48k = np.clip(audio_48k * gain, -1.0, 1.0).astype(np.float32)
        n_out = (len(audio_48k) // AUDIO_CHUNK) * AUDIO_CHUNK
        if n_out > 0:
            for i in range(0, n_out, AUDIO_CHUNK):
                chunk = audio_48k[i : i + AUDIO_CHUNK]
                try:
                    out_queue.put_nowait(chunk)
                except queue.Full:
                    out_queue.get_nowait()
                    out_queue.put_nowait(chunk)
        audio_leftover = audio_48k[n_out:]


def main():
    freq_mhz = float(sys.argv[1]) if len(sys.argv) > 1 else 98.5
    freq_hz = int(freq_mhz * 1e6)
    if not (88 <= freq_mhz <= 108):
        print("Warning: typical FM band is 88–108 MHz")

    print("Pluto FM radio — tuning to {:.1f} MHz. Ctrl+C to stop.".format(freq_mhz))
    sdr = adi.ad9363(URI)
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_enabled_channels = [0]
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_lo = freq_hz
    sdr.rx_buffer_size = BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN

    q = queue.Queue(maxsize=256)
    thread = threading.Thread(target=run_sdr, args=(sdr, freq_hz, q), daemon=True)
    thread.start()
    # Pre-fill queue so playback doesn't underflow at start
    print("Buffering...", end=" ", flush=True)
    while q.qsize() < PREBUFFER_CHUNKS:
        sd.sleep(50)
    print("playing.")

    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(status)
        written = 0
        while written < frames:
            try:
                c = q.get(timeout=0.5)
            except queue.Empty:
                outdata[written:, 0] = 0
                return
            take = min(frames - written, len(c))
            outdata[written : written + take, 0] = c[:take]
            written += take
            if take < len(c):
                q.put(c[take:])

    stream = sd.OutputStream(
        samplerate=AUDIO_RATE,
        channels=1,
        dtype="float32",
        blocksize=AUDIO_CHUNK,
        callback=audio_callback,
        latency="high",
    )
    stream.start()
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        pass
    stream.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()
