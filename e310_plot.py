# -*- coding: utf-8 -*-
"""Live matplotlib scope for E310 TX / RX IQ."""

from __future__ import annotations

import os
import numpy as np

NFFT = 2048
WATERFALL_ROWS = 72
MAX_PLOT_PTS = 4000

# Cursor/debugpy sets MPLBACKEND=Agg, which draws nothing on screen.
if os.environ.get("MPLBACKEND", "").lower() == "agg":
    os.environ["MPLBACKEND"] = "macosx"


def _use_interactive_backend():
    """Force a real GUI backend. Agg makes plt.show() a no-op."""
    import matplotlib
    current = matplotlib.get_backend().lower()
    if current in ("macosx", "tkagg", "qt5agg", "qtagg", "webagg"):
        return matplotlib.get_backend()
    last_err = None
    for name in ("MacOSX", "TkAgg", "QtAgg", "WebAgg"):
        try:
            matplotlib.use(name, force=True)
            return matplotlib.get_backend()
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "No GUI matplotlib backend (need MacOSX). "
        f"Last error: {last_err}"
    )


def _as_iq(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return np.asarray(x, dtype=np.complex64).ravel()


def _decimate(x, max_pts=MAX_PLOT_PTS):
    n = len(x)
    if n <= max_pts:
        return np.arange(n), x
    step = int(np.ceil(n / max_pts))
    return np.arange(0, n, step), x[::step]


def _spectrum(x, fs, nfft=NFFT):
    x = np.asarray(x, dtype=np.complex64)
    if len(x) >= nfft:
        x = x[:nfft]
    else:
        x = np.pad(x, (0, nfft - len(x)))
    w = np.hanning(nfft).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(x * w))
    psd = 20.0 * np.log10(np.abs(spec) / nfft + 1e-12)
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs)) / 1e6
    return freq, psd


def _wrap_window(iq, start, win):
    n = len(iq)
    start %= n
    if start + win <= n:
        return iq[start:start + win]
    return np.concatenate([iq[start:], iq[:win - (n - start)]])


class LiveIqScope:
    """2x2 live view: I/Q time, spectrum, waterfall, constellation."""

    def __init__(self, fs, title="E310"):
        backend = _use_interactive_backend()
        import matplotlib.pyplot as plt
        print(f"matplotlib backend: {backend}")

        self.fs = float(fs)
        self.wf = np.full((WATERFALL_ROWS, NFFT), -90.0, dtype=np.float32)
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 7))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.fig.suptitle(title, fontsize=12)

        ax_t, ax_f, ax_w, ax_c = self.ax[0, 0], self.ax[0, 1], self.ax[1, 0], self.ax[1, 1]
        self.line_i, = ax_t.plot([], [], lw=0.8, label="I")
        self.line_q, = ax_t.plot([], [], lw=0.8, label="Q")
        ax_t.set_xlabel("µs")
        ax_t.set_ylabel("amplitude")
        ax_t.set_title("time")
        ax_t.legend(loc="upper right", fontsize=8)
        ax_t.grid(alpha=0.3)

        nyq = self.fs / 2e6
        self.line_psd, = ax_f.plot([], [], lw=0.9)
        ax_f.set_xlim(-nyq, nyq)
        ax_f.set_ylim(-90, 5)
        ax_f.set_xlabel("MHz from LO")
        ax_f.set_ylabel("dBFS")
        ax_f.set_title("spectrum")
        ax_f.grid(alpha=0.3)

        self.im = ax_w.imshow(
            self.wf, aspect="auto", origin="lower", cmap="magma",
            extent=[-nyq, nyq, 0, WATERFALL_ROWS], vmin=-85, vmax=-10,
            interpolation="nearest",
        )
        ax_w.set_xlabel("MHz from LO")
        ax_w.set_ylabel("frame")
        ax_w.set_title("waterfall")

        self.scat = ax_c.plot([], [], ".", ms=2, alpha=0.35)[0]
        ax_c.set_xlim(-2, 2)
        ax_c.set_ylim(-2, 2)
        ax_c.set_aspect("equal")
        ax_c.set_xlabel("I")
        ax_c.set_ylabel("Q")
        ax_c.set_title("constellation")
        ax_c.grid(alpha=0.3)
        self.ax_c = ax_c
        self.status = self.fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    def update(self, iq, constellation=None, status=""):
        iq = _as_iq(iq)
        if iq.size == 0:
            return
        t_idx, iq_d = _decimate(iq)
        t_us = t_idx / self.fs * 1e6
        self.line_i.set_data(t_us, iq_d.real)
        self.line_q.set_data(t_us, iq_d.imag)
        peak = float(np.max(np.abs(iq_d))) + 1e-6
        self.ax[0, 0].set_xlim(t_us[0], t_us[-1])
        self.ax[0, 0].set_ylim(-1.2 * peak, 1.2 * peak)

        freq, psd = _spectrum(iq, self.fs)
        self.line_psd.set_data(freq, psd)
        self.ax[0, 1].set_ylim(float(np.percentile(psd, 5)) - 5, float(np.max(psd)) + 3)

        self.wf = np.roll(self.wf, -1, axis=0)
        self.wf[-1] = psd.astype(np.float32)
        self.im.set_data(self.wf)
        lo, hi = float(np.percentile(self.wf, 10)), float(np.max(self.wf))
        if hi - lo < 1:
            hi = lo + 1
        self.im.set_clim(lo, hi)

        pts = _as_iq(constellation) if constellation is not None else iq_d
        pts = pts[np.isfinite(pts)]
        if pts.size > 8000:
            pts = pts[:: int(np.ceil(pts.size / 8000))]
        self.scat.set_data(pts.real, pts.imag)
        r = float(np.percentile(np.abs(pts), 99)) if pts.size else 1.0
        r = max(r * 1.3, 1.5)
        self.ax_c.set_xlim(-r, r)
        self.ax_c.set_ylim(-r, r)
        self.status.set_text(status)
        self.fig.canvas.draw_idle()

    def run(self, frames, interval_ms=40):
        import matplotlib.pyplot as plt

        # FuncAnimation + Agg/debugpy often never maps a window. Pause-loop
        # processes Cocoa events and keeps the figure on screen.
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)
        print(f"plot window open  backend={plt.get_backend()}  "
              "close the window or Ctrl+C to stop")
        interval = max(interval_ms / 1000.0, 0.02)
        try:
            while plt.fignum_exists(self.fig.number):
                try:
                    iq, const, status = next(frames)
                except StopIteration:
                    plt.pause(0.2)
                    continue
                self.update(iq, const, status)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(interval)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()


def iter_tx_windows(iq, fs, win, hop, ref=None):
    """Yield sliding windows of the cyclic TX frame."""
    n = len(iq)
    pos = 0
    while True:
        chunk = _wrap_window(iq, pos, win)
        t_ms = pos / fs * 1e3
        status = f"TX cyclic  {t_ms:7.3f} ms into frame  |  {n} samples loop"
        const = None
        if ref is not None:
            const = ref.data_syms.ravel()
        yield chunk, const, status
        pos = (pos + hop) % n


def iter_rx_live(sdr, cfg, ref, decode=True):
    """Yield successive SDR captures; decode OFDM when the buffer is long enough."""
    from zc_ofdm_common import process_capture

    while True:
        raw = _as_iq(sdr.rx())
        const = None
        rms = float(np.sqrt(np.mean(np.abs(raw) ** 2)))
        status = f"RX {len(raw)} samp  rms={rms:.1f}  peak={np.max(np.abs(raw)):.1f}"
        if decode and len(raw) >= cfg.FRAME_LEN + 4 * cfg.SYM_LEN:
            try:
                res = process_capture(raw, ref)
                const = res.eq_data.ravel()
                status += (
                    f"  |  EVM {res.evm_db:.1f} dB  SNR {res.snr_db:.1f} dB  "
                    f"CFO {res.cfo_hz / 1e3:+.2f} kHz"
                )
            except Exception as e:
                status += f"  |  {e}"
        yield raw, const, status


def iter_tx_rx_live(sdr, tx_iq, fs, win, hop, ref=None):
    """TX playhead plus a live RX capture from the same radio (loopback / leakage)."""
    pos = 0
    n = len(tx_iq)
    while True:
        try:
            raw = _as_iq(sdr.rx())
        except Exception as e:
            dummy = _wrap_window(tx_iq, pos, win)
            yield dummy, None, f"TX window  |  RX error: {e}"
            pos = (pos + hop) % n
            continue
        rms = float(np.sqrt(np.mean(np.abs(raw) ** 2)))
        status = (
            f"TX playhead {(pos / fs) * 1e3:6.2f} ms  |  "
            f"RX {len(raw)} samp  rms={rms:.1f}"
        )
        yield raw, raw, status
        pos = (pos + hop) % n


def show_tx(iq, fs, ref=None, title="E310 TX", sdr=None):
    """Animate the cyclic TX waveform. If sdr is set, overlay live RX."""
    sym = ref.cfg.SYM_LEN if ref is not None else 2048
    cfg_win = min(len(iq), max(2048, 2 * sym))
    hop = max(64, len(iq) // 80)
    scope = LiveIqScope(fs, title=title)
    if sdr is not None:
        frames = iter_tx_rx_live(sdr, iq, fs, cfg_win, hop, ref)
        scope.ax[0, 0].set_title("RX time (TX looping in hardware)")
        scope.ax_c.set_title("RX I/Q")
    else:
        frames = iter_tx_windows(iq, fs, cfg_win, hop, ref)
        scope.ax_c.set_title("TX QPSK (known data)")
    scope.run(frames, interval_ms=40)


def show_rx_live(sdr, cfg, ref, title="E310 RX"):
    scope = LiveIqScope(cfg.FS, title=title)
    scope.ax_c.set_title("equalized QPSK (or raw I/Q)")
    scope.run(iter_rx_live(sdr, cfg, ref), interval_ms=20)


def show_rx_recording(rx, fs, title="E310 RX recording"):
    rx = _as_iq(rx)
    win = min(len(rx), 8192)
    hop = max(64, win // 8)
    pos = 0

    def frames():
        nonlocal pos
        while pos + win <= len(rx):
            chunk = rx[pos:pos + win]
            yield chunk, None, f"offset {pos} / {len(rx)}"
            pos += hop
        yield rx[-win:], None, f"offset {len(rx) - win} / {len(rx)}"

    scope = LiveIqScope(fs, title=title)
    scope.run(frames(), interval_ms=40)
