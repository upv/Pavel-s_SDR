"""ANTSDR E316 (AD9361, Pluto-compatible firmware) through libiio.

Pure-Python ``iio`` bindings (libiio 0.x) are loaded from ``$PYLIBIIO_PY`` or the
usual site-packages; on this Mac they must bind to Homebrew's arm64
``libiio.dylib``, which ``find_library`` does not find by itself.

Pitfalls handled here (measured on the E316, 2026-10-14):
  * libiio queues 4 kernel blocks per RX buffer, so after a retune or gain change the
    first refills return STALE data -> :meth:`E316.capture` flushes them.
  * SoapySDR/other users leave the LO powered down -> ``powerdown`` is cleared.
  * A buffer must be destroyed before the context or the binding segfaults at exit.
"""

from __future__ import annotations

import ctypes.util
import glob
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

BREW_LIBIIO = "/opt/homebrew/lib/libiio.dylib"


def load_iio():
    """Return the ``iio`` module, loading libiio 0.x bindings from a known iio.py if needed."""
    try:
        import iio  # type: ignore
        return iio
    except Exception:
        pass
    cands = []
    if os.environ.get("PYLIBIIO_PY"):
        cands.append(os.environ["PYLIBIIO_PY"])
    cands += sorted(glob.glob(str(Path.home() / ".local/lib/python3*/site-packages/iio.py")))
    cands += sorted(glob.glob("/opt/homebrew/lib/python3*/site-packages/iio.py"))
    if os.path.exists(BREW_LIBIIO):
        orig = ctypes.util.find_library
        ctypes.util.find_library = lambda n: BREW_LIBIIO if n == "iio" else orig(n)
    for path in cands:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("iio", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:      # wrong architecture / library, try the next one
            print(f"iio bindings at {path} failed: {exc}", file=sys.stderr)
            continue
        sys.modules["iio"] = mod
        return mod
    raise ImportError("no usable pylibiio 0.x bindings found; set PYLIBIIO_PY=/path/to/iio.py")


def find_uri(preferred: str | None = None) -> str:
    """``preferred`` if given, else the first USB Pluto/ANTSDR context, else ip:192.168.2.2."""
    if preferred and preferred not in ("auto", ""):
        return preferred
    iio = load_iio()
    try:
        ctxs = iio.scan_contexts()
    except Exception:
        ctxs = {}
    usb = [u for u, d in ctxs.items() if u.startswith("usb:") and ("0456:b673" in d or "PlutoSDR" in d or "ANTSDR" in d)]
    if usb:
        return usb[0]
    other = [u for u, d in ctxs.items() if "ANTSDR" in d or "Pluto" in d]
    return other[0] if other else "ip:192.168.2.2"


class E316:
    """RX capture and cyclic TX on the E316. All frequencies in Hz, gains in dB."""

    def __init__(self, uri: str | None = None):
        self.iio = load_iio()
        self.uri = find_uri(uri)
        self.ctx = self.iio.Context(self.uri)
        self.phy = self.ctx.find_device("ad9361-phy")
        self.rx_dev = self.ctx.find_device("cf-ad9361-lpc")
        self.tx_dev = self.ctx.find_device("cf-ad9361-dds-core-lpc")
        self.rx_lo = self.phy.find_channel("altvoltage0", True)
        self.tx_lo = self.phy.find_channel("altvoltage1", True)
        self.rx_chan = [self.phy.find_channel(f"voltage{i}", False) for i in range(2)]
        self.tx_chan = [self.phy.find_channel(f"voltage{i}", True) for i in range(2)]
        self._rx_buf = None
        self._tx_buf = None
        self.num_rx = 1

    # ------------------------------------------------------------------ info
    @property
    def description(self) -> str:
        a = self.ctx.attrs
        return f"{a['hw_model']} fw {a['fw_version']} @ {self.uri}"

    def _set(self, chan, name, value):
        chan.attrs[name].value = str(value)

    def _get(self, chan, name) -> str:
        return chan.attrs[name].value

    # ------------------------------------------------------------------ RX
    def configure_rx(self, freq: float, fs: float, bw: float, gain: float | str = 40.0,
                     channels: tuple[int, ...] = (0,)) -> float:
        """Tune RX; ``gain`` is dB (manual), ``"agc"`` (slow attack) or ``"auto"`` (AGC once, then frozen).

        Returns the RX gain in dB that is in effect.
        """
        self._set(self.rx_lo, "powerdown", 0)
        self._set(self.rx_lo, "frequency", int(freq))
        for i in channels:
            ch = self.rx_chan[i]
            self._set(ch, "sampling_frequency", int(fs))
            self._set(ch, "rf_bandwidth", int(bw))
            if gain == "agc":
                self._set(ch, "gain_control_mode", "slow_attack")
            elif gain == "auto":
                self._set(ch, "gain_control_mode", "slow_attack")
            else:
                self._set(ch, "gain_control_mode", "manual")
                self._set(ch, "hardwaregain", float(gain))
        self.num_rx = len(channels)
        for ch in self.rx_dev.channels:
            idx = int(ch.id.replace("voltage", ""))
            ch.enabled = (idx // 2) in channels
        self._rx_buf = None
        if gain == "auto":
            time.sleep(0.1)
            self.capture(1 << 14, flush=4)
            g = self.rx_gain(channels[0])
            for i in channels:
                self._set(self.rx_chan[i], "gain_control_mode", "manual")
                self._set(self.rx_chan[i], "hardwaregain", g)
        return self.rx_gain(channels[0])

    def rx_gain(self, channel: int = 0) -> float:
        return float(self._get(self.rx_chan[channel], "hardwaregain").split()[0])

    def capture(self, num_samples: int, flush: int = 4) -> np.ndarray:
        """``num_samples`` per enabled RX channel -> interleaved int16 (I0 Q0 [I1 Q1] ...).

        ``flush`` refills are discarded first: libiio keeps 4 blocks in flight and
        the oldest ones predate the last retune.
        """
        if self._rx_buf is None or self._rx_buf_n != num_samples:
            self._rx_buf = None
            self._rx_buf = self.iio.Buffer(self.rx_dev, num_samples, False)
            self._rx_buf_n = num_samples
        for _ in range(flush):
            self._rx_buf.refill()
        self._rx_buf.refill()
        return np.frombuffer(self._rx_buf.read(), dtype=np.int16).copy()

    # ------------------------------------------------------------------ TX
    def configure_tx(self, freq: float, fs: float, bw: float, attenuation_db: float = 30.0,
                     channel: int = 0) -> None:
        self._set(self.tx_lo, "powerdown", 0)
        self._set(self.tx_lo, "frequency", int(freq))
        ch = self.tx_chan[channel]
        self._set(ch, "sampling_frequency", int(fs))
        self._set(ch, "rf_bandwidth", int(bw))
        self._set(ch, "hardwaregain", -abs(float(attenuation_db)))
        for c in self.tx_dev.channels:
            if c.id.startswith("altvoltage"):     # DDS tone generators off -> DMA data
                if "raw" in c.attrs:
                    c.attrs["raw"].value = "0"
            elif c.id.startswith("voltage"):
                c.enabled = int(c.id.replace("voltage", "")) // 2 == channel

    def transmit_cyclic(self, iq_int16: np.ndarray) -> None:
        """Play interleaved int16 IQ forever (until :meth:`stop_tx`)."""
        self.stop_tx()
        n = iq_int16.size // 2
        self._tx_buf = self.iio.Buffer(self.tx_dev, n, True)
        self._tx_buf.write(bytearray(np.ascontiguousarray(iq_int16, dtype=np.int16).tobytes()))
        self._tx_buf.push()

    def stop_tx(self) -> None:
        if self._tx_buf is not None:
            self._tx_buf = None               # destroying the cyclic buffer stops the DMA

    def close(self) -> None:
        self._rx_buf = None
        self.stop_tx()
        self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
