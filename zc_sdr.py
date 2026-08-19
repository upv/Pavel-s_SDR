#!/usr/bin/env python3
"""
Transmit and receive the OFDM Zadoff-Chu pilots from ofdm_zc_detect.py
on the Pluto SDR (USB).

Frame (same as the Q15 file / detector):
    NFFT = 128, CP = 16, 9 symbols @ 1.92 MHz
    ZC length 127, DC nulled, centered subcarrier map
    Pilots on symbols n and n+2 (default 2 and 4)
    Candidate roots {21, 23, 25}
"""

import argparse
import os.path as _ospath
import time

# pylibiio 1.0 parses a Linux soname (libiio.so.1). On macOS find_library
# returns ".../iio.framework/iio", so split(".so.") raises IndexError.
_basename = _ospath.basename


def _libiio_basename(path):
    name = _basename(path)
    if ".so." not in name and (name == "iio" or "libiio" in name or name.endswith(".dylib")):
        return "libiio.so.1"
    return name


_ospath.basename = _libiio_basename

import adi
import iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ofdm_zc_detect import (
    CP,
    FS_BB,
    N_SYMBOLS,
    NFFT,
    ROOTS,
    SYM_LEN,
    interpret,
    pilot_time_symbol,
    search_zc,
    zc_sequence,
)

# Match the original capture: pilots on symbols 2 and 4, root 25.
PILOT_SYMBOLS = (2, 4)
TX_ROOT = 25
CENTER_FREQ = 915e6
TX_GAIN_DB = 0
RX_GAIN_DB = 50
N_FRAME_REPEAT = 16
N_RX_FLUSH = 4


def pluto_uri(fallback="ip:192.168.2.1"):
    """Prefer USB Pluto; IP 192.168.2.1 is often unreachable on this Mac."""
    try:
        for uri, desc in iio.scan_contexts().items():
            if "Pluto" in desc or "0456:b673" in desc:
                print(f"Found Pluto at {uri}")
                return uri
    except Exception:
        pass
    print(f"No USB Pluto found, trying {fallback}")
    return fallback


def zc_freq_bins(root):
    """Centered 128-bin grid: ZC[0:64] | DC=0 | ZC[64:127]."""
    zc = zc_sequence(root)
    X = np.zeros(NFFT, dtype=complex)
    X[0:64] = zc[0:64]
    X[64] = 0.0
    X[65:128] = zc[64:127]
    return X


def qpsk_freq_bins(rng):
    angles = rng.choice([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4], size=NFFT)
    X = np.exp(1j * angles)
    X[64] = 0.0
    return X


def ofdm_symbol_time(freq_centered):
    """IFFT of a centered grid, with cyclic prefix. Matches pilot_time_symbol()."""
    x = np.fft.ifft(np.fft.ifftshift(freq_centered)) * np.sqrt(NFFT)
    return np.concatenate([x[-CP:], x])


def build_frame(root=TX_ROOT, pilot_symbols=PILOT_SYMBOLS, seed=0):
    rng = np.random.default_rng(seed)
    symbols = []
    for n in range(N_SYMBOLS):
        X = zc_freq_bins(root) if n in pilot_symbols else qpsk_freq_bins(rng)
        symbols.append(ofdm_symbol_time(X))
    return np.concatenate(symbols)


def scale_for_pluto(iq, peak=2**14):
    m = np.max(np.abs(iq))
    if m < 1e-12:
        return iq.astype(np.complex64)
    return (iq / m * peak).astype(np.complex64)


def configure_sdr(sdr, fs, rx_buf, tx_gain, rx_gain):
    sdr.sample_rate = int(fs)
    sdr.tx_enabled_channels = [0]
    sdr.rx_enabled_channels = [0]
    sdr.tx_rf_bandwidth = int(fs)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.tx_hardwaregain_chan0 = int(tx_gain)
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(rx_buf)
    sdr.tx_cyclic_buffer = True


def _disable_dds(sdr):
    if hasattr(sdr, "disable_dds"):
        sdr.disable_dds()
        return
    for ch in sdr._txdac.channels:
        if ch.output and "raw" in ch.attrs:
            try:
                ch.attrs["raw"].value = "0"
            except Exception:
                pass


def _iq_to_i16(iq):
    packed = np.empty(len(iq) * 2, dtype=np.int16)
    packed[0::2] = np.real(iq).astype(np.int16)
    packed[1::2] = np.imag(iq).astype(np.int16)
    return packed


def _channel_pair(dev, names, output):
    return [dev.find_channel(names[0], output), dev.find_channel(names[1], output)]


def _tx_rx_libiio1(sdr, tx_iq, rx_len, n_flush=N_RX_FLUSH):
    """Cyclic TX + RX using pylibiio 1.0 Buffer(device, mask), not Device.get_buffer()."""
    _disable_dds(sdr)
    dac, adc = sdr._txdac, sdr._rxadc
    tx_mask = iio.ChannelsMask(dac)
    tx_mask.channels = _channel_pair(dac, sdr._tx_channel_names, True)
    rx_mask = iio.ChannelsMask(adc)
    rx_mask.channels = _channel_pair(adc, sdr._rx_channel_names, False)

    packed = _iq_to_i16(tx_iq)
    nbytes = tx_mask.sample_size * len(tx_iq)
    if packed.nbytes != nbytes:
        raise RuntimeError(
            f"TX packing mismatch: packed {packed.nbytes} B vs mask {nbytes} B"
        )

    tx_buf = iio.Buffer(dac, tx_mask)
    tx_block = iio.Block(tx_buf, nbytes)
    tx_block.write(bytearray(packed))
    tx_block.enqueue(cyclic=True)
    tx_buf.enabled = True
    # Keep these alive: cyclic DMA stops if they are garbage-collected.
    sdr._zc_tx = (tx_mask, tx_buf, tx_block)

    rx_buf = iio.Buffer(adc, rx_mask)
    rx_stream = iio.Stream(rx_buf, rx_len)
    block = None
    for _ in range(n_flush + 1):
        block = next(rx_stream)
    i = np.frombuffer(rx_mask.channels[0].read(block), dtype=np.int16)
    q = np.frombuffer(rx_mask.channels[1].read(block), dtype=np.int16)
    n = min(len(i), len(q))
    return i[:n].astype(np.float64) + 1j * q[:n].astype(np.float64)


def rx_complex(sdr):
    raw = sdr.rx()
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return np.asarray(raw, dtype=np.complex128)


def pluto_tx_rx(sdr, tx_iq, rx_len):
    """TX then RX. pyadi 0.0.21 calls Device.get_buffer() (newer libiio 1.x);
    this venv's pylibiio still uses Buffer(device, mask)."""
    if hasattr(sdr._txdac, "get_buffer"):
        sdr.tx(tx_iq)
        time.sleep(0.15)
        for _ in range(N_RX_FLUSH):
            sdr.rx()
        return rx_complex(sdr)
    print("Using pylibiio 1.0 Buffer API (Device.get_buffer is unavailable)")
    return _tx_rx_libiio1(sdr, tx_iq, rx_len)


def plot_detection(results, det, path="zc_sdr_correlation.png"):
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(11, 8), sharex=True)
    for ax, root in zip(axes, ROOTS):
        c = results[root]["corr"]
        ax.plot(c, lw=0.7)
        ax.set_ylabel(f"root {root}")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if root == det["root"]:
            ax.plot(det["pilot1_pos"], c[det["pilot1_pos"]], "rv", ms=10,
                    label=f"pilot 1 @ {det['pilot1_pos']}")
            if det["pilot2_pos"] is not None:
                ax.plot(det["pilot2_pos"], c[det["pilot2_pos"]], "g^", ms=10,
                        label=f"pilot 2 @ {det['pilot2_pos']}")
            ax.legend(loc="upper right")
            ax.set_title(
                f"root {root}  <-- DETECTED (peak {det['peak_val']:.3f})",
                color="crimson",
            )
    axes[-1].set_xlabel(f"sample index @ {FS_BB/1e6:.2f} MHz")
    fig.suptitle("Pluto loopback: normalized correlation vs ZC pilot symbols")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved plot: {path}")


def print_detection(results, det):
    print("\nPeak normalized correlation per candidate root:")
    for root in ROOTS:
        r = results[root]
        print(f"    root {root}: peak = {r['peak']:.3f} at sample {r['idx']}"
              f"  (spectrum layout: {'centered' if r['centered'] else 'raw'})")
    print(f"\n  -> detected root index : {det['root']}")
    print(f"  -> pilot #1 FFT-part starts at sample {det['pilot1_pos']}")
    if det["pilot2_pos"] is not None:
        print(f"  -> pilot #2 confirmed at sample {det['pilot2_pos']} "
              f"(= pilot#1 + 2 symbols)")
    n1, n2 = det["pilot_symbols"]
    print(f"  -> ZC pilots occupy OFDM symbols {n1} and {n2} (0-based)")
    print(f"  -> estimated frame start (incl. CP): sample {det['frame_start']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, default=TX_ROOT, choices=ROOTS)
    parser.add_argument("--uri", default=None, help="IIO URI (default: auto USB Pluto)")
    parser.add_argument("--tx-gain", type=int, default=TX_GAIN_DB)
    parser.add_argument("--rx-gain", type=int, default=RX_GAIN_DB)
    args = parser.parse_args()

    frame = build_frame(root=args.root)
    assert len(frame) == N_SYMBOLS * SYM_LEN
    # Guard so the detector can see an energy onset before the first symbol.
    guard = np.zeros(2 * SYM_LEN, dtype=complex)
    tx_bb = np.concatenate([guard, np.tile(frame, N_FRAME_REPEAT)])
    tx = scale_for_pluto(tx_bb)

    # Sanity: local replica of the useful (no-CP) pilot must match the detector.
    ref = pilot_time_symbol(args.root, centered=True)
    local = frame[PILOT_SYMBOLS[0] * SYM_LEN + CP : PILOT_SYMBOLS[0] * SYM_LEN + CP + NFFT]
    assert np.allclose(local, ref), "TX pilot does not match ofdm_zc_detect replica"

    fs = int(FS_BB)
    rx_buf = max(len(tx), 8 * N_SYMBOLS * SYM_LEN)

    uri = args.uri or pluto_uri()
    sdr = adi.ad9363(uri)
    configure_sdr(sdr, fs, rx_buf, args.tx_gain, args.rx_gain)
    print(f"TX {len(tx)} samples @ {fs/1e6:.2f} Msps, LO {CENTER_FREQ/1e6:.1f} MHz, "
          f"root {args.root}, pilots on symbols {PILOT_SYMBOLS}")
    print(f"TX gain {args.tx_gain} dB, RX gain {args.rx_gain} dB, rx_buffer {rx_buf}")

    try:
        rx = pluto_tx_rx(sdr, tx, rx_buf)
    finally:
        tx_keep = getattr(sdr, "_zc_tx", None)
        if tx_keep is not None:
            try:
                tx_keep[1].enabled = False
            except Exception:
                pass
            sdr._zc_tx = None
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    rx = rx - np.mean(rx)
    print(f"RX {len(rx)} samples, |z| max = {np.max(np.abs(rx)):.1f} (DC removed)")

    results = search_zc(rx)
    det = interpret(results, rx)
    print_detection(results, det)
    plot_detection(results, det)

    if det["root"] != args.root or det["peak_val"] < 0.5:
        print("\nWarning: weak or wrong-root detection. Try a TX→RX SMA cable, "
              "or raise --tx-gain / --rx-gain.")


if __name__ == "__main__":
    main()
