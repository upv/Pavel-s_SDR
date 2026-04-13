#!/usr/bin/env python3
"""
AD9361 2xRX receiver for transmit_zhadoff.py waveform.

Matches TX parameters:
- Fs = 30.72 MHz (N_FFT=1024, subcarrier spacing 30 kHz)
- LO = 3.19 GHz
- OFDM symbol = CP(128) + FFT(1024) [+ optional guard]
- Preamble = 1 Zadoff-Chu OFDM symbol per channel (roots 7 / 13)
- Then QPSK data symbols on 600 active subcarriers
"""
import sys
from typing import Optional

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import matplotlib.pyplot as plt
import numpy as np

# --- RF / interface ---
URI_RX = "ip:192.168.2.2"
LO = int(3.19e9)

# Must match transmitter
N_FFT = 1024
SUBCARRIER_SPACING_HZ = 30e3
FS = int(N_FFT * SUBCARRIER_SPACING_HZ)  # 30_720_000
N_CP = 128
N_GUARD_SAMPLES = 0
N_SC_DATA = 600
ZC_ROOT_CH0 = 7
ZC_ROOT_CH1 = 13
N_DATA_SYMBOLS = 32
PAYLOAD_SEED = 20260413

# Receiver settings
RX_GAIN_CH0 = 40
RX_GAIN_CH1 = 40


def zadoff_chu(length: int, root: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * n * (n + 1.0) / length)


def active_subcarrier_indices(n_fft: int, n_sc: int) -> np.ndarray:
    half = n_sc // 2
    pos = np.arange(1, half + 1, dtype=int)
    neg = np.arange(n_fft - half, n_fft, dtype=int)
    return np.concatenate([pos, neg])


def build_zc_preamble_time(root: int) -> np.ndarray:
    zc = zadoff_chu(N_SC_DATA, root)
    idx = active_subcarrier_indices(N_FFT, N_SC_DATA)
    X = np.zeros(N_FFT, dtype=np.complex128)
    half = N_SC_DATA // 2
    X[idx[:half]] = zc[:half]
    X[idx[half:]] = zc[half:]
    x = np.fft.ifft(X, n=N_FFT, norm="ortho")
    sym = np.concatenate([x[-N_CP:], x])
    if N_GUARD_SAMPLES > 0:
        sym = np.concatenate([sym, np.zeros(N_GUARD_SAMPLES, dtype=np.complex128)])
    return sym.astype(np.complex64)


def qpsk_slicer(sym: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * (np.sign(sym.real) + 1j * np.sign(sym.imag))


def qpsk_map_bits(bits: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1]))


def deterministic_payload_bits(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(N_DATA_SYMBOLS, N_SC_DATA, 2), dtype=np.int8)


def qpsk_bits_from_symbols(sym: np.ndarray) -> np.ndarray:
    bits = np.zeros(sym.shape + (2,), dtype=np.int8)
    bits[..., 0] = (sym.real < 0).astype(np.int8)
    bits[..., 1] = (sym.imag < 0).astype(np.int8)
    return bits


def find_preamble_start(rx: np.ndarray, preamble: np.ndarray) -> tuple[int, float, float]:
    """
    Returns:
        start_idx, peak_value, peak_to_median_ratio
    """
    if len(rx) < len(preamble):
        return 0, 0.0, 0.0
    corr = np.abs(np.convolve(rx, preamble[::-1].conj(), mode="valid"))
    peak_idx = int(np.argmax(corr))
    peak = float(corr[peak_idx])
    med = float(np.median(corr) + 1e-12)
    return peak_idx, peak, peak / med


def extract_data_symbols(
    frame: np.ndarray,
    n_data_symbols: int,
    n_fft: int,
    n_cp: int,
    n_guard_samples: int,
) -> Optional[np.ndarray]:
    sym_len = n_cp + n_fft + n_guard_samples
    out = []
    for i in range(1, n_data_symbols + 1):
        st = i * sym_len
        en = st + sym_len
        if en > len(frame):
            return None
        sym_t = frame[st:en]
        sym_no_cp = sym_t[n_cp : n_cp + n_fft]
        X = np.fft.fft(sym_no_cp, n=n_fft, norm="ortho")
        out.append(X)
    return np.array(out, dtype=np.complex128)


def main() -> None:
    pre0 = build_zc_preamble_time(ZC_ROOT_CH0)
    pre1 = build_zc_preamble_time(ZC_ROOT_CH1)
    idx_data = active_subcarrier_indices(N_FFT, N_SC_DATA)
    ref_bits0 = deterministic_payload_bits(PAYLOAD_SEED + 0)
    ref_bits1 = deterministic_payload_bits(PAYLOAD_SEED + 1)
    ref_qpsk0 = qpsk_map_bits(ref_bits0)
    ref_qpsk1 = qpsk_map_bits(ref_bits1)

    sym_len = N_CP + N_FFT + N_GUARD_SAMPLES
    frame_len = (1 + N_DATA_SYMBOLS) * sym_len
    rx_buffer_size = 2 * frame_len

    sdr = adi.ad9361(URI_RX)
    sdr.sample_rate = FS
    sdr.rx_lo = LO
    sdr.rx_rf_bandwidth = FS
    sdr.rx_buffer_size = rx_buffer_size
    sdr.rx_enabled_channels = [0, 1]
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(RX_GAIN_CH0)
    sdr.rx_hardwaregain_chan1 = int(RX_GAIN_CH1)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line_ber0, = ax.semilogy([], [], label="BER ch0")
    line_ber1, = ax.semilogy([], [], label="BER ch1")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-7, 5e-1)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()

    t_hist: list[float] = []
    ber0_hist: list[float] = []
    ber1_hist: list[float] = []
    t0_wall = None

    print(
        f"RX ready: Fs={FS/1e6:.3f} MHz, LO={LO/1e9:.6f} GHz, "
        f"N_FFT={N_FFT}, N_CP={N_CP}, frame_len={frame_len}, rx_buffer={rx_buffer_size}"
    )
    print(f"Expecting ZC roots: ch0={ZC_ROOT_CH0}, ch1={ZC_ROOT_CH1}")
    print(f"True BER mode: deterministic payload seed={PAYLOAD_SEED} (ch1 uses seed+1).")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = float(np.datetime64("now", "ms").astype("int64")) / 1000.0
            if t0_wall is None:
                t0_wall = now
            t_rel = now - t0_wall

            data = sdr.rx()
            rx0 = data[0].astype(np.complex64)
            rx1 = data[1].astype(np.complex64)

            # Remove DC component before synchronization
            rx0 = rx0 - np.mean(rx0)
            rx1 = rx1 - np.mean(rx1)

            st0, pk0, rat0 = find_preamble_start(rx0, pre0)
            st1, pk1, rat1 = find_preamble_start(rx1, pre1)

            # Require a modest correlation margin to treat as a lock
            lock0 = rat0 > 6.0 and (st0 + frame_len <= len(rx0))
            lock1 = rat1 > 6.0 and (st1 + frame_len <= len(rx1))

            msg = (
                f"ch0: start={st0:6d}, peak/med={rat0:7.2f} | "
                f"ch1: start={st1:6d}, peak/med={rat1:7.2f}"
            )

            if not (lock0 and lock1):
                print(msg + "  [searching]")
                t_hist.append(t_rel)
                ber0_hist.append(np.nan)
                ber1_hist.append(np.nan)
                continue

            fr0 = rx0[st0 : st0 + frame_len]
            fr1 = rx1[st1 : st1 + frame_len]
            data_syms0 = extract_data_symbols(fr0, N_DATA_SYMBOLS, N_FFT, N_CP, N_GUARD_SAMPLES)
            data_syms1 = extract_data_symbols(fr1, N_DATA_SYMBOLS, N_FFT, N_CP, N_GUARD_SAMPLES)
            if data_syms0 is None or data_syms1 is None:
                print(msg + "  [short frame]")
                continue

            sc0 = data_syms0[:, idx_data].reshape(-1)
            sc1 = data_syms1[:, idx_data].reshape(-1)

            dec0 = qpsk_slicer(sc0)
            dec1 = qpsk_slicer(sc1)
            err0 = sc0 - dec0
            err1 = sc1 - dec1
            evm0 = 100.0 * np.sqrt(np.mean(np.abs(err0) ** 2) / (np.mean(np.abs(dec0) ** 2) + 1e-12))
            evm1 = 100.0 * np.sqrt(np.mean(np.abs(err1) ** 2) / (np.mean(np.abs(dec1) ** 2) + 1e-12))

            rx_sc0 = data_syms0[:, idx_data]
            rx_sc1 = data_syms1[:, idx_data]

            # Single-tap per-subcarrier equalization using known deterministic symbols.
            h0 = np.mean(rx_sc0 * np.conj(ref_qpsk0), axis=0) / (
                np.mean(np.abs(ref_qpsk0) ** 2, axis=0) + 1e-12
            )
            h1 = np.mean(rx_sc1 * np.conj(ref_qpsk1), axis=0) / (
                np.mean(np.abs(ref_qpsk1) ** 2, axis=0) + 1e-12
            )
            eq0 = rx_sc0 / (h0[None, :] + 1e-12)
            eq1 = rx_sc1 / (h1[None, :] + 1e-12)

            rx_bits0 = qpsk_bits_from_symbols(eq0)
            rx_bits1 = qpsk_bits_from_symbols(eq1)
            bit_err0 = int(np.count_nonzero(rx_bits0 != ref_bits0))
            bit_err1 = int(np.count_nonzero(rx_bits1 != ref_bits1))
            nbits = int(ref_bits0.size)
            ber0 = bit_err0 / nbits
            ber1 = bit_err1 / nbits

            t_hist.append(t_rel)
            ber0_hist.append(ber0)
            ber1_hist.append(ber1)

            # Keep recent history window for readability
            if len(t_hist) > 300:
                t_hist = t_hist[-300:]
                ber0_hist = ber0_hist[-300:]
                ber1_hist = ber1_hist[-300:]

            line_ber0.set_data(t_hist, ber0_hist)
            line_ber1.set_data(t_hist, ber1_hist)
            if t_hist:
                ax.set_xlim(max(0.0, t_hist[0]), max(1.0, t_hist[-1]))
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                msg
                + f"  [LOCK] EVM: ch0={evm0:6.2f}% ch1={evm1:6.2f}%"
                + f" | BER: ch0={ber0:.3e} ({bit_err0}/{nbits})"
                + f" ch1={ber1:.3e} ({bit_err1}/{nbits})"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
