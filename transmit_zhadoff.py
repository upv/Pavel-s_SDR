#!/usr/bin/env python3
"""
AD9361: OFDM with Zadoff–Chu preamble (2 TX), then QPSK random data.
Subcarrier spacing 30 kHz, N_FFT=1024 → Fs = N * Δf = 30.72 MHz.
Includes cyclic prefix, frequency guard subcarriers, and optional inter-symbol zero guard.
"""
import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import numpy as np

# --- RF / interface ---
URI_TX = "ip:192.168.2.1"
LO = int(3.19e9)

# OFDM: Δf = Fs / N_FFT  →  Fs = N_FFT * 30 kHz
N_FFT = 1024
SUBCARRIER_SPACING_HZ = 30e3
FS = int(N_FFT * SUBCARRIER_SPACING_HZ)  # 30_720_000 Hz

# Time guard: cyclic prefix (copy of tail) + optional extra zero samples between symbols
N_CP = 128
N_GUARD_SAMPLES = 0  # set >0 if you want additional silence between OFDM symbols

# Active subcarriers (exclude DC); symmetric guards at band edges
N_SC_DATA = 600  # 300 positive + 300 negative indices
# ZC roots must be coprime with N_SC_DATA
ZC_ROOT_CH0 = 7
ZC_ROOT_CH1 = 13

# Frame: 1 ZC–OFDM preamble symbol + this many data OFDM symbols (repeated cyclically)
N_DATA_SYMBOLS = 32

# DAC scaling (Pluto / AD9361 style int16 path; pyadi packs complex → int16)
TX_SCALE = 1.0 * (2**14)


def zadoff_chu(length: int, root: int) -> np.ndarray:
    """Prime-length ZC is typical; length must be coprime with root."""
    n = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * n * (n + 1.0) / length)


def active_subcarrier_indices(n_fft: int, n_sc: int) -> tuple[np.ndarray, slice, slice]:
    """
    Return FFT bin indices for n_sc carriers (even split around DC), excluding DC.
    Positive bins: 1 .. n_pos; negative bins: n_fft-n_neg .. n_fft-1
    """
    half = n_sc // 2
    pos = np.arange(1, half + 1, dtype=int)
    neg = np.arange(n_fft - half, n_fft, dtype=int)
    idx = np.concatenate([pos, neg])
    return idx, slice(0, half), slice(half, n_sc)


def freq_frame_to_time(
    X: np.ndarray,
    n_fft: int,
    n_cp: int,
    n_guard_samples: int,
) -> np.ndarray:
    """IFFT, cyclic prefix, optional trailing zero guard."""
    x = np.fft.ifft(X, n=n_fft, norm="ortho")
    sym = np.concatenate([x[-n_cp:], x])
    if n_guard_samples > 0:
        sym = np.concatenate([sym, np.zeros(n_guard_samples, dtype=np.complex128)])
    return sym


def build_zc_preamble(n_fft: int, zc_root: int) -> np.ndarray:
    zc = zadoff_chu(N_SC_DATA, zc_root)
    idx, s_pos, s_neg = active_subcarrier_indices(n_fft, N_SC_DATA)
    X = np.zeros(n_fft, dtype=np.complex128)
    half = N_SC_DATA // 2
    X[idx[:half]] = zc[s_pos]
    X[idx[half:]] = zc[s_neg]
    return freq_frame_to_time(X, n_fft, N_CP, N_GUARD_SAMPLES)


def build_data_symbol(rng: np.random.Generator, n_fft: int) -> np.ndarray:
    """Random bits → QPSK on the same active carriers as preamble."""
    bits = rng.integers(0, 2, size=(N_SC_DATA, 2))
    sym = (1.0 / np.sqrt(2)) * ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1]))
    idx, _, _ = active_subcarrier_indices(n_fft, N_SC_DATA)
    X = np.zeros(n_fft, dtype=np.complex128)
    half = N_SC_DATA // 2
    X[idx[:half]] = sym[:half]
    X[idx[half:]] = sym[half:]
    return freq_frame_to_time(X, n_fft, N_CP, N_GUARD_SAMPLES)


def build_frame_complex64(
    rng: np.random.Generator, zc_root_ch0: int, zc_root_ch1: int
) -> tuple[np.ndarray, np.ndarray]:
    pre0 = build_zc_preamble(N_FFT, zc_root_ch0)
    pre1 = build_zc_preamble(N_FFT, zc_root_ch1)
    syms0 = [pre0]
    syms1 = [pre1]
    for _ in range(N_DATA_SYMBOLS):
        syms0.append(build_data_symbol(rng, N_FFT))
        syms1.append(build_data_symbol(rng, N_FFT))
    t0 = np.concatenate(syms0).astype(np.complex64)
    t1 = np.concatenate(syms1).astype(np.complex64)
    return t0, t1


def main() -> None:
    rng = np.random.default_rng()

    sdr = adi.ad9361(URI_TX)
    sdr.sample_rate = FS
    sdr.tx_lo = LO
    sdr.tx_rf_bandwidth = FS
    sdr.tx_enabled_channels = [0, 1]
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.tx_hardwaregain_chan0 = int(-20)
    sdr.tx_hardwaregain_chan1 = int(-20)
    sdr.tx_cyclic_buffer = True

    t0, t1 = build_frame_complex64(rng, ZC_ROOT_CH0, ZC_ROOT_CH1)
    n = min(len(t0), len(t1))
    t0 = t0[:n] * TX_SCALE
    t1 = t1[:n] * TX_SCALE

    print(
        f"Fs={FS/1e6:.3f} MHz, Δf={SUBCARRIER_SPACING_HZ/1e3:.0f} kHz, N_FFT={N_FFT}, "
        f"N_CP={N_CP}, guard_samples={N_GUARD_SAMPLES}, frame_len={n} samples"
    )
    print(
        f"Preamble: ZC OFDM (roots {ZC_ROOT_CH0} / {ZC_ROOT_CH1}), "
        f"then {N_DATA_SYMBOLS} QPSK data symbols per channel (cyclic TX)."
    )

    sdr.tx([t0, t1])

    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()
