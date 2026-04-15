#!/usr/bin/env python3
"""
Periodic Zadoff-Chu OFDM preamble TX on AD9361 (2 TX channels).

- Carrier: 3.19 GHz
- OFDM: N_FFT=1024, Δf=30 kHz, Fs=30.72 Msps
- Chunk length: 20 preamble slots (cyclic)
- Preamble period: configurable in "preamble slots"
"""
import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import numpy as np

# --- RF / interface ---
URI_TX = "ip:192.168.2.1"
LO = int(3.19e9)

# OFDM: Δf = Fs / N_FFT  ->  Fs = N_FFT * 30 kHz
N_FFT = 1024
SUBCARRIER_SPACING_HZ = 30e3
FS = int(N_FFT * SUBCARRIER_SPACING_HZ)  # 30_720_000 Hz
N_CP = 128

# Active subcarriers (exclude DC)
N_SC_DATA = 600
ZC_ROOT_CH0 = 7
ZC_ROOT_CH1 = 13

# Transmission structure:
# one slot = one OFDM preamble symbol (CP included)
# we place preamble every PREAMBLES_PER_PERIOD slots, zeros elsewhere.
PREAMBLES_PER_PERIOD = 4
PREAMBLES_IN_CHUNK = 20

TX_SCALE = 1.0 * (2**14)
TX_GAIN_CH0_DB = 0
TX_GAIN_CH1_DB = 0


def zadoff_chu(length: int, root: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * n * (n + 1.0) / length)


def active_subcarrier_indices(n_fft: int, n_sc: int) -> np.ndarray:
    half = n_sc // 2
    pos = np.arange(1, half + 1, dtype=int)
    neg = np.arange(n_fft - half, n_fft, dtype=int)
    return np.concatenate([pos, neg])


def ofdm_symbol_from_freq_bins(freq_bins: np.ndarray, n_fft: int, n_cp: int) -> np.ndarray:
    x = np.fft.ifft(freq_bins, n=n_fft, norm="ortho")
    return np.concatenate([x[-n_cp:], x])


def build_zc_preamble_symbol(n_fft: int, n_cp: int, n_sc_data: int, zc_root: int) -> np.ndarray:
    zc = zadoff_chu(n_sc_data, zc_root)
    idx = active_subcarrier_indices(n_fft, n_sc_data)
    X = np.zeros(n_fft, dtype=np.complex128)
    X[idx] = zc
    return ofdm_symbol_from_freq_bins(X, n_fft, n_cp).astype(np.complex64)


def build_periodic_preamble_chunk(
    preamble_sym: np.ndarray,
    preambles_per_period: int,
    preambles_in_chunk: int,
) -> np.ndarray:
    if preambles_per_period <= 0:
        raise ValueError("PREAMBLES_PER_PERIOD must be > 0")
    if preambles_in_chunk <= 0:
        raise ValueError("PREAMBLES_IN_CHUNK must be > 0")

    slot_len = len(preamble_sym)
    zero_slot = np.zeros(slot_len, dtype=np.complex64)
    slots = []
    for slot_idx in range(preambles_in_chunk):
        if (slot_idx % preambles_per_period) == 0:
            slots.append(preamble_sym)
        else:
            slots.append(zero_slot)
    return np.concatenate(slots)


def configure_sdr() -> adi.ad9361:
    sdr = adi.ad9361(URI_TX)
    sdr.sample_rate = FS
    sdr.tx_lo = LO
    sdr.tx_rf_bandwidth = FS
    sdr.tx_enabled_channels = [0, 1]
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.tx_hardwaregain_chan0 = int(TX_GAIN_CH0_DB)
    sdr.tx_hardwaregain_chan1 = int(TX_GAIN_CH1_DB)
    sdr.tx_cyclic_buffer = True
    return sdr


def main() -> None:
    pre0 = build_zc_preamble_symbol(N_FFT, N_CP, N_SC_DATA, ZC_ROOT_CH0)
    pre1 = build_zc_preamble_symbol(N_FFT, N_CP, N_SC_DATA, ZC_ROOT_CH1)

    tx0 = build_periodic_preamble_chunk(pre0, PREAMBLES_PER_PERIOD, PREAMBLES_IN_CHUNK) * TX_SCALE
    tx1 = build_periodic_preamble_chunk(pre1, PREAMBLES_PER_PERIOD, PREAMBLES_IN_CHUNK) * TX_SCALE

    chunk_samples = len(tx0)
    slot_samples = len(pre0)
    slot_duration_ms = 1e3 * slot_samples / FS
    chunk_duration_ms = 1e3 * chunk_samples / FS

    print(
        f"TX @ {LO/1e9:.3f} GHz, Fs={FS/1e6:.3f} Msps, N_FFT={N_FFT}, N_CP={N_CP}, "
        f"chunk={PREAMBLES_IN_CHUNK} preamble-slots"
    )
    print(
        f"period={PREAMBLES_PER_PERIOD} slots, roots: ch0={ZC_ROOT_CH0}, ch1={ZC_ROOT_CH1}, "
        f"slot={slot_duration_ms:.3f} ms, chunk={chunk_duration_ms:.3f} ms"
    )

    sdr = configure_sdr()
    sdr.tx([tx0, tx1])

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
