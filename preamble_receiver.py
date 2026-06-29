#!/usr/bin/env python3
"""
Receive periodic OFDM preamble from transmit_preamble.py.

Synchronous detection is done by normalized correlation with the known
time-domain preamble (OFDM symbol with CP, built from Zadoff-Chu sequence).
"""
import sys

sys.path.append("/home/pavel/pyadi-iio/")
sys.path.append("/home/pavel/libiio")

import adi
import numpy as np

# --- RF / interface ---
URI_RX = "ip:192.168.2.2"
LO = int(3.19e9)

# Must match transmit_preamble.py
N_FFT = 1024
SUBCARRIER_SPACING_HZ = 30e3
FS = int(N_FFT * SUBCARRIER_SPACING_HZ)  # 30_720_000 Hz
N_CP = 128
N_SC_DATA = 600
ZC_ROOT_CH0 = 7
ZC_ROOT_CH1 = 13
PREAMBLES_PER_PERIOD = 4
PREAMBLES_IN_CHUNK = 20

# RX setup
RX_GAIN_CH0 = 40
RX_GAIN_CH1 = 40
MIN_PEAK_TO_MEDIAN = 6.0
BUFFERS_PER_SECOND_PRINT = 4


def zadoff_chu(length: int, root: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * n * (n + 1.0) / length)


def active_subcarrier_indices(n_fft: int, n_sc: int) -> np.ndarray:
    half = n_sc // 2
    pos = np.arange(1, half + 1, dtype=int)
    neg = np.arange(n_fft - half, n_fft, dtype=int)
    return np.concatenate([pos, neg])


def build_zc_preamble_symbol(n_fft: int, n_cp: int, n_sc_data: int, zc_root: int) -> np.ndarray:
    zc = zadoff_chu(n_sc_data, zc_root)
    idx = active_subcarrier_indices(n_fft, n_sc_data)
    X = np.zeros(n_fft, dtype=np.complex128)
    X[idx] = zc
    x = np.fft.ifft(X, n=n_fft, norm="ortho")
    return np.concatenate([x[-n_cp:], x]).astype(np.complex64)


def norm_corr(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Normalized sliding correlation in [0..1]:
      |sum(rx * conj(ref))| / sqrt(E_rx * E_ref)
    """
    corr = np.convolve(rx, ref[::-1].conj(), mode="valid")
    corr_abs = np.abs(corr)
    ref_energy = float(np.sum(np.abs(ref) ** 2))
    win = np.ones(len(ref), dtype=np.float64)
    rx_energy = np.convolve(np.abs(rx) ** 2, win, mode="valid")
    return corr_abs / (np.sqrt(ref_energy * rx_energy) + 1e-12)


def find_preamble(rx: np.ndarray, ref: np.ndarray) -> tuple[int, float, float]:
    c = norm_corr(rx, ref)
    peak_idx = int(np.argmax(c))
    peak = float(c[peak_idx])
    median = float(np.median(c) + 1e-12)
    return peak_idx, peak, peak / median


def estimate_period_slots(corr: np.ndarray, slot_len: int, expected_slots: int) -> float:
    """
    Estimate inter-peak distance in slot units around strongest peak.
    """
    if len(corr) < 2 * slot_len:
        return float("nan")
    p0 = int(np.argmax(corr))
    search_from = min(len(corr) - 1, p0 + slot_len // 2)
    search_to = min(len(corr), p0 + 2 * expected_slots * slot_len)
    if search_to <= search_from + 1:
        return float("nan")
    rel = int(np.argmax(corr[search_from:search_to]))
    p1 = search_from + rel
    return (p1 - p0) / slot_len


def configure_rx(rx_buffer_size: int) -> adi.ad9361:
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
    return sdr


def main() -> None:
    pre0 = build_zc_preamble_symbol(N_FFT, N_CP, N_SC_DATA, ZC_ROOT_CH0)
    pre1 = build_zc_preamble_symbol(N_FFT, N_CP, N_SC_DATA, ZC_ROOT_CH1)
    slot_len = len(pre0)
    chunk_samples = PREAMBLES_IN_CHUNK * slot_len
    rx_buffer_size = 2 * chunk_samples

    sdr = configure_rx(rx_buffer_size)

    print(
        f"RX @ {LO/1e9:.3f} GHz, Fs={FS/1e6:.3f} Msps, N_FFT={N_FFT}, N_CP={N_CP}, "
        f"slot={slot_len} samples, buffer={rx_buffer_size}"
    )
    print(
        f"Expecting periodic preamble every {PREAMBLES_PER_PERIOD} slots "
        f"(roots: ch0={ZC_ROOT_CH0}, ch1={ZC_ROOT_CH1})."
    )
    print("Press Ctrl+C to stop.\n")

    print_each = max(1, int(FS / rx_buffer_size / BUFFERS_PER_SECOND_PRINT))
    counter = 0

    try:
        while True:
            data = sdr.rx()
            rx0 = data[0].astype(np.complex64)
            rx1 = data[1].astype(np.complex64)
            rx0 = rx0 - np.mean(rx0)
            rx1 = rx1 - np.mean(rx1)

            st00, pk00, rat00 = find_preamble(rx0, pre0)
            st01, pk01, rat01 = find_preamble(rx0, pre1)
            st10, pk10, rat10 = find_preamble(rx1, pre0)
            st11, pk11, rat11 = find_preamble(rx1, pre1)

            # Best root mapping between RX and TX channels.
            score_a = pk00 + pk11  # rx0->root0, rx1->root1
            score_b = pk01 + pk10  # rx0->root1, rx1->root0
            if score_a >= score_b:
                mapping = "rx0->root7, rx1->root13"
                st0, pk0, rat0, c0 = st00, pk00, rat00, norm_corr(rx0, pre0)
                st1, pk1, rat1, c1 = st11, pk11, rat11, norm_corr(rx1, pre1)
            else:
                mapping = "rx0->root13, rx1->root7"
                st0, pk0, rat0, c0 = st01, pk01, rat01, norm_corr(rx0, pre1)
                st1, pk1, rat1, c1 = st10, pk10, rat10, norm_corr(rx1, pre0)

            lock0 = rat0 >= MIN_PEAK_TO_MEDIAN
            lock1 = rat1 >= MIN_PEAK_TO_MEDIAN
            period0 = estimate_period_slots(c0, slot_len, PREAMBLES_PER_PERIOD)
            period1 = estimate_period_slots(c1, slot_len, PREAMBLES_PER_PERIOD)

            counter += 1
            if counter % print_each != 0:
                continue

            status = "LOCK" if (lock0 and lock1) else "SEARCH"
            print(
                f"[{status}] {mapping} | "
                f"ch0: st={st0:6d}, corr={pk0:.3f}, p/m={rat0:5.2f}, T={period0:4.2f} slots | "
                f"ch1: st={st1:6d}, corr={pk1:.3f}, p/m={rat1:5.2f}, T={period1:4.2f} slots"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()