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
CFO_MAX_HZ = 100e3


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
        start_idx, peak_norm_corr, peak_to_median_ratio
    """
    if len(rx) < len(preamble):
        return 0, 0.0, 0.0

    # Normalized correlation for robust preamble search under gain variation.
    corr = np.convolve(rx, preamble[::-1].conj(), mode="valid")
    corr_abs = np.abs(corr)
    pre_energy = float(np.sum(np.abs(preamble) ** 2))
    win = np.ones(len(preamble), dtype=np.float64)
    rx_energy = np.convolve(np.abs(rx) ** 2, win, mode="valid")
    norm_corr = corr_abs / (np.sqrt(rx_energy * pre_energy) + 1e-12)

    peak_idx = int(np.argmax(norm_corr))
    peak = float(norm_corr[peak_idx])
    med = float(np.median(norm_corr) + 1e-12)
    return peak_idx, peak, peak / med


def detect_best_preamble(
    rx: np.ndarray, pre_a: np.ndarray, pre_b: np.ndarray
) -> tuple[int, float, float, int, float, float]:
    """
    Returns (st_a, pk_a, rat_a, st_b, pk_b, rat_b) for two candidate preambles.
    """
    st_a, pk_a, rat_a = find_preamble_start(rx, pre_a)
    st_b, pk_b, rat_b = find_preamble_start(rx, pre_b)
    return st_a, pk_a, rat_a, st_b, pk_b, rat_b


def estimate_cfo_hz_from_preamble(rx_pre: np.ndarray, tx_pre: np.ndarray, fs: float) -> float:
    """
    Estimate CFO using phase slope in rx_pre * conj(tx_pre).
    Returns CFO in Hz.
    """
    n = min(len(rx_pre), len(tx_pre))
    if n < 4:
        return 0.0
    z = rx_pre[:n] * np.conj(tx_pre[:n])
    dphi = np.angle(np.sum(z[1:] * np.conj(z[:-1])))
    return float(dphi * fs / (2.0 * np.pi))


def apply_cfo_correction(x: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(-1j * 2.0 * np.pi * cfo_hz * n / fs)
    return x * rot


def equalize_and_detect_bits(rx_sc: np.ndarray, ref_qpsk: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return (eq_symbols, detected_bits, mode)
    mode:
      - "direct": rx ~ h * ref
      - "conjugate": rx ~ h * conj(ref)
    """
    # Hypothesis 1: direct complex path
    h_d = np.mean(rx_sc * np.conj(ref_qpsk), axis=0) / (np.mean(np.abs(ref_qpsk) ** 2, axis=0) + 1e-12)
    eq_d = rx_sc / (h_d[None, :] + 1e-12)
    bits_d = qpsk_bits_from_symbols(eq_d)

    # Hypothesis 2: IQ-conjugated path
    h_c = np.mean(rx_sc * ref_qpsk, axis=0) / (np.mean(np.abs(ref_qpsk) ** 2, axis=0) + 1e-12)
    eq_c = np.conj(rx_sc / (h_c[None, :] + 1e-12))
    bits_c = qpsk_bits_from_symbols(eq_c)

    return eq_d, bits_d, "direct", eq_c, bits_c, "conjugate"


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
    ref_bits0 = deterministic_payload_bits(PAYLOAD_SEED + 0)  # TX ch0 payload
    ref_bits1 = deterministic_payload_bits(PAYLOAD_SEED + 1)  # TX ch1 payload
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
    fig, (ax_ber, ax_cnst) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1.1, 1.0])
    line_ber0, = ax_ber.semilogy([], [], label="cum BER ch0", color="C0")
    line_ber1, = ax_ber.semilogy([], [], label="cum BER ch1", color="C1")
    ax_ber.set_xlabel("Time, s")
    ax_ber.set_ylabel("Cumulative BER (errors / total bits)")
    ax_ber.set_ylim(1e-7, 0.5)
    ax_ber.grid(True, which="both")
    ax_ber.legend(loc="upper right")
    ax_ber.set_title("BER since start (smoothes per-frame noise)")

    sc_plot0 = ax_cnst.scatter([], [], s=4, alpha=0.35, c="C0", label="ch0 (eq)")
    sc_plot1 = ax_cnst.scatter([], [], s=4, alpha=0.35, c="C1", label="ch1 (eq)")
    ax_cnst.set_aspect("equal", adjustable="box")
    ax_cnst.set_xlim(-1.6, 1.6)
    ax_cnst.set_ylim(-1.6, 1.6)
    ax_cnst.axhline(0, color="gray", linewidth=0.5)
    ax_cnst.axvline(0, color="gray", linewidth=0.5)
    ax_cnst.set_xlabel("I (real)")
    ax_cnst.set_ylabel("Q (imag)")
    ax_cnst.set_title("Equalized QPSK symbols (last frame)")
    ax_cnst.legend(loc="upper right", fontsize=8)
    ax_cnst.grid(True, alpha=0.3)
    fig.tight_layout()

    t_hist: list[float] = []
    ber_cum0_hist: list[float] = []
    ber_cum1_hist: list[float] = []
    cum_err0 = 0
    cum_err1 = 0
    cum_bits = 0
    t0_wall = None
    _const_max_pts = 4000

    print(
        f"RX ready: Fs={FS/1e6:.3f} MHz, LO={LO/1e9:.6f} GHz, "
        f"N_FFT={N_FFT}, N_CP={N_CP}, frame_len={frame_len}, rx_buffer={rx_buffer_size}"
    )
    print(f"Expecting ZC roots: tx-ch0={ZC_ROOT_CH0}, tx-ch1={ZC_ROOT_CH1}")
    print(f"True BER mode: deterministic payload seed={PAYLOAD_SEED} (ch1 uses seed+1).")
    print("Auto-mapping enabled: RX channels are matched to TX roots each buffer.")
    print("Each lock: prints BER (frame + cumulative); plot shows BER + constellation.")
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

            st00, pk00, rat00, st01, pk01, rat01 = detect_best_preamble(rx0, pre0, pre1)
            st10, pk10, rat10, st11, pk11, rat11 = detect_best_preamble(rx1, pre0, pre1)

            # Two possible assignments:
            # A: rx0->tx0(root0), rx1->tx1(root1)
            score_a = pk00 + pk11
            # B: rx0->tx1(root1), rx1->tx0(root0)
            score_b = pk01 + pk10

            if score_a >= score_b:
                # mapping: rx0 uses tx0 refs, rx1 uses tx1 refs
                map_name = "A(rx0->tx0, rx1->tx1)"
                st_rx0, pk_rx0, rat_rx0, pre_rx0 = st00, pk00, rat00, pre0
                st_rx1, pk_rx1, rat_rx1, pre_rx1 = st11, pk11, rat11, pre1
                ref_bits_rx0, ref_bits_rx1 = ref_bits0, ref_bits1
                ref_qpsk_rx0, ref_qpsk_rx1 = ref_qpsk0, ref_qpsk1
            else:
                # mapping: rx0 uses tx1 refs, rx1 uses tx0 refs
                map_name = "B(rx0->tx1, rx1->tx0)"
                st_rx0, pk_rx0, rat_rx0, pre_rx0 = st01, pk01, rat01, pre1
                st_rx1, pk_rx1, rat_rx1, pre_rx1 = st10, pk10, rat10, pre0
                ref_bits_rx0, ref_bits_rx1 = ref_bits1, ref_bits0
                ref_qpsk_rx0, ref_qpsk_rx1 = ref_qpsk1, ref_qpsk0

            # Require decent correlation margin and valid frame range.
            lock0 = rat_rx0 > 6.0 and (st_rx0 + frame_len <= len(rx0))
            lock1 = rat_rx1 > 6.0 and (st_rx1 + frame_len <= len(rx1))

            msg = (
                f"{map_name} | "
                f"rx0: start={st_rx0:6d}, peak={pk_rx0:6.3f}, peak/med={rat_rx0:6.2f} | "
                f"rx1: start={st_rx1:6d}, peak={pk_rx1:6.3f}, peak/med={rat_rx1:6.2f}"
            )

            if not (lock0 and lock1):
                print(msg + "  [searching/weak lock]")
                continue

            fr0 = rx0[st_rx0 : st_rx0 + frame_len]
            fr1 = rx1[st_rx1 : st_rx1 + frame_len]

            # Coarse CFO estimation from detected preamble and correction per channel.
            rx_pre0 = fr0[: len(pre_rx0)]
            rx_pre1 = fr1[: len(pre_rx1)]
            cfo0_hz = estimate_cfo_hz_from_preamble(rx_pre0, pre_rx0, FS)
            cfo1_hz = estimate_cfo_hz_from_preamble(rx_pre1, pre_rx1, FS)
            cfo0_hz = float(np.clip(cfo0_hz, -CFO_MAX_HZ, CFO_MAX_HZ))
            cfo1_hz = float(np.clip(cfo1_hz, -CFO_MAX_HZ, CFO_MAX_HZ))
            fr0 = apply_cfo_correction(fr0, cfo0_hz, FS)
            fr1 = apply_cfo_correction(fr1, cfo1_hz, FS)

            data_syms0 = extract_data_symbols(fr0, N_DATA_SYMBOLS, N_FFT, N_CP, N_GUARD_SAMPLES)
            data_syms1 = extract_data_symbols(fr1, N_DATA_SYMBOLS, N_FFT, N_CP, N_GUARD_SAMPLES)
            if data_syms0 is None or data_syms1 is None:
                print(msg + "  [short frame]")
                continue

            sc0_flat = data_syms0[:, idx_data].reshape(-1)
            sc1_flat = data_syms1[:, idx_data].reshape(-1)

            dec0 = qpsk_slicer(sc0_flat)
            dec1 = qpsk_slicer(sc1_flat)
            err0 = sc0_flat - dec0
            err1 = sc1_flat - dec1
            evm0 = 100.0 * np.sqrt(np.mean(np.abs(err0) ** 2) / (np.mean(np.abs(dec0) ** 2) + 1e-12))
            evm1 = 100.0 * np.sqrt(np.mean(np.abs(err1) ** 2) / (np.mean(np.abs(dec1) ** 2) + 1e-12))

            rx_sc0 = data_syms0[:, idx_data]
            rx_sc1 = data_syms1[:, idx_data]

            eq0_d, bits0_d, mode0_d, eq0_c, bits0_c, mode0_c = equalize_and_detect_bits(rx_sc0, ref_qpsk_rx0)
            eq1_d, bits1_d, mode1_d, eq1_c, bits1_c, mode1_c = equalize_and_detect_bits(rx_sc1, ref_qpsk_rx1)

            bit_err0_d = int(np.count_nonzero(bits0_d != ref_bits_rx0))
            bit_err0_c = int(np.count_nonzero(bits0_c != ref_bits_rx0))
            bit_err1_d = int(np.count_nonzero(bits1_d != ref_bits_rx1))
            bit_err1_c = int(np.count_nonzero(bits1_c != ref_bits_rx1))

            if bit_err0_d <= bit_err0_c:
                eq0 = eq0_d
                rx_bits0 = bits0_d
                bit_err0 = bit_err0_d
                mode0 = mode0_d
            else:
                eq0 = eq0_c
                rx_bits0 = bits0_c
                bit_err0 = bit_err0_c
                mode0 = mode0_c

            if bit_err1_d <= bit_err1_c:
                eq1 = eq1_d
                rx_bits1 = bits1_d
                bit_err1 = bit_err1_d
                mode1 = mode1_d
            else:
                eq1 = eq1_c
                rx_bits1 = bits1_c
                bit_err1 = bit_err1_c
                mode1 = mode1_c
            nbits = int(ref_bits_rx0.size)
            ber0 = bit_err0 / nbits
            ber1 = bit_err1 / nbits

            cum_err0 += bit_err0
            cum_err1 += bit_err1
            cum_bits += nbits
            ber_cum0 = cum_err0 / cum_bits
            ber_cum1 = cum_err1 / cum_bits

            t_hist.append(t_rel)
            ber_cum0_hist.append(ber_cum0)
            ber_cum1_hist.append(ber_cum1)

            if len(t_hist) > 400:
                t_hist = t_hist[-400:]
                ber_cum0_hist = ber_cum0_hist[-400:]
                ber_cum1_hist = ber_cum1_hist[-400:]

            line_ber0.set_data(t_hist, ber_cum0_hist)
            line_ber1.set_data(t_hist, ber_cum1_hist)
            if t_hist:
                ax_ber.set_xlim(max(0.0, t_hist[0]), max(1.0, t_hist[-1]))

            z0 = np.column_stack([eq0.real.ravel(), eq0.imag.ravel()])
            z1 = np.column_stack([eq1.real.ravel(), eq1.imag.ravel()])
            if len(z0) > _const_max_pts:
                r0 = np.random.default_rng(0).choice(len(z0), _const_max_pts, replace=False)
                z0 = z0[r0]
            if len(z1) > _const_max_pts:
                r1 = np.random.default_rng(1).choice(len(z1), _const_max_pts, replace=False)
                z1 = z1[r1]
            sc_plot0.set_offsets(z0)
            sc_plot1.set_offsets(z1)

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                msg
                + f"  [LOCK] CFO: ch0={cfo0_hz:+9.1f} Hz ch1={cfo1_hz:+9.1f} Hz"
                + f" | EVM: ch0={evm0:6.2f}% ch1={evm1:6.2f}%"
                + f" | mode: ch0={mode0}, ch1={mode1}"
            )
            print(
                f"  BER frame: ch0={ber0:.6e} ({bit_err0}/{nbits})  "
                f"ch1={ber1:.6e} ({bit_err1}/{nbits})"
            )
            print(
                f"  BER cum:   ch0={ber_cum0:.6e} ({cum_err0}/{cum_bits})  "
                f"ch1={ber_cum1:.6e} ({cum_err1}/{cum_bits})"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
