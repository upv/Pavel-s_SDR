# -*- coding: utf-8 -*-
"""
Shared module for the E310 <-> E310 link:
  - OFDM parameters (15 kHz SCS, FFT 1024, fs = 15.36 MHz)
  - Zadoff-Chu generation (cyclic extension, as in NR low-PAPR sequences, TS 38.211 5.2.2)
  - frame build: SC preamble -> fine-timing ZC preamble -> pilot symbols (comb-2 ZC + QPSK data)
  - full receiver chain: coarse timing + fractional CFO (Schmidl-Cox), integer CFO
    (differential frequency correlation), fine timing (cross-correlation with ZC),
    LS channel estimate, residual CFO / phase tracking, delay from phase slope, EVM.

NumPy only; hardware is attached in tx_e310.py / rx_e310.py.
"""

from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------- parameters

@dataclass
class Cfg:
    N_FFT: int = 1024
    CP: int = 72                 # simplification: same CP on every symbol
    SCS: float = 15e3
    N_ACT: int = 600             # 50 RB, DC unused
    N_PILOT_SYMS: int = 8        # symbols with comb-2 pilots + data
    GAP: int = 240               # trailing zeros (multiple of 16 for DMA)
    ZC_ROOT_SC: int = 7          # ZC root for the SC preamble
    ZC_ROOT_FINE: int = 25       # ZC root for the fine-timing preamble
    ZC_ROOT_PILOT: int = 11      # ZC root for pilots
    DATA_SEED: int = 2026        # receiver knows the data -> can measure EVM
    TIMING_ADVANCE: int = 8      # shift FFT window into the CP (late-path guard)

    @property
    def FS(self) -> float:
        return self.N_FFT * self.SCS          # 15.36 MHz

    @property
    def SYM_LEN(self) -> int:
        return self.N_FFT + self.CP

    @property
    def N_SYMS(self) -> int:
        return 2 + self.N_PILOT_SYMS          # SC + fine + pilots

    @property
    def FRAME_LEN(self) -> int:
        return self.N_SYMS * self.SYM_LEN + self.GAP


# ----------------------------------------------------------------------------- utilities

def largest_prime_below(n: int) -> int:
    def is_prime(m):
        if m < 2:
            return False
        for p in range(2, int(m ** 0.5) + 1):
            if m % p == 0:
                return False
        return True
    m = n
    while not is_prime(m):
        m -= 1
    return m


def zadoff_chu(root: int, length: int) -> np.ndarray:
    """Base ZC of odd prime length: x_q(m) = exp(-j*pi*q*m*(m+1)/N_zc)."""
    m = np.arange(length)
    return np.exp(-1j * np.pi * root * m * (m + 1) / length)


def zc_extended(root: int, target_len: int) -> np.ndarray:
    """Cyclic extension of a prime-length ZC to target_len — as in NR (38.211 5.2.2.1)."""
    n_zc = largest_prime_below(target_len)
    base = zadoff_chu(root, n_zc)
    return base[np.arange(target_len) % n_zc]


def active_offsets(cfg: Cfg) -> np.ndarray:
    """Occupied subcarrier offsets relative to DC: -300..-1, 1..300 (DC skipped)."""
    half = cfg.N_ACT // 2
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


def offsets_to_bins(offsets: np.ndarray, cfg: Cfg) -> np.ndarray:
    return np.mod(offsets, cfg.N_FFT)


def ofdm_symbol(freq_vals: np.ndarray, offsets: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Freq -> time with CP; normalize to unit mean power."""
    X = np.zeros(cfg.N_FFT, dtype=complex)
    X[offsets_to_bins(offsets, cfg)] = freq_vals
    x = np.fft.ifft(X) * cfg.N_FFT / np.sqrt(len(offsets))
    return np.concatenate([x[-cfg.CP:], x])


# ----------------------------------------------------------------------------- frame build

@dataclass
class FrameRef:
    """Everything the receiver needs: preamble, pilot, and data references."""
    cfg: Cfg
    offsets: np.ndarray            # all occupied offsets
    sc_offsets: np.ndarray         # even offsets (SC preamble)
    pilot_offsets: np.ndarray      # comb-2: every other occupied offset
    data_offsets: np.ndarray       # the rest
    sc_freq: np.ndarray
    fine_freq: np.ndarray          # ZC on all occupied subcarriers
    fine_td: np.ndarray            # time-domain useful part (no CP) for cross-correlation
    pilot_freq: np.ndarray
    data_syms: np.ndarray          # [N_PILOT_SYMS, len(data_offsets)] QPSK
    frame: np.ndarray              # assembled time-domain frame


def build_frame(cfg: Cfg = Cfg()) -> FrameRef:
    offs = active_offsets(cfg)

    # SC preamble: even offsets only -> two identical halves in time
    sc_offs = offs[offs % 2 == 0]
    sc_freq = zc_extended(cfg.ZC_ROOT_SC, len(sc_offs))
    sym_sc = ofdm_symbol(sc_freq, sc_offs, cfg)

    # fine-timing preamble: ZC on all 600 subcarriers
    fine_freq = zc_extended(cfg.ZC_ROOT_FINE, len(offs))
    sym_fine = ofdm_symbol(fine_freq, offs, cfg)
    fine_td = sym_fine[cfg.CP:]

    # pilot symbols: comb-2 ZC + QPSK data on the other comb
    pilot_offs = offs[0::2]
    data_offs = offs[1::2]
    pilot_freq = zc_extended(cfg.ZC_ROOT_PILOT, len(pilot_offs))

    rng = np.random.default_rng(cfg.DATA_SEED)
    bits = rng.integers(0, 2, size=(cfg.N_PILOT_SYMS, len(data_offs), 2))
    data_syms = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2)

    body = [sym_sc, sym_fine]
    for s in range(cfg.N_PILOT_SYMS):
        vals = np.zeros(len(offs), dtype=complex)
        vals[0::2] = pilot_freq
        vals[1::2] = data_syms[s]
        body.append(ofdm_symbol(vals, offs, cfg))

    frame = np.concatenate(body + [np.zeros(cfg.GAP, dtype=complex)])
    # PAPR headroom: leave peak at ~0.5 full scale to the TX script
    return FrameRef(cfg, offs, sc_offs, pilot_offs, data_offs,
                    sc_freq, fine_freq, fine_td, pilot_freq, data_syms, frame)


# ----------------------------------------------------------------------------- receiver

@dataclass
class RxResult:
    coarse_idx: int = 0
    frame_start: int = 0
    cfo_frac_sc: float = 0.0        # fractional CFO (subcarriers) from Schmidl-Cox
    cfo_int_sc: int = 0             # integer CFO (subcarriers)
    cfo_hz: float = 0.0             # total CFO
    cfo_resid_hz: float = 0.0       # residual CFO from pilot phase tracking
    delay_frac_samples: float = 0.0 # residual delay from the phase slope of H
    snr_db: float = 0.0             # SNR from pilots
    evm_db: float = 0.0             # data EVM after equalization
    H: np.ndarray = field(default_factory=lambda: np.zeros(0))   # mean channel on pilot subcarriers
    H_syms: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per-symbol channel [S, K]
    eq_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sc_metric: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _schmidl_cox(rx: np.ndarray, cfg: Cfg):
    """Metric M(d)=|P|^2/R^2 from half-symbol repetition; returns metric, coarse index, fractional CFO."""
    L = cfg.N_FFT // 2
    a = rx[:-L] if L else rx
    b = rx[L:]
    p = np.conj(a) * b
    q = np.abs(b) ** 2
    win = np.ones(L)
    P = np.convolve(p, win, mode='valid')
    R = np.convolve(q, win, mode='valid') + 1e-12
    M = np.abs(P) ** 2 / R ** 2

    d_max = int(np.argmax(M))
    # plateau midpoint (>=90% of the peak) — more stable than argmax alone
    thr = 0.9 * M[d_max]
    lo = d_max
    while lo > 0 and M[lo - 1] >= thr:
        lo -= 1
    hi = d_max
    while hi < len(M) - 1 and M[hi + 1] >= thr:
        hi += 1
    d = (lo + hi) // 2
    eps_frac = np.angle(P[d]) / np.pi   # in subcarriers, range +-1
    return M, d, eps_frac


def _apply_cfo(rx: np.ndarray, eps_subcarriers: float, cfg: Cfg) -> np.ndarray:
    n = np.arange(len(rx))
    return rx * np.exp(-2j * np.pi * eps_subcarriers * n / cfg.N_FFT)


def _integer_cfo(rx: np.ndarray, fine_start_coarse: int, ref: FrameRef, max_shift: int = 40) -> int:
    """Integer CFO from the fine-timing preamble.

    Differential frequency correlation: Z[k] = Y[k]*conj(Y[k+delta]) is independent of
    the linear phase ramp from a timing error (it becomes a common factor), so this
    still works when coarse timing is off by about a CP.
    """
    cfg = ref.cfg
    seg = rx[fine_start_coarse:fine_start_coarse + cfg.N_FFT]
    if len(seg) < cfg.N_FFT:
        return 0
    Y = np.fft.fft(seg)

    bins = offsets_to_bins(ref.offsets, cfg)
    X = np.zeros(cfg.N_FFT, dtype=complex)
    X[bins] = ref.fine_freq

    # differentials on neighbouring occupied bins (step 1 inside each half of the spectrum)
    d_ref = X[bins] * np.conj(np.roll(X, -1)[bins])
    best, best_val = 0, -1.0
    for s in range(-max_shift, max_shift + 1):
        Ys = np.roll(Y, -s)
        d_rx = Ys[bins] * np.conj(np.roll(Ys, -1)[bins])
        val = np.abs(np.vdot(d_ref, d_rx))
        if val > best_val:
            best_val, best = val, s
    return best


def _fine_timing(rx: np.ndarray, expect: int, ref: FrameRef, search: int) -> int:
    cfg = ref.cfg
    lo = max(0, expect - search)
    hi = min(len(rx) - cfg.N_FFT, expect + search)
    tpl = ref.fine_td
    best, best_val = lo, -1.0
    for d in range(lo, hi):
        val = np.abs(np.vdot(tpl, rx[d:d + cfg.N_FFT]))
        if val > best_val:
            best_val, best = val, d
    return best


def process_capture(rx: np.ndarray, ref: FrameRef) -> RxResult:
    cfg = ref.cfg
    res = RxResult()
    rx = rx.astype(complex) / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-12)

    # --- 1. coarse timing + fractional CFO (Schmidl-Cox, CFO-insensitive)
    M, d, eps_frac = _schmidl_cox(rx, cfg)
    res.sc_metric, res.coarse_idx, res.cfo_frac_sc = M, d, eps_frac
    rx1 = _apply_cfo(rx, eps_frac, cfg)

    # --- 2. integer CFO (before ZC cross-correlation, else delay-Doppler shifts the peak)
    fine_coarse = d + cfg.SYM_LEN            # coarse position of the fine-preamble useful part
    eps_int = _integer_cfo(rx1, fine_coarse, ref)
    res.cfo_int_sc = eps_int
    rx2 = _apply_cfo(rx1, eps_int, cfg)
    res.cfo_hz = (eps_frac + eps_int) * cfg.SCS

    # --- 3. fine timing by ZC cross-correlation (CFO now compensated)
    fine_idx = _fine_timing(rx2, fine_coarse, ref, search=2 * cfg.CP)
    frame_start = fine_idx - cfg.SYM_LEN - cfg.CP
    res.frame_start = frame_start
    if frame_start < 0 or frame_start + cfg.FRAME_LEN > len(rx2):
        raise RuntimeError("frame does not fit in the capture: increase rx_buffer_size "
                           "or check that the transmitter is running")

    # --- 4. demodulate pilot symbols (FFT window shifted into the CP)
    adv = cfg.TIMING_ADVANCE
    pil_bins = offsets_to_bins(ref.pilot_offsets, cfg)
    dat_bins = offsets_to_bins(ref.data_offsets, cfg)
    n_rot = np.exp(2j * np.pi * adv * np.arange(cfg.N_FFT) / cfg.N_FFT)  # undo the window shift

    H_syms, Y_data = [], []
    for s in range(cfg.N_PILOT_SYMS):
        st = frame_start + (2 + s) * cfg.SYM_LEN + cfg.CP - adv
        Y = np.fft.fft(rx2[st:st + cfg.N_FFT]) * n_rot * np.sqrt(cfg.N_ACT) / cfg.N_FFT
        H_syms.append(Y[pil_bins] / ref.pilot_freq)
        Y_data.append(Y[dat_bins])
    H_syms = np.array(H_syms)
    Y_data = np.array(Y_data)

    # --- 5. phase tracking: common symbol-to-symbol phase ramp -> residual CFO
    ph = np.angle(np.sum(H_syms * np.conj(H_syms[0]), axis=1))
    ph = np.unwrap(ph)
    sym_t = cfg.SYM_LEN / cfg.FS
    slope = np.polyfit(np.arange(cfg.N_PILOT_SYMS), ph, 1)[0]
    res.cfo_resid_hz = slope / (2 * np.pi * sym_t)
    corr = np.exp(-1j * ph)[:, None]
    H_syms_c = H_syms * corr
    Y_data_c = Y_data * corr

    # --- 6. residual delay: slope of phase H vs frequency (comb step = 2 bins)
    pairs = np.sum(H_syms_c[:, 1:] * np.conj(H_syms_c[:, :-1]))
    ph_per_bin = np.angle(pairs) / 2.0
    res.delay_frac_samples = -ph_per_bin * cfg.N_FFT / (2 * np.pi)

    # --- 7. channel, SNR, data equalization, EVM
    H = np.mean(H_syms_c, axis=0)
    res.H, res.H_syms = H, H_syms_c
    noise = H_syms_c - H[None, :]
    res.snr_db = 10 * np.log10(np.mean(np.abs(H) ** 2) /
                               (np.mean(np.abs(noise) ** 2) + 1e-15))

    # linear interpolation of H onto data subcarriers
    H_data = (np.interp(ref.data_offsets, ref.pilot_offsets, H.real) +
              1j * np.interp(ref.data_offsets, ref.pilot_offsets, H.imag))
    eq = Y_data_c / H_data[None, :]
    res.eq_data = eq
    err = eq - ref.data_syms
    res.evm_db = 10 * np.log10(np.mean(np.abs(err) ** 2) /
                               np.mean(np.abs(ref.data_syms) ** 2))
    return res


def summary(res: RxResult, cfg: Cfg) -> str:
    return "\n".join([
        f"coarse index (SC)        : {res.coarse_idx}",
        f"frame start              : {res.frame_start}",
        f"CFO fractional / integer : {res.cfo_frac_sc:+.3f} / {res.cfo_int_sc:+d} subcarriers",
        f"CFO total                : {res.cfo_hz / 1e3:+.3f} kHz",
        f"CFO residual (tracking)  : {res.cfo_resid_hz:+.1f} Hz",
        f"residual delay           : {res.delay_frac_samples:+.2f} samples "
        f"({res.delay_frac_samples / cfg.FS * 1e9:+.0f} ns)",
        f"pilot SNR                : {res.snr_db:.1f} dB",
        f"data EVM                 : {res.evm_db:.1f} dB",
    ])
