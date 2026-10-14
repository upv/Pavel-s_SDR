"""Frame detection, timing and CFO for the PUSCH frame, NumPy only.

The receiver knows nothing but the DMRS and the frame layout:

1. Segmented cross-correlation of the capture with the time-domain DMRS symbol
   (``Frame.dmrs_ref``): four 256-sample partial correlations summed in
   magnitude, so the peak survives a CFO of up to ±60 kHz. The DMRS symbol is
   comb-2 (period 512), so the correlation has ±512-sample sidelobes.
2. The cyclic-prefix metric over all 14 symbols of the candidate slot resolves
   the ±512 ambiguity and gives the fractional CFO (±15 kHz, one SCS).
3. The phase progression between the partial correlations gives a coarse CFO
   (±60 kHz) whose integer-SCS part is combined with the fractional one.
4. The gap after the slot measures the noise variance per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .waveform import Frame


@dataclass
class SyncResult:
    slot_start: int           # sample index of the slot (first CP sample) in the capture
    cfo_hz: float             # estimated carrier frequency offset, applied to the capture
    cfo_frac_hz: float        # CP-based fractional part
    cfo_coarse_hz: float      # DMRS partial-correlation estimate
    dmrs_metric: float        # normalised DMRS correlation at the chosen timing (0..1)
    cp_metric: float          # normalised CP correlation of the slot (0..1)
    noise_var: float          # per-sample noise variance from the gap (NaN if not measurable)
    signal_power: float       # mean |r|^2 over the slot
    extra: dict = field(default_factory=dict)

    @property
    def snr_db(self) -> float:
        if not np.isfinite(self.noise_var) or self.noise_var <= 0:
            return float("nan")
        return float(10 * np.log10(max(self.signal_power - self.noise_var, 1e-30) / self.noise_var))


def _xcorr(r: np.ndarray, ref: np.ndarray, n_out: int) -> np.ndarray:
    """c[d] = sum_n r[d+n] conj(ref[n]) for d in [0, n_out) via FFT."""
    n = 1 << int(np.ceil(np.log2(r.size + ref.size)))
    R = np.fft.fft(r, n)
    K = np.fft.fft(ref, n)
    return np.fft.ifft(R * np.conj(K))[:n_out]


def dmrs_correlation(r: np.ndarray, ref: np.ndarray, num_segments: int = 4):
    """Segmented, energy-normalised correlation metric and the partial correlations.

    Returns ``(metric [n_out], partial [num_segments, n_out])`` where ``n_out`` is the
    number of start positions that keep the whole reference inside ``r``.
    """
    L = ref.size
    S = L // num_segments
    n_out = r.size - L + 1
    partial = np.empty((num_segments, n_out), dtype=np.complex128)
    for i in range(num_segments):
        seg = np.zeros(L, dtype=np.complex128)
        seg[i * S:(i + 1) * S] = ref[i * S:(i + 1) * S]
        partial[i] = _xcorr(r, seg, n_out)
    energy = np.convolve(np.abs(r) ** 2, np.ones(L), mode="valid")[:n_out]
    metric = np.abs(partial).sum(axis=0) / np.sqrt(np.maximum(energy, 1e-12) * np.sum(np.abs(ref) ** 2))
    return metric, partial


def cp_correlation(r: np.ndarray, start: int, frame: Frame) -> tuple[complex, float]:
    """Sum of CP x tail correlations over the slot at ``start``; ``(complex sum, normalised |.|)``."""
    N, cp, L = frame.fft_size, frame.cp, frame.sym_len
    acc = 0j
    energy = 0.0
    for l in range(frame.num_symbols):
        s = start + l * L
        a = r[s:s + cp]
        b = r[s + N:s + N + cp]
        acc += np.vdot(b, a)                       # sum a * conj(b)
        energy += 0.5 * (np.sum(np.abs(a) ** 2) + np.sum(np.abs(b) ** 2))
    return acc, float(abs(acc) / max(energy, 1e-12))


def find_frames(r: np.ndarray, frame: Frame, min_metric: float = 0.3,
                rel_threshold: float = 0.5, max_frames: int | None = None) -> list[SyncResult]:
    """Detect every complete frame in capture ``r`` (one RX antenna, complex) and sync each.

    Candidates come from the DMRS correlation peaks; each is refined with the CP
    metric over the ±512-sample comb ambiguity. Only slots whose gap also lies inside
    the capture are returned (the gap is needed for the noise estimate).
    """
    r = np.asarray(r, dtype=np.complex128)
    fs, N, cp, L = frame.fs, frame.fft_size, frame.cp, frame.sym_len
    scs = fs / N
    d0 = frame.dmrs_offsets()[0]                   # ref position relative to the slot start
    metric, partial = dmrs_correlation(r, frame.dmrs_ref)
    S = frame.dmrs_ref.size // partial.shape[0]

    thr = max(min_metric, rel_threshold * float(metric.max()))
    order = np.argsort(metric)[::-1]
    peaks: list[int] = []
    for p in order:
        if metric[p] < thr:
            break
        if all(abs(p - q) > frame.frame_len // 2 for q in peaks):
            peaks.append(int(p))
    peaks.sort()

    results = []
    for p in peaks:
        best = None
        for shift in (-N // 2, 0, N // 2):           # comb-2 ambiguity of the DMRS symbol
            start = p + shift - d0
            if start < 0 or start + frame.frame_len > r.size:
                continue
            acc, cpm = cp_correlation(r, start, frame)
            if best is None or cpm > best[1]:
                best = (start, cpm, acc, p + shift)
        if best is None:
            continue
        start, cpm, acc, p_ref = best
        if p_ref >= metric.size:
            continue
        # fractional CFO from the CP: a * conj(b) ~ exp(-j 2 pi f N / fs)
        cfo_frac = -np.angle(acc) / (2 * np.pi) * scs
        # coarse CFO from the phase advance between consecutive partial correlations
        z = partial[:, p_ref]
        dphi = np.angle(np.sum(z[1:] * np.conj(z[:-1])))
        cfo_coarse = dphi / (2 * np.pi) * fs / S
        cfo = cfo_frac + scs * np.round((cfo_coarse - cfo_frac) / scs)

        slot = r[start:start + frame.slot_len]
        g0, g1 = start + frame.slot_len + 64, start + frame.frame_len - 64
        noise_var = float(np.mean(np.abs(r[g0:g1]) ** 2)) if g1 - g0 >= 256 else float("nan")
        results.append(SyncResult(
            slot_start=int(start), cfo_hz=float(cfo), cfo_frac_hz=float(cfo_frac),
            cfo_coarse_hz=float(cfo_coarse), dmrs_metric=float(metric[p_ref]), cp_metric=cpm,
            noise_var=noise_var, signal_power=float(np.mean(np.abs(slot) ** 2)),
            extra={"metric": metric, "peak": p_ref},
        ))
        if max_frames and len(results) >= max_frames:
            break
    return results


def correct_cfo(r: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    """Remove a carrier offset: r * exp(-j 2 pi f n / fs), n counted from the capture start."""
    n = np.arange(r.shape[-1])
    return r * np.exp(-2j * np.pi * cfo_hz * n / fs)


def ofdm_demodulate(r: np.ndarray, frame: Frame, slot_start: int, timing_advance: int = 16):
    """Capture [rx, n] -> Sionna grid ``y`` [1, 1, rx, symbols, fft] (torch), window advanced by ``timing_advance``.

    The advance moves the FFT window into the cyclic prefix so a channel with
    (slightly) negative taps or a timing error of a few samples is still cyclic;
    the demodulator compensates the resulting linear phase (``l_min``).
    """
    import torch
    from sionna.phy.ofdm import OFDMDemodulator

    r = np.atleast_2d(r)
    ta = int(min(timing_advance, frame.cp - 1, slot_start))
    s = slot_start - ta
    seg = r[:, s:s + frame.slot_len].astype(np.complex64)
    dem = OFDMDemodulator(frame.fft_size, -ta, frame.cp)
    y = dem(torch.as_tensor(seg)[None, None])     # [1, 1, rx, sym, fft]
    return y, ta
