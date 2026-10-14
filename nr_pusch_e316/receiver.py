"""The receiver: capture -> sync -> OFDM demod -> WirelessNN channel estimation, equalisation, LDPC.

:func:`process_capture` runs every frame found in a capture through the chain and
returns one :class:`SlotResult` per frame; :func:`summarize` aggregates BER / BLER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sync import SyncResult, correct_cfo, find_frames, ofdm_demodulate
from .waveform import Frame


@dataclass
class SlotResult:
    sync: SyncResult
    noise_var: float                       # per-RE noise variance used by the receiver
    snr_db: float                          # mean |h_hat|^2 / noise_var on the used band
    bit_errors: dict = field(default_factory=dict)   # (estimator, receiver) -> errors
    block_errors: dict = field(default_factory=dict)
    evm_db: dict = field(default_factory=dict)       # estimator -> EVM of ZF/MRC equalised data REs
    h_hat: dict = field(default_factory=dict)        # estimator -> [rx, sym, used sc] numpy
    eq_symbols: dict = field(default_factory=dict)   # estimator -> equalised data REs (numpy, flat)
    tb_bits: int = 0


def _noise_per_re(sync: SyncResult, y: np.ndarray, frame: Frame) -> float:
    """Noise variance per resource element: from the gap (unitary FFT keeps the variance)."""
    if np.isfinite(sync.noise_var) and sync.noise_var > 0:
        return float(sync.noise_var)
    left, right = frame.guard_carriers       # fallback: the guard bins (filtered -> low estimate)
    g = np.concatenate([y[..., :left - 8], y[..., y.shape[-1] - right + 8:]], axis=-1)
    return float(np.mean(np.abs(g) ** 2))


def process_capture(r: np.ndarray, frame: Frame, rx, cfg, tx=None, timing_advance: int = 16,
                    max_frames: int | None = None, verbose: bool = True) -> list[SlotResult]:
    """Run the full chain on capture ``r`` ([rx_ant, n] or [n] complex). ``rx``/``cfg`` from build_link."""
    import torch

    r = np.atleast_2d(np.asarray(r))
    if r.shape[0] != cfg.num_rx_ant:
        raise ValueError(f"capture has {r.shape[0]} antennas, receiver expects {cfg.num_rx_ant}")
    syncs = find_frames(r[0], frame, max_frames=max_frames)
    if verbose:
        print(f"frames found: {len(syncs)} in {r.shape[1]} samples")
    bits = torch.as_tensor(frame.bits.astype(np.float32))[None, None]   # [1, UE, tb]
    x_known = None
    if tx is not None:
        x_known = frame_grid(frame, tx, cfg)
    results = []
    for k, s in enumerate(syncs):
        rc = correct_cfo(r, s.cfo_hz, frame.fs)
        y, ta = ofdm_demodulate(rc, frame, s.slot_start, timing_advance)
        yn = y[0, 0].numpy()                                    # [rx, sym, fft]
        no = _noise_per_re(s, yn, frame)
        no_t = torch.tensor(no, dtype=torch.float32)
        res = SlotResult(sync=s, noise_var=no, snr_db=float("nan"), tb_bits=int(frame.bits.size))
        for est in cfg.channel_estimators:
            if est == "perfect":
                continue
            h_hat, err_var = rx.estimate(y, no_t, est)
            hn = h_hat[0, 0, :, 0, 0].numpy()                   # [rx, sym, used sc]
            res.h_hat[est] = hn
            if np.isnan(res.snr_db):
                res.snr_db = float(10 * np.log10(np.mean(np.abs(hn) ** 2) / no))
            for rec in cfg.receivers:
                b_hat = rx.decode(y, h_hat, err_var, no_t, rec)
                err = int((b_hat != bits).sum())
                res.bit_errors[(est, rec)] = err
                res.block_errors[(est, rec)] = int(err > 0)
            if x_known is not None:
                res.evm_db[est], res.eq_symbols[est] = evm_known_data(yn, hn, x_known, frame, rx)
        if verbose:
            errs = " ".join(f"{e}/{rc_}={v}" for (e, rc_), v in res.bit_errors.items())
            evm = " ".join(f"{e}:{v:.1f}dB" for e, v in res.evm_db.items())
            print(f"  slot {k}: start={s.slot_start} cfo={s.cfo_hz:+.0f} Hz dmrs={s.dmrs_metric:.2f} "
                  f"cp={s.cp_metric:.2f} snr~{res.snr_db:.1f} dB  bit errors {errs}  evm {evm}")
        results.append(res)
    return results


def frame_grid(frame: Frame, tx, cfg) -> np.ndarray:
    """The transmitted resource grid [sym, used sc] (the receiver regenerates it from the seed)."""
    import sionna.phy
    import torch

    sionna.phy.config.seed = cfg.seed
    torch.manual_seed(cfg.seed)
    x, b = tx(1)
    if not np.array_equal(b[0, 0].numpy().astype(np.uint8), frame.bits):
        raise RuntimeError("regenerated transport block differs from the frame's bits (seed mismatch)")
    left, right = frame.guard_carriers
    return (x[0, 0, 0].numpy() * float(cfg.layer_power_scale))[:, left:x.shape[-1] - right]


def evm_known_data(yn: np.ndarray, hn: np.ndarray, x_used: np.ndarray, frame: Frame, rx):
    """EVM (dB) of MRC-equalised data REs against the known transmitted symbols."""
    left, right = frame.guard_carriers
    yu = yn[:, :, left:yn.shape[-1] - right]                   # [rx, sym, used]
    mask = rx.pilot_grid[0, 0].numpy() != 0                     # DMRS REs
    data = (~mask) & (x_used != 0)
    num = np.sum(np.conj(hn) * yu, axis=0)
    den = np.sum(np.abs(hn) ** 2, axis=0)
    x_hat = num / np.maximum(den, 1e-12)
    e = x_hat[data] - x_used[data]
    evm = np.sqrt(np.mean(np.abs(e) ** 2) / np.mean(np.abs(x_used[data]) ** 2))
    return float(20 * np.log10(evm)), x_hat[data]


def summarize(results: list[SlotResult]) -> dict:
    """Aggregate BER / BLER per (estimator, receiver) over slots."""
    out = {}
    if not results:
        return out
    keys = results[0].bit_errors.keys()
    for key in keys:
        bits = sum(r.tb_bits for r in results)
        errs = sum(r.bit_errors[key] for r in results)
        blk = sum(r.block_errors[key] for r in results)
        out[key] = {"slots": len(results), "bit_errors": errs, "ber": errs / bits,
                    "block_errors": blk, "bler": blk / len(results)}
    return out


def simulate_capture(frame: Frame, snr_db: float = 20.0, cfo_hz: float = 3000.0, delay: int = 777,
                     taps=(1.0, 0.0, 0.0, 0.35j), scale: float = 0.2, num_frames: int = 3,
                     num_rx: int = 1, seed: int = 0) -> np.ndarray:
    """A synthetic capture: frames in a row, multipath, CFO, noise at ``snr_db`` per sample."""
    rng = np.random.default_rng(seed)
    w = np.tile(frame.waveform, num_frames)
    caps = []
    for a in range(num_rx):
        h = np.asarray(taps, dtype=np.complex128) * (1.0 if a == 0 else np.exp(1j * rng.uniform(0, 2 * np.pi)))
        s = np.convolve(w, h)[:w.size]
        s = np.concatenate([np.zeros(delay + 101 * a), s])
        n = np.arange(s.size)
        s = s * np.exp(2j * np.pi * cfo_hz * n / frame.fs)
        p_sig = np.mean(np.abs(frame.slot) ** 2) * np.sum(np.abs(h) ** 2)
        sigma = np.sqrt(p_sig / 10 ** (snr_db / 10) / 2)
        s = s + sigma * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
        caps.append(scale * s)
    n = max(c.size for c in caps)
    caps = [np.concatenate([c, np.zeros(n - c.size, dtype=c.dtype)]) for c in caps]
    return np.stack(caps).astype(np.complex64)
