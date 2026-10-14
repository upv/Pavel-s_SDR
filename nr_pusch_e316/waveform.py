"""PUSCH slot -> time-domain frame, and the DMRS reference the receiver correlates with.

Frame layout (samples at 30.72 MSps):

    [ slot: 14 x (CP + 1024) ][ gap: zeros ]

Sionna's PUSCH transmitter uses one CP length for every symbol (the long NR CP of
slot 0, 89 samples on the 1024 grid), so the slot is 15582 samples; the gap pads
the frame to a multiple of 16 (DMA alignment of the cyclic TX buffer) and gives
the receiver a signal-free stretch for the noise estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wirelessnn import LinkConfig, build_link


@dataclass
class Frame:
    link: LinkConfig
    fs: float                 # sample rate [Hz]
    fft_size: int
    cp: int                   # cyclic prefix [samples]
    num_symbols: int
    dmrs_symbols: list        # OFDM symbol indices carrying DMRS
    slot: np.ndarray          # [num_symbols*(cp+fft)] complex64, unit mean power on the band
    gap: int
    bits: np.ndarray          # [tb_size] transmitted bits (the receiver's ground truth)
    dmrs_ref: np.ndarray      # [fft_size] time-domain useful part of the first DMRS symbol
    guard_carriers: tuple

    @property
    def sym_len(self) -> int:
        return self.cp + self.fft_size

    @property
    def slot_len(self) -> int:
        return self.num_symbols * self.sym_len

    @property
    def frame_len(self) -> int:
        return self.slot_len + self.gap

    @property
    def waveform(self) -> np.ndarray:
        return np.concatenate([self.slot, np.zeros(self.gap, dtype=np.complex64)])

    def dmrs_offsets(self) -> list:
        """Sample offsets (from the slot start) of the useful part of every DMRS symbol."""
        return [s * self.sym_len + self.cp for s in self.dmrs_symbols]

    def save(self, path: str | Path) -> None:
        np.savez(path, slot=self.slot, bits=self.bits, dmrs_ref=self.dmrs_ref,
                 fs=self.fs, fft_size=self.fft_size, cp=self.cp, num_symbols=self.num_symbols,
                 dmrs_symbols=np.array(self.dmrs_symbols), gap=self.gap,
                 guard_carriers=np.array(self.guard_carriers),
                 modulation=self.link.modulation, num_prb=self.link.num_prb, seed=self.link.seed)


def ofdm_modulate(grid, cp: int):
    """Sionna's OFDM modulator on a [..., num_symbols, fft] grid -> [..., samples] (numpy)."""
    import torch
    from sionna.phy.ofdm import OFDMModulator

    x = torch.as_tensor(np.asarray(grid, dtype=np.complex64))
    return OFDMModulator(cp)(x).numpy()


def make_frame(link: LinkConfig, tx=None, rx=None, cfg=None) -> Frame:
    """Generate one PUSCH slot of ``link`` with the WirelessNN transmitter."""
    if tx is None:
        cfg, tx, rx = build_link(link)
    rg = tx.resource_grid
    cp = int(rg.cyclic_prefix_length)
    x, b = tx(1)                                    # x [1, UE, ant, sym, fft], b [1, UE, tb]
    x = x[0, 0, 0].numpy() * float(cfg.layer_power_scale)
    slot = ofdm_modulate(x, cp).astype(np.complex64)

    # DMRS-only grid: the reference for timing / CFO (data REs set to zero)
    left, right = cfg.guard_carriers
    pg = rx.pilot_grid[0, 0].numpy()                # [sym, used sc]
    ref_grid = np.zeros_like(x)
    ref_grid[:, left:rg.fft_size - right] = pg
    dmrs_syms = list(rx.dmrs_syms)
    ref_time = ofdm_modulate(ref_grid, cp)
    s0 = dmrs_syms[0]
    dmrs_ref = ref_time[s0 * (cp + rg.fft_size) + cp: (s0 + 1) * (cp + rg.fft_size)].astype(np.complex64)

    gap = int(link.gap_samples)
    total = slot.size + gap
    if total % 16:
        gap += 16 - total % 16
    return Frame(link=link, fs=float(rg.bandwidth), fft_size=int(rg.fft_size), cp=cp,
                 num_symbols=int(rg.num_ofdm_symbols), dmrs_symbols=dmrs_syms, slot=slot,
                 gap=gap, bits=b[0, 0].numpy().astype(np.uint8), dmrs_ref=dmrs_ref,
                 guard_carriers=(int(left), int(right)))


def to_int16_iq(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Complex -> interleaved int16 for the AD9361 DAC, peak at ``peak`` of full scale."""
    scale = peak * 32767.0 / max(float(np.abs(x).max()), 1e-12)
    out = np.empty(2 * x.size, dtype=np.int16)
    out[0::2] = np.round(x.real * scale)
    out[1::2] = np.round(x.imag * scale)
    return out


def from_int16_iq(raw: np.ndarray, num_channels: int = 1) -> np.ndarray:
    """Interleaved int16 (I0 Q0 [I1 Q1] ...) -> [num_channels, n] complex64 in units of ADC full scale."""
    raw = np.asarray(raw, dtype=np.int16).astype(np.float32) / 2048.0
    iq = raw[0::2] + 1j * raw[1::2]
    return iq.reshape(-1, num_channels).T.astype(np.complex64)
