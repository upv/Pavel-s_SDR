"""Bridge to the WirelessNN simulator (``nr_ul_sim``): config, transmitter, receiver.

WirelessNN is found through ``$WIRELESSNN_DIR`` or ``../../WirelessNN`` relative to
this file. Run everything with its virtual environment (Sionna + PyTorch):

    ~/Sources/WirelessNN/.venv/bin/python -m nr_pusch_e316.rx_pusch ...
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def wirelessnn_dir() -> Path:
    env = os.environ.get("WIRELESSNN_DIR")
    cands = [Path(env)] if env else []
    cands.append(Path(__file__).resolve().parents[2] / "WirelessNN")
    cands.append(Path.home() / "Sources" / "WirelessNN")
    for c in cands:
        if (c / "nr_ul_sim" / "__init__.py").exists():
            return c
    raise FileNotFoundError(
        "WirelessNN not found; set WIRELESSNN_DIR to the repository that contains nr_ul_sim/"
    )


def import_nr_ul_sim():
    root = str(wirelessnn_dir())
    if root not in sys.path:
        sys.path.insert(0, root)
    import nr_ul_sim  # noqa: F401

    return nr_ul_sim


@dataclass
class LinkConfig:
    """What the E316 link needs on top of the WirelessNN numerology."""

    modulation: str = "qpsk"          # qpsk | qam16 | qam64 (MCS presets of WirelessNN)
    num_prb: int = 68                 # 68 PRB = 816 subcarriers = 24.5 MHz, FFT 1024 @ 30.72 MSps
    num_rx_ant: int = 1               # E316 RX1 only, or 2 for RX1+RX2
    seed: int = 42                    # bits of the transmitted transport block
    gap_samples: int = 2466           # zeros after the slot (noise estimate, DMA alignment)
    estimators: tuple[str, ...] = ("ls_lin", "ls_soft_window")
    receivers: tuple[str, ...] = ("lmmse",)
    extra: dict = field(default_factory=dict)   # forwarded to SimConfig

    def sim_config(self):
        import_nr_ul_sim()
        from nr_ul_sim import SimConfig

        return SimConfig(
            num_ue=1, num_layers=1, num_rx_ant=self.num_rx_ant,
            modulation=self.modulation, num_prb=self.num_prb, fft_size=1024,
            subcarrier_spacing_khz=30.0, num_interferers=0,
            receivers=self.receivers, channel_estimators=self.estimators,
            seed=self.seed, **self.extra,
        )


def build_link(link: LinkConfig):
    """``(cfg, transmitter, receiver)`` of WirelessNN for this link, seeded."""
    import_nr_ul_sim()
    import numpy as np
    import sionna.phy
    import torch
    from nr_ul_sim.pusch import build_transmitter
    from nr_ul_sim.receivers import build_receivers

    cfg = link.sim_config()
    sionna.phy.config.device = "cpu"
    sionna.phy.config.seed = cfg.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tx = build_transmitter(cfg)
    rx = build_receivers(tx, cfg)
    return cfg, tx, rx
