"""Synthetic end-to-end check: frame -> multipath + CFO + noise -> receiver -> no bit errors."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nr_pusch_e316.receiver import process_capture, simulate_capture  # noqa: E402
from nr_pusch_e316.waveform import from_int16_iq, make_frame, to_int16_iq  # noqa: E402
from nr_pusch_e316.wirelessnn import LinkConfig, build_link  # noqa: E402


@pytest.fixture(scope="module")
def link_qpsk():
    link = LinkConfig(modulation="qpsk", num_prb=68, seed=7)
    cfg, tx, rx = build_link(link)
    return link, cfg, tx, rx, make_frame(link, tx, rx, cfg)


def test_frame_layout(link_qpsk):
    link, cfg, tx, rx, frame = link_qpsk
    assert frame.fs == 30.72e6 and frame.fft_size == 1024 and frame.num_symbols == 14
    assert frame.frame_len % 16 == 0
    assert frame.dmrs_symbols == [2, 11]
    assert abs(np.mean(np.abs(frame.slot) ** 2) - 1.0 * 816 / 1024) < 0.1   # unit power per used RE
    rt = from_int16_iq(to_int16_iq(frame.waveform), 1)[0]
    assert np.corrcoef(np.abs(rt), np.abs(frame.waveform))[0, 1] > 0.999


@pytest.mark.parametrize("cfo_hz,snr_db", [(0.0, 25.0), (3000.0, 20.0), (-41000.0, 15.0)])
def test_chain_decodes(link_qpsk, cfo_hz, snr_db):
    link, cfg, tx, rx, frame = link_qpsk
    cap = simulate_capture(frame, snr_db=snr_db, cfo_hz=cfo_hz, delay=1234, num_frames=3)
    res = process_capture(cap, frame, rx, cfg, tx=tx, verbose=False)
    assert len(res) == 3, "all three tiled frames (with their gaps) are complete"
    for r in res:
        assert abs(r.sync.cfo_hz - cfo_hz) < 300, (r.sync.cfo_hz, cfo_hz)
        assert r.sync.dmrs_metric > 0.5 and r.sync.cp_metric > 0.5
        assert abs(r.snr_db - snr_db) < 3.0, (r.snr_db, snr_db)
        for key, errs in r.bit_errors.items():
            assert errs == 0, (key, errs)
        for est, evm in r.evm_db.items():
            assert evm < -(snr_db - 4), (est, evm)


def test_timing_exact(link_qpsk):
    link, cfg, tx, rx, frame = link_qpsk
    delay = 2222
    cap = simulate_capture(frame, snr_db=30.0, cfo_hz=0.0, delay=delay, taps=(1.0,), num_frames=2)
    res = process_capture(cap, frame, rx, cfg, tx=tx, verbose=False)
    assert res and res[0].sync.slot_start == delay


def test_two_antennas(link_qpsk):
    link = LinkConfig(modulation="qam16", num_prb=68, seed=3, num_rx_ant=2, estimators=("ls_lin",))
    cfg, tx, rx = build_link(link)
    frame = make_frame(link, tx, rx, cfg)
    cap = simulate_capture(frame, snr_db=18.0, cfo_hz=1500.0, num_rx=2, num_frames=2)
    res = process_capture(cap, frame, rx, cfg, tx=tx, verbose=False)
    assert res and all(v == 0 for r in res for v in r.bit_errors.values())
