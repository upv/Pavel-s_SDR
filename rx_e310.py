# -*- coding: utf-8 -*-
"""
Receiver (Orange Pi AIpro + second AntSDR E310).

Captures an IQ buffer (or reads a recorded .npy) and runs:
Schmidl-Cox -> fractional CFO -> integer CFO -> fine ZC timing ->
LS channel estimate -> phase tracking -> delay -> equalization -> EVM.

Examples:
    python rx_e310.py --uri usb: --freq 2.4e9 --gain 40 --plot
    python rx_e310.py --uri ip:192.168.2.1 --freq 2.4e9 --gain 40 --save-raw cap.npy --plot
    python rx_e310.py --file cap.npy --plot          # offline: process a recording
"""

import argparse
import numpy as np
from iio_compat import open_e310
from zc_ofdm_common import Cfg, build_frame, process_capture, summary


def _as_iq(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return np.asarray(x, dtype=np.complex64)


def _open_rx(args, cfg):
    sdr = open_e310(uri=args.uri, driver=args.driver)
    sdr.rx_enabled_channels = [0]
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    sdr.sample_rate = int(cfg.FS)
    sdr.rx_rf_bandwidth = int(cfg.FS)
    sdr.rx_lo = int(args.freq)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = args.gain
    sdr.rx_buffer_size = args.nsamp
    if int(sdr.sample_rate) != int(cfg.FS):
        print(f"warning: sample_rate={sdr.sample_rate}, expected {int(cfg.FS)}")
    _ = sdr.rx()
    return sdr


def capture(args, cfg):
    sdr = _open_rx(args, cfg)
    rx = _as_iq(sdr.rx())
    sdr.rx_destroy_buffer()
    rms = float(np.sqrt(np.mean(np.abs(rx) ** 2)))
    print(f"RX capture: {len(rx)} samples, fs={sdr.sample_rate / 1e6:.2f} MHz, "
          f"LO={sdr.rx_lo / 1e6:.1f} MHz, gain={sdr.rx_hardwaregain_chan0:.0f} dB, "
          f"rms={rms:.1f}, peak={np.max(np.abs(rx)):.1f}")
    return rx


def make_plots(rx, res, ref, prefix):
    import matplotlib.pyplot as plt
    cfg = ref.cfg

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].plot(res.sc_metric)
    ax[0, 0].axvline(res.coarse_idx, color="r", ls="--", lw=1)
    ax[0, 0].set_title("Schmidl-Cox metric")
    ax[0, 0].set_xlabel("sample")

    ax[0, 1].plot(ref.pilot_offsets * cfg.SCS / 1e6,
                  20 * np.log10(np.abs(res.H) + 1e-9))
    ax[0, 1].set_title("|H(f)|, dB (averaged over symbols)")
    ax[0, 1].set_xlabel("MHz from carrier")

    ax[1, 0].plot(ref.pilot_offsets * cfg.SCS / 1e6,
                  np.unwrap(np.angle(res.H)))
    ax[1, 0].set_title("phase H(f), rad")
    ax[1, 0].set_xlabel("MHz from carrier")

    ax[1, 1].plot(res.eq_data.real.ravel(), res.eq_data.imag.ravel(),
                  ".", ms=2, alpha=0.4)
    ax[1, 1].set_title(f"Equalized QPSK, EVM = {res.evm_db:.1f} dB")
    ax[1, 1].set_xlim(-2, 2); ax[1, 1].set_ylim(-2, 2)
    ax[1, 1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{prefix}_result.png", dpi=130)
    print(f"plot: {prefix}_result.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="usb:")
    ap.add_argument("--driver", choices=["pluto", "ad9361"], default="pluto")
    ap.add_argument("--freq", type=float, default=2.4e9)
    ap.add_argument("--gain", type=float, default=40.0, help="RX gain, dB (AGC off)")
    ap.add_argument("--nsamp", type=int, default=1 << 16, help="capture size")
    ap.add_argument("--file", default=None, help="offline: .npy IQ instead of a live capture")
    ap.add_argument("--save-raw", default=None, help="save raw capture to .npy")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="live RX scope (default on; --no-plot for one-shot text only)")
    args = ap.parse_args()

    cfg = Cfg()
    ref = build_frame(cfg)

    if args.file:
        rx = np.load(args.file)
        res = process_capture(rx, ref)
        print(summary(res, cfg))
        if args.plot:
            make_plots(rx, res, ref, prefix=args.file)
            from e310_plot import show_rx_recording
            show_rx_recording(rx, cfg.FS, title=f"E310 RX  {args.file}")
        return

    if args.plot:
        sdr = _open_rx(args, cfg)
        print(f"live RX at {args.freq / 1e6:.1f} MHz, gain {args.gain:.0f} dB. "
              "Close the plot window to stop.")
        try:
            from e310_plot import show_rx_live
            title = (f"E310 RX {args.freq / 1e6:.1f} MHz  "
                     f"gain {args.gain:.0f} dB  fs {cfg.FS / 1e6:.2f} MHz")
            show_rx_live(sdr, cfg, ref, title=title)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                sdr.rx_destroy_buffer()
            except Exception:
                pass
        return

    rx = capture(args, cfg)
    if args.save_raw:
        np.save(args.save_raw, rx)
        print(f"raw capture: {args.save_raw}")

    res = process_capture(rx, ref)
    print(summary(res, cfg))
    if np.max(res.sc_metric) < 0.5 or res.snr_db < 0:
        print("warning: no OFDM frame in the capture (looks like noise). "
              "Is tx_e310.py running on the same frequency?")


if __name__ == "__main__":
    main()
