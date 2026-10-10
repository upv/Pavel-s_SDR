# -*- coding: utf-8 -*-
"""
Transmitter (MacBook + AntSDR E310).

The frame is loaded once and loops in hardware (tx_cyclic_buffer):
the host is idle after start; the receiver grabs a buffer from an arbitrary point.

Examples:
    python tx_e310.py --uri usb: --freq 2.4e9 --atten 10 --plot
    python tx_e310.py --driver ad9361 --uri usb:
    python tx_e310.py --save frame.npy            # save frame only, no hardware

Pluto firmware -> --driver pluto (default), native E310 firmware -> --driver ad9361.
"""

import argparse
import numpy as np
from iio_compat import open_e310
from zc_ofdm_common import Cfg, build_frame


def _configure_rx_monitor(sdr, cfg, freq, gain):
    sdr.rx_enabled_channels = [0]
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass
    sdr.rx_rf_bandwidth = int(cfg.FS)
    sdr.rx_lo = int(freq)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = gain
    sdr.rx_buffer_size = 8192
    try:
        _ = sdr.rx()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="usb:")
    ap.add_argument("--driver", choices=["pluto", "ad9361"], default="pluto")
    ap.add_argument("--freq", type=float, default=2.4e9, help="carrier, Hz")
    ap.add_argument("--atten", type=float, default=10.0,
                    help="TX attenuation, dB (0 = max power)")
    ap.add_argument("--save", default=None, help="save frame to .npy and exit")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="live TX/RX scope (default on; --no-plot to disable)")
    ap.add_argument("--rx-gain", type=float, default=40.0,
                    help="RX gain for the on-radio monitor (used with --plot)")
    args = ap.parse_args()

    cfg = Cfg()
    ref = build_frame(cfg)
    frame = ref.frame

    # scale: peak ~ quarter of int16 full scale (2**14) — headroom for PAPR and DAC
    scale = 2 ** 14 * 0.25 / np.max(np.abs(frame))
    iq = (frame * scale).astype(np.complex64)

    print(f"frame: {len(frame)} samples = {len(frame) / cfg.FS * 1e3:.3f} ms, "
          f"fs = {cfg.FS / 1e6:.2f} MHz, PAPR = "
          f"{10 * np.log10(np.max(np.abs(frame))**2 / np.mean(np.abs(frame)**2)):.1f} dB")

    if args.save:
        np.save(args.save, iq)
        print(f"saved to {args.save}")
        return

    sdr = open_e310(uri=args.uri, driver=args.driver)

    sdr.sample_rate = int(cfg.FS)
    sdr.tx_rf_bandwidth = int(cfg.FS)
    sdr.tx_lo = int(args.freq)
    sdr.tx_enabled_channels = [0]
    sdr.tx_hardwaregain_chan0 = -abs(args.atten)
    sdr.tx_cyclic_buffer = True
    sdr.tx(iq)

    print(f"TX started at {args.freq / 1e6:.1f} MHz, attenuation {args.atten} dB. "
          f"Ctrl+C to stop" + (" — close the plot window to stop." if args.plot else "."))

    try:
        if args.plot:
            from e310_plot import show_tx
            rx_ok = False
            try:
                _configure_rx_monitor(sdr, cfg, args.freq, args.rx_gain)
                rx_ok = True
            except Exception as e:
                print(f"RX monitor unavailable ({e}); plotting TX waveform only")
            title = (f"E310 TX {args.freq / 1e6:.1f} MHz  "
                     f"atten {args.atten:.0f} dB  fs {cfg.FS / 1e6:.2f} MHz")
            show_tx(iq, cfg.FS, ref=ref, title=title, sdr=sdr if rx_ok else None)
        else:
            import time
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        print("stopped")


if __name__ == "__main__":
    main()
