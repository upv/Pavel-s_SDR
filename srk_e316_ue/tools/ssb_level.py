"""SSB (PSS) level of a 15.36 MSps cf32 capture, PCI 0 / N_ID2 0, CFO -37.5 kHz."""
import sys
import numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import ssb_capture as sc

for path in sys.stdin.read().split():
    x = np.fromfile(path, np.complex64)[: int(0.1 * sc.FS)].astype(complex)
    x -= x.mean()
    t = np.arange(len(x)) / sc.FS
    y = x * np.exp(-2j * np.pi * -37.5e3 * t)
    ref = sc.to_time(sc.pss(0))
    c = np.abs(sc.xcorr(y, ref)) / np.linalg.norm(ref) ** 2  # amplitude of PSS component
    m = sc.norm_corr(y, ref)
    pk = np.sort(c)[-5:].mean()
    print(f"{path.split('/')[-1]:28s} PSS {20*np.log10(pk):6.1f} dBFS  norm.peak {m.max():.2f}  "
          f"total {10*np.log10(np.mean(abs(x)**2)):6.1f} dBFS")
