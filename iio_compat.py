# -*- coding: utf-8 -*-
"""Import pyadi-iio on macOS and open the AntSDR E310.

pylibiio 1.0 splits the library filename on ``.so.``. On macOS
``find_library('iio')`` returns the framework path
``.../iio.framework/Versions/1.0/iio``, which crashes at import.
This module applies the upstream-style loader fix, then scans USB
for the E310 (Pluto-compatible firmware).
"""

from __future__ import annotations

import os.path as _ospath

_basename = _ospath.basename


def _libiio_basename(path):
    name = _basename(path)
    if ".so." not in name and (
        name == "iio" or "libiio" in name or name.endswith(".dylib")
    ):
        return "libiio.so.1"
    return name


if getattr(_ospath.basename, "__name__", "") != "_libiio_basename":
    _ospath.basename = _libiio_basename

import adi  # noqa: E402
import iio  # noqa: E402

_PLUTO_HINTS = ("Pluto", "ANTSDR", "0456:b673", "ADALM-PLUTO", "AD9361")


def find_e310_uri(preferred: str | None = None) -> str:
    """Return a usable IIO URI. Prefer USB; IP 192.168.1.10 is often down."""
    if preferred and preferred not in ("auto", ""):
        return preferred
    try:
        contexts = iio.scan_contexts()
    except Exception:
        contexts = {}
    for uri, desc in contexts.items():
        blob = f"{uri} {desc}"
        if any(h in blob for h in _PLUTO_HINTS):
            return uri
    if contexts:
        return next(iter(contexts))
    return "usb:"


def open_e310(uri: str | None = None, driver: str = "pluto"):
    """Open the E310. Tries the given URI, then USB scan, then ``usb:``."""
    candidates = []
    if uri and uri not in ("auto", ""):
        candidates.append(uri)
    scanned = find_e310_uri()
    if scanned not in candidates:
        candidates.append(scanned)
    if "usb:" not in candidates:
        candidates.append("usb:")

    last_err = None
    cls = adi.Pluto if driver == "pluto" else adi.ad9361
    for candidate in candidates:
        try:
            sdr = cls(uri=candidate)
            if candidate != uri:
                print(f"opened {candidate}")
            return sdr
        except Exception as e:
            last_err = e
    raise last_err
