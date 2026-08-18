import numpy as np

ZC_LEN = 127

# ---------------------------------------------------------------------------
# Zadoff-Chu local replicas
# ---------------------------------------------------------------------------
def zadoff_chu(root: int, length: int = ZC_LEN) -> np.ndarray:
    """seq[m] = exp(-j * pi * R * m * (m+1) / N), m = 0..N-1."""
    m = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * m * (m + 1.0) / length)

sig_1 = zadoff_chu(21)
sig_1 = sig_1 / np.linalg.norm(sig_1)

sig_2 = zadoff_chu(25)
sig_2 = sig_2 / np.linalg.norm(sig_2)

corr = np.abs(np.sum(sig_1.conj() * sig_2))

pass









