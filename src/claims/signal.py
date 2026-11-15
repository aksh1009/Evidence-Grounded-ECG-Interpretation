"""Step 2 - signal preprocessing.

Two views of every ECG:
* ``clean_for_measurement``: diagnostic-grade (0.05-40 Hz + 50 Hz notch), in mV. The low 0.05 Hz
  high-pass cut-off preserves the ST segment, which a 0.5 Hz filter would distort.
* ``prepare_for_cnn``: 0.5-40 Hz, downsampled to 100 Hz, per-lead standardised with *train-set*
  statistics (not per-record z-scoring, which would erase absolute voltage information).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss


def _sos_band(fs: float, lo: float, hi: float):
    return ss.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")


def clean_for_measurement(x_mv: np.ndarray, fs: float = 500, highpass: float = 0.05,
                          lowpass: float = 40.0, notch: float | None = 50.0) -> np.ndarray:
    """x_mv: (T, 12) in mV -> filtered (T, 12) float64, zero-phase."""
    x = np.asarray(x_mv, dtype=np.float64)
    x = x - np.median(x, axis=0, keepdims=True)
    x = ss.sosfiltfilt(_sos_band(fs, highpass, lowpass), x, axis=0)
    if notch and notch < fs / 2:
        b, a = ss.iirnotch(notch, Q=30, fs=fs)
        x = ss.filtfilt(b, a, x, axis=0)
    return x


def bandpass_for_cnn(x_mv: np.ndarray, fs: float = 500, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    return ss.sosfiltfilt(_sos_band(fs, lo, hi), np.asarray(x_mv, dtype=np.float64), axis=-2)


def downsample(x: np.ndarray, fs_in: int = 500, fs_out: int = 100) -> np.ndarray:
    """Anti-aliased decimation along the time axis (axis=-2). Input is already low-passed at 40 Hz."""
    if fs_in == fs_out:
        return x
    q = fs_in // fs_out
    assert fs_in == q * fs_out
    return ss.decimate(x, q, ftype="fir", zero_phase=True, axis=-2)


def prepare_batch_for_cnn(x_uv: np.ndarray, fs_in: int = 500, fs_out: int = 100) -> np.ndarray:
    """(N, T, 12) int16 microvolts -> (N, 12, T') float32 mV, band-passed and downsampled."""
    x = np.asarray(x_uv, dtype=np.float64) / 1000.0
    x = bandpass_for_cnn(x, fs_in)
    x = downsample(x, fs_in, fs_out)
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1)), dtype=np.float32)


class Standardizer:
    """Per-lead mean/std fitted on the training set only."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "Standardizer":  # x: (N, 12, T)
        self.mean = x.mean(axis=(0, 2), keepdims=True)[0]
        self.std = x.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "Standardizer":
        return cls(np.asarray(s["mean"], dtype=np.float32), np.asarray(s["std"], dtype=np.float32))
