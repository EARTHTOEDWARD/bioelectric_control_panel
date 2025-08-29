import numpy as np

from bcp.core.chaos import compute_window_metrics
from bcp.core.stream import streaming_windows


def _gen_sine(fs=1000.0, f=3.5, dur=6.0):
    t = np.arange(0, dur, 1.0 / fs)
    x = np.sin(2 * np.pi * f * t)
    return x


def _gen_colored_noise(fs=500.0, dur=6.0):
    n = int(fs * dur)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    phases = np.exp(1j * 2 * np.pi * np.random.rand(freqs.size))
    amp = np.ones_like(freqs)
    amp[1:] = 1.0 / np.power(freqs[1:], 0.5)
    x = np.fft.irfft(phases * amp, n=n)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x


def test_compute_window_metrics_basic():
    fs = 1000.0
    x = _gen_sine(fs=fs)
    m = compute_window_metrics(x, fs=fs, modality="cardiac")
    assert set(m.keys()) >= {
        "lambda1",
        "D2",
        "mse_auc_1_20",
        "rqa_det",
        "rqa_Lmax",
        "rqa_trap",
        "quality",
        "m",
        "tau_samples",
    }


def test_metrics_separate_sine_vs_noise():
    fs = 500.0
    sine = _gen_sine(fs=fs)
    noise = _gen_colored_noise(fs=fs)
    ms = compute_window_metrics(sine, fs=fs, modality="cardiac")
    mn = compute_window_metrics(noise, fs=fs, modality="generic")
    # Sine should have low lambda1 and high determinism
    assert ms["lambda1"] < 0.2 or np.isnan(ms["lambda1"])  # near zero or small
    assert ms["rqa_det"] >= 0.6
    # Noise higher lambda1 and lower determinism
    assert mn["lambda1"] > 0.2
    assert mn["rqa_det"] < 0.6


def test_streaming_windows_iterates():
    fs = 200.0
    x = _gen_sine(fs=fs, dur=4.0)
    it = list(streaming_windows(x, fs=fs, window_s=2.0, step_s=1.0, modality="cardiac"))
    # Expect at least a couple of windows
    assert len(it) >= 2
    t0, m = it[0]
    assert isinstance(t0, float)
    assert "lambda1" in m

