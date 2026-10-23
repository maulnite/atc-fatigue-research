"""Deterministic offline voice-channel stress test, not a validated ATC receiver model."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfreqz


def recording_seed(sample_id: str, seed: int) -> int:
    """Stable per-recording randomness independent of label, split and processing order."""
    token = f"radio-v1:{seed}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def channel_filter(sample_rate: int, config: dict) -> np.ndarray:
    low, high = config["band_hz"]
    if not 0 < low < high < sample_rate / 2 or not 1 <= config["filter_order"] <= 8:
        raise ValueError("Invalid voice-band filter parameters")
    return butter(config["filter_order"], [low, high], btype="bandpass",
                  fs=sample_rate, output="sos")


def frequency_response(sample_rate: int, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Forward-backward filtering squares the single-pass magnitude response."""
    frequency, response = sosfreqz(channel_filter(sample_rate, config), worN=2048, fs=sample_rate)
    magnitude = np.abs(response) ** 2
    return frequency, 20 * np.log10(np.maximum(magnitude, 1e-8))


def band_energy_fraction(signal: np.ndarray, sample_rate: int, band: list[float]) -> float:
    energy = np.abs(np.fft.rfft(signal)) ** 2
    frequency = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    total = energy.sum()
    return float(energy[(frequency >= band[0]) & (frequency <= band[1])].sum() / total) if total else 0.0


def radio_style(signal: np.ndarray, sample_rate: int, config: dict, seed: int,
                snr_db: float | None = None) -> dict:
    """Band-limit, apply fixed gain and add band-limited noise at a measured SNR.

    Reference is the transformed speech before added noise, not an estimate of
    unknown noise already in the source. The two-sided filter is offline/acausal.
    Joint peak attenuation preserves SNR; no clipping or codec is silently added.
    """
    source = np.asarray(signal, dtype=float)
    if source.ndim != 1 or len(source) < sample_rate // 2 or not np.isfinite(source).all():
        raise ValueError("Finite mono audio of at least 0.5 seconds required")
    requested_snr = float(config["snr_db"] if snr_db is None else snr_db)
    if not np.isfinite(requested_snr) or not -10 <= requested_snr <= 80:
        raise ValueError("Invalid SNR")
    sos = channel_filter(sample_rate, config)
    reference = sosfiltfilt(sos, source) * 10 ** (config["gain_db"] / 20)
    signal_power = np.mean(reference ** 2)
    if signal_power <= np.finfo(float).eps:
        raise ValueError("No usable in-band signal; do not manufacture an available recording")
    noise = sosfiltfilt(sos, np.random.default_rng(seed).standard_normal(len(source)))
    noise *= np.sqrt(signal_power / (10 ** (requested_snr / 10) * np.mean(noise ** 2)))
    peak = np.max(np.abs(reference + noise))
    attenuation = min(1.0, config["peak_limit"] / peak)
    reference, noise = reference * attenuation, noise * attenuation
    output = (reference + noise).astype(np.float32)
    # Measure using the actual float32 output sent to the extractor/player.
    residual = output.astype(float) - reference
    measured_snr = 10 * np.log10(np.mean(reference ** 2) / np.mean(residual ** 2))
    return {"signal": output, "reference": reference, "noise": residual,
            "qc": {"seed": int(seed), "requested_snr_db": requested_snr,
                   "measured_snr_db": float(measured_snr), "gain_db": config["gain_db"],
                   "peak_attenuation": float(attenuation), "sample_rate": sample_rate,
                   "duration_seconds": len(source) / sample_rate,
                   "source_rms": float(np.sqrt(np.mean(source ** 2))),
                   "radio_rms": float(np.sqrt(np.mean(output.astype(float) ** 2))),
                   "source_peak": float(np.max(np.abs(source))), "radio_peak": float(np.max(np.abs(output))),
                   "source_clipping_fraction": float(np.mean(np.abs(source) >= .999)),
                   "radio_clipping_fraction": float(np.mean(np.abs(output) >= .999)),
                   "source_inband_energy_fraction": band_energy_fraction(source, sample_rate, config["band_hz"]),
                   "radio_inband_energy_fraction": band_energy_fraction(output, sample_rate, config["band_hz"]),
                   "codec_simulated": False, "filter": "butterworth_sos_forward_backward_offline"}}
