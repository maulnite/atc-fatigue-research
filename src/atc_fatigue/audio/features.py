"""eGeMAPSv02 and MFCC summaries for independent utterances."""

from functools import lru_cache
from math import gcd

import librosa
import numpy as np
import opensmile
import soundfile as sf
from scipy.signal import resample_poly

FEATURE_VERSION = "audio-egemapsv02-mfcc13-v1"


@lru_cache(maxsize=1)
def smile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=1,
    )


def feature_names(n_mfcc=13):
    names = ["egemaps__" + name for name in smile().feature_names]
    if len(names) != 88:
        raise ValueError("Unexpected eGeMAPSv02 schema")
    return names + [f"mfcc_{i:02d}_{stat}" for stat in ("mean", "std") for i in range(n_mfcc)]


def resample(signal, source_rate, target_rate):
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    common = gcd(source_rate, target_rate)
    return resample_poly(signal, target_rate // common, source_rate // common)


def extract_signal(signal, source_rate, config):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite mono signal")
    if source_rate <= 0:
        raise ValueError("Invalid source sample rate")
    n_mfcc = config["n_mfcc"]
    names = feature_names(n_mfcc)
    duration = len(signal) / source_rate
    rms = float(np.sqrt(np.mean(signal**2))) if len(signal) else 0.0
    quality = (
        "no_signal"
        if rms < config["silence_rms"]
        else "too_short"
        if duration < config["min_duration_seconds"]
        else "ok"
    )
    result = {name: np.nan for name in names}
    result.update(
        {
            "source_sample_rate": source_rate,
            "sample_rate": config["sample_rate"],
            "duration_seconds": duration,
            "rms": rms,
            "clipping_fraction": float(np.mean(np.abs(signal) >= 0.999)) if len(signal) else 0.0,
            "audio_available": quality == "ok",
            "quality": quality,
            "speech_seconds": 0.0 if quality == "no_signal" else np.nan,
            "speech_estimation": "not_measured; utterance_duration_is_not_VAD_speech",
            "feature_version": FEATURE_VERSION,
        }
    )
    if quality != "ok":
        return result
    waveform = resample(signal, source_rate, config["sample_rate"]).astype(np.float32)
    egemaps = smile().process_signal(waveform, config["sample_rate"]).iloc[0]
    result.update(
        {"egemaps__" + k: float(v) if np.isfinite(v) else np.nan for k, v in egemaps.items()}
    )
    mfcc = librosa.feature.mfcc(y=waveform, sr=config["sample_rate"], n_mfcc=n_mfcc)
    for stat, values in (("mean", mfcc.mean(axis=1)), ("std", mfcc.std(axis=1))):
        result.update({f"mfcc_{i:02d}_{stat}": float(v) for i, v in enumerate(values)})
    return result


def extract_file(path, config):
    signal, rate = sf.read(path, always_2d=True)
    result = extract_signal(signal.mean(axis=1), rate, config)
    result["source_channels"] = signal.shape[1]
    return result
