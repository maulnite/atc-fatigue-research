import numpy as np
import pytest

from atc_fatigue.audio.features import extract_signal, feature_names, resample
from atc_fatigue.utils.runtime import load_config


def test_resampling_duration():
    signal = np.ones(44100)
    out = resample(signal, 44100, 16000)
    assert abs(len(out) / 16000 - 1) < 1 / 16000


def test_empty_and_silent_audio_preserve_missing_masks():
    for signal in (np.array([]), np.zeros(16000)):
        result = extract_signal(signal, 16000, load_config()["audio"])
        assert not result["audio_available"]
        assert result["speech_seconds"] == 0
        assert all(np.isnan(result[name]) for name in feature_names())


def test_short_signal_is_not_fatal():
    result = extract_signal(np.ones(100) * 0.1, 16000, load_config()["audio"])
    assert result["quality"] == "too_short"
    assert not result["audio_available"]


def test_egemaps_schema_and_real_extraction():
    rate = 16000
    signal = 0.1 * np.sin(2 * np.pi * 180 * np.arange(rate) / rate)
    result = extract_signal(signal, rate, load_config()["audio"])
    keys = [k for k in result if k.startswith("egemaps__")]
    assert len(keys) == 88
    assert result["audio_available"]
    assert all(isinstance(result[k], float) for k in keys)


def test_bad_audio_is_rejected():
    with pytest.raises(ValueError):
        extract_signal(np.array([np.nan]), 16000, load_config()["audio"])
