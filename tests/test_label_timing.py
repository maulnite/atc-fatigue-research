import numpy as np
import pandas as pd
import pytest

from atc_fatigue.io.validation import ensure_no_future_data, validate_causal_label_availability


def causal_input():
    return pd.DataFrame(
        {
            "subject_id": ["synthetic_01"],
            "session_id": ["test"],
            "label_subject_id": ["synthetic_01"],
            "label_session_id": ["test"],
            "feature_end_ms": [1000.0],
            "prediction_ms": [1000.0],
            "label_available_ms": [900.0],
        }
    )


def test_causal_label_availability_accepts_only_available_inputs():
    frame = causal_input()
    validate_causal_label_availability(frame)
    frame["label_available_ms"] = 1001
    with pytest.raises(ValueError, match="not available"):
        validate_causal_label_availability(frame)


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_subject_id", "other"),
        ("label_session_id", "other"),
        ("feature_end_ms", 1001),
        ("label_available_ms", np.nan),
        ("prediction_ms", np.inf),
    ],
)
def test_causal_label_guards_reject_bad_provenance_and_time(field, value):
    frame = causal_input()
    frame[field] = value
    with pytest.raises(ValueError):
        validate_causal_label_availability(frame)


def test_unknown_feature_time_cannot_pass_causal_check():
    with pytest.raises(ValueError, match="finite"):
        ensure_no_future_data([np.nan], [1000])
