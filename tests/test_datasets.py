import io
import zipfile

import numpy as np
import pytest

from atc_fatigue.io.download import extract_members
from atc_fatigue.io.drozy import read_landmarks
from atc_fatigue.io.mrl import parse_filename
from atc_fatigue.io.ravdess import parse_filename as parse_ravdess


def test_mrl_label_and_subject_mapping():
    row = parse_filename("s0001_00001_0_1_1_0_1_01.png")
    assert row["subject_id"] == "s0001" and row["target_value"] == 1
    with pytest.raises(ValueError):
        parse_filename("invented_01.png")


def test_ravdess_requires_actual_av():
    row = parse_ravdess("01-01-03-01-01-01-12.mp4")
    assert row["emotion"] == 3 and row["subject_id"] == "12"
    with pytest.raises(ValueError):
        parse_ravdess("02-01-03-01-01-01-12.mp4")


def test_extraction_budget_and_existing_integrity(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("x/test.txt", "source")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as z:
        with pytest.raises(ValueError):
            extract_members(z, ["x/test.txt"], tmp_path, 2)
        paths = extract_members(z, ["x/test.txt"], tmp_path, 10)
        assert paths[0].read_text() == "source"
        paths[0].write_text("truncated")
        with pytest.raises(IOError):
            extract_members(z, ["x/test.txt"], tmp_path, 10)


def test_ambiguous_drozy_alignment_is_rejected(tmp_path):
    landmarks, timestamps = tmp_path / "points.txt", tmp_path / "timestamps.txt"
    np.savetxt(landmarks, np.zeros((3, 136)))
    np.savetxt(timestamps, np.zeros((2, 8)))
    with pytest.raises(ValueError, match="Frame counts differ"):
        read_landmarks(landmarks, timestamps)
