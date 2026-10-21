import cv2
import numpy as np
import pytest

from atc_fatigue.vision.appearance import appearance_features


def test_corrupt_video_is_rejected(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"synthetic-corrupt-video-fixture")
    with pytest.raises(ValueError, match="Cannot decode"):
        appearance_features(path)


def test_real_blank_video_has_no_invented_face_features(tmp_path):
    path = tmp_path / "synthetic_blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened(), "Test environment must support the MJPG fixture codec"
    try:
        for _ in range(10):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    result = appearance_features(path)
    assert not result["video_available"]
    assert result["video_quality"] == "no_face"
    assert result["sampled_frames"] > 0
    assert result["face_detection_rate"] == 0
    assert all(np.isnan(v) for k, v in result.items() if k.startswith("visual_hog_"))


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_sampling_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="frequency"):
        appearance_features("not_read_when_rate_invalid", samples_per_second=rate)
