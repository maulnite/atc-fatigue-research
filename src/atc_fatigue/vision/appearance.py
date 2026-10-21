"""Sampled face appearance features, explicitly unsuitable for blink measurement."""

import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize


def appearance_features(path, samples_per_second=2):
    if not np.isfinite(samples_per_second) or samples_per_second <= 0:
        raise ValueError("Sampling frequency must be finite and positive")
    cv2.setNumThreads(2)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Face cascade unavailable")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot decode video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or count <= 0:
        capture.release()
        raise ValueError("Video has invalid duration metadata")
    duration_ms = count / fps * 1000
    vectors, sampled, requested = [], 0, 0
    try:
        for stamp in np.arange(0, duration_ms, 1000 / samples_per_second):
            requested += 1
            capture.set(cv2.CAP_PROP_POS_MSEC, float(stamp))
            ok, image = capture.read()
            if not ok:
                continue
            sampled += 1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Bound detector work for 720p source while retaining enough face detail.
            scale = min(1.0, 480 / gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if not len(faces):
                continue
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            crop = resize(gray[y : y + h, x : x + w], (64, 64), anti_aliasing=True)
            vectors.append(
                hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            )
    finally:
        capture.release()
    n_features = len(
        hog(np.zeros((64, 64)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    )
    matrix = np.stack(vectors) if vectors else np.full((1, n_features), np.nan)
    result = {}
    for stat, values in (("mean", matrix.mean(axis=0)), ("std", matrix.std(axis=0))):
        result.update({f"visual_hog_{i:04d}_{stat}": float(v) for i, v in enumerate(values)})
    result.update(
        {
            "video_available": bool(vectors),
            "sampled_frames": sampled,
            "requested_frames": requested,
            "decode_failure_count": requested - sampled,
            "face_detection_rate": len(vectors) / sampled if sampled else 0,
            "video_quality": "ok" if vectors else "no_face" if sampled else "decode_failed",
            "video_duration_ms": duration_ms,
            "visual_feature_version": f"sampled-face-hog-{samples_per_second:g}fps-v1",
            "blink_measurement_valid": False,
        }
    )
    return result
