"""Real paired AV to masked log-Mel/frame tensors, without fitted test statistics."""

import json
from pathlib import Path

import av
import cv2
import librosa
import numpy as np
import pandas as pd

from atc_fatigue.io.ravdess import parse_filename
from atc_fatigue.uta.assets import verify_assets
from atc_fatigue.uta.data import FaceCropper, display_quarter_turns
from .artifacts import atomic_json, digest


def verify_data(config):
    manifest = pd.read_parquet(config["manifest"]).sort_values("sample_id").reset_index(drop=True)
    if (
        len(manifest) != 128
        or manifest.subject_id.nunique() != 8
        or manifest.is_synthetic_pair.any()
    ):
        raise ValueError("Expected128genuine paired recordings /8actors")
    if manifest.source_checksum.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Duplicate source recording")
    for row in manifest.itertuples():
        meta = parse_filename(row.original_path)
        if meta["subject_id"] != row.subject_id or meta["emotion"] != row.target_value:
            raise ValueError("Native filename identity/label disagrees with manifest")
        if digest(row.original_path) != row.source_checksum:
            raise ValueError("Real source-video checksum changed")
    parent = Path(config["historical_features_run"])
    integrity = json.loads((parent / "integrity.json").read_text())
    if digest(parent / "features/features.parquet") != integrity["features/features.parquet"]:
        raise ValueError("Historical feature cache changed")
    features = pd.read_parquet(parent / "features/features.parquet")
    joined = manifest[["sample_id", "source_checksum"]].merge(
        features[["sample_id", "source_checksum", "paired_recording_verified"]],
        on="sample_id",
        validate="one_to_one",
    )
    if (
        len(joined) != 128
        or not joined.paired_recording_verified.all()
        or not joined.source_checksum_x.eq(joined.source_checksum_y).all()
    ):
        raise ValueError("Feature-to-source pairing mismatch")
    return manifest, features


def decode_audio(path, config):
    sample_rate = config["sample_rate"]
    arrays = []
    with av.open(str(path)) as container:
        streams = container.streams.audio
        if len(streams) != 1:
            raise ValueError("Expected exactly one original audio stream")
        stream = streams[0]
        stream.codec_context.thread_count = 2
        start = (
            float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        )
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate
        )
        for frame in container.decode(stream):
            arrays.extend(value.to_ndarray().reshape(-1) for value in resampler.resample(frame))
        arrays.extend(value.to_ndarray().reshape(-1) for value in resampler.resample(None))
    if not arrays:
        raise ValueError("No decoded audio observations")
    waveform = np.concatenate(arrays).astype(np.float32)
    if not np.isfinite(waveform).all() or np.sqrt(np.mean(waveform**2)) < 1e-6:
        raise ValueError("Invalid or silent source audio; no synthetic replacement")
    maximum = round(config["seconds"] * sample_rate)
    cropped = len(waveform) > maximum
    if cropped:
        offset = (len(waveform) - maximum) // 2
        waveform = waveform[offset : offset + maximum]
    if config["dc_remove"]:
        waveform = waveform - waveform.mean()
    power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=config["n_fft"],
        win_length=config["win_length"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        center=config["center"],
    )
    decibels = librosa.power_to_db(
        power, ref=config["db_reference"], top_db=config["top_db"]
    ).astype(np.float32)
    count = 1 + (maximum - config["n_fft"]) // config["hop_length"]
    if decibels.shape[1] > count:
        raise ValueError("Unexpected log-Mel frame count")
    values = np.zeros((config["n_mels"], count), np.float32)
    valid = np.zeros(count, bool)
    values[:, : decibels.shape[1]] = decibels
    valid[: decibels.shape[1]] = True
    return (
        values,
        valid,
        {
            "audio_start_seconds": start,
            "audio_sample_rate": sample_rate,
            "audio_samples": len(waveform),
            "audio_duration_seconds": len(waveform) / sample_rate,
            "audio_cropped": cropped,
            "logmel_valid_frames": int(valid.sum()),
            "logmel_total_frames": count,
            "logmel_db_min": float(decibels.min()),
            "logmel_db_max": float(decibels.max()),
        },
    )


def sample_video(path, config, cropper):
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        start = (
            float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no native timestamp")
            frames.append(frame)
            times.append(float(frame.pts * frame.time_base))
        nominal = float(stream.average_rate) if stream.average_rate else None
    ts = np.asarray(times)
    if len(ts) < config["frames"] or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("Insufficient/nonmonotonic native frame timestamps")
    targets = np.linspace(
        ts[0] + (ts[-1] - ts[0]) * config["start_fraction"],
        ts[0] + (ts[-1] - ts[0]) * config["end_fraction"],
        config["frames"],
    )
    selected = np.array([np.abs(ts - target).argmin() for target in targets])
    if len(np.unique(selected)) != config["frames"]:
        raise ValueError("Temporal sampling would duplicate a source frame")
    values = np.zeros((config["frames"], config["face_size"], config["face_size"], 3), np.uint8)
    valid = np.zeros(config["frames"], bool)
    failures = []
    for index, native in enumerate(selected):
        frame = frames[native]
        rgb = np.ascontiguousarray(
            np.rot90(frame.to_ndarray(format="rgb24"), display_quarter_turns(frame))
        )
        try:
            crop, _ = cropper.crop([rgb])
            values[index] = crop[0]
            valid[index] = True
        except (cv2.error, ValueError) as error:
            failures.append({"sampled_frame": index, "reason": str(error)})
    return (
        values,
        valid,
        ts[selected],
        {
            "video_start_seconds": start,
            "native_frames": len(frames),
            "nominal_fps": nominal,
            "sampled_native_timestamps": ts[selected].tolist(),
            "valid_frames": int(valid.sum()),
            "face_coverage": float(valid.mean()),
            "face_failures": failures,
            "visual_available": bool(valid.any()),
        },
    )


def prepare_raw(root, manifest, config, signature):
    root = Path(root)
    directory = root / "raw_tensors"
    directory.mkdir(parents=True, exist_ok=True)
    assets = verify_assets(config["asset_lock"])
    cropper = FaceCropper(assets["yunet"]["path"], config["visual"])
    audits = []
    for position, row in enumerate(manifest.itertuples()):
        target = directory / f"{row.sample_id}.npz"
        sidecar = target.with_suffix(".json")
        if sidecar.exists():
            info = json.loads(sidecar.read_text())
            if info["signature"] != signature or digest(target) != info["tensor_sha256"]:
                raise ValueError("Raw tensor resume source/config/cache mismatch")
            audits.append(info)
            continue
        mel, audio_mask, audio_info = decode_audio(row.original_path, config["audio"])
        images, frame_mask, times, video_info = sample_video(
            row.original_path, config["visual"], cropper
        )
        temporary = target.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                logmel=mel,
                audio_mask=audio_mask,
                images=images,
                frame_mask=frame_mask,
                timestamps=times,
            )
        temporary.replace(target)
        info = {
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "truth": int(row.target_value),
            "source_checksum": row.source_checksum,
            "signature": signature,
            "tensor_sha256": digest(target),
            "tensor_cache": str(target),
            "audio_available": True,
            **audio_info,
            **video_info,
        }
        atomic_json(sidecar, info)
        audits.append(info)
        if (position + 1) % 16 == 0:
            print(f"FRESH paired raw tensors: {position + 1}/128 recordings", flush=True)
    atomic_json(root / "preprocessing_audit.json", audits)
    return audits
