"""Verified release identities, native-PTS clips and a common three-model cohort."""

import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from atc_fatigue.io.visual_candidates import UTA_LABELS
from atc_fatigue.utils.runtime import checksum, write_json


def display_quarter_turns(frame):
    """Honor source DISPLAYMATRIX, using FFmpeg's av_display_rotation_get convention.

    Only right-angle rotations are accepted. No orientation is chosen by looking
    for a favorable face-detection result.
    """
    for side in frame.side_data:
        if side.type.name != "DISPLAYMATRIX":
            continue
        matrix = np.frombuffer(side, dtype=np.int32)
        if len(matrix) != 9:
            raise ValueError("Invalid native display matrix")
        scale_x, scale_y = np.hypot(matrix[0], matrix[3]), np.hypot(matrix[1], matrix[4])
        if min(scale_x, scale_y) == 0:
            raise ValueError("Singular display matrix")
        angle = -np.degrees(np.arctan2(matrix[1] / scale_y, matrix[0] / scale_x))
        quarter = round(float(angle) / 90)
        if abs(angle - quarter * 90) > 0.01:
            raise ValueError("Unsupported non-right-angle source rotation; inspect explicitly")
        return quarter
    return 0


def source_manifest(index, root):
    """Use verified official ZIP mapping, never guess identity from an arbitrary path."""
    rows = []
    for entry in index["videos"]:
        row = dict(
            entry,
            file_location=str(Path(root) / entry["relative_path"]),
            label_name=UTA_LABELS[entry["label"]],
            decoding_status="not_checked",
            missing=True,
            source_sha256=None,
        )
        path = Path(row["file_location"])
        if path.exists():
            row.update(missing=False, source_sha256=checksum(path), file_bytes=path.stat().st_size)
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    duration = (
                        float(stream.duration * stream.time_base)
                        if stream.duration is not None
                        else float(container.duration / av.time_base)
                        if container.duration
                        else None
                    )
                    if duration is None or duration <= 0:
                        raise ValueError(
                            "No valid duration available for bounded temporal sampling"
                        )
                    first = next(container.decode(stream), None)
                    if first is None or first.pts is None:
                        raise ValueError("No decodable timestamped first frame")
                    row.update(
                        duration_seconds=duration,
                        nominal_fps=float(stream.average_rate) if stream.average_rate else None,
                        width=stream.width,
                        height=stream.height,
                        codec=stream.codec_context.name,
                        first_pts_seconds=float(first.pts * first.time_base),
                        rotation_quarter_turns=display_quarter_turns(first),
                        decoding_status="first_frame_valid; sampled_clip_decode_pending",
                    )
            except (av.error.FFmpegError, OSError, ValueError, IndexError) as exc:
                row.update(decoding_status="invalid", failure_reason=type(exc).__name__)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.original_video_id.duplicated().any():
        raise ValueError("Duplicate original file identity")
    if frame.dropna(subset=["source_sha256"]).source_sha256.duplicated().any():
        raise ValueError("Duplicate source video bytes; reconcile before splitting")
    if frame.groupby("subject_id").official_fold.nunique().max() != 1:
        raise ValueError("Participant leakage in official source release")
    return frame


def plan_clips(parts, config):
    """A fixed budget per recording, not per physical split file; no boundary crossing."""
    parts = parts.sort_values("part_index")
    durations = parts.duration_seconds.to_numpy(float)
    boundaries = np.concatenate([[0.0], np.cumsum(durations)])
    length = config["clip_seconds"]
    total = boundaries[-1]
    if total < length or not np.isfinite(total):
        raise ValueError("Recording shorter than the fixed clip duration")
    centers = np.linspace(length / 2, total - length / 2, config["clips_per_recording"])
    plans = []
    for index, center in enumerate(centers):
        part_index = min(int(np.searchsorted(boundaries[1:], center, side="right")), len(parts) - 1)
        part = parts.iloc[part_index]
        if part.duration_seconds < length:
            raise ValueError("Source fragment shorter than one clip")
        relative = np.clip(
            center - boundaries[part_index] - length / 2, 0, part.duration_seconds - length
        )
        start = float(part.first_pts_seconds + relative)
        plans.append(
            {
                "clip_id": index,
                "original_video_id": part.original_video_id,
                "recording_group_id": part.recording_group_id,
                "source_file": part.file_location,
                "rotation_quarter_turns": int(part.rotation_quarter_turns)
                if "rotation_quarter_turns" in part
                else 0,
                "start_seconds": start,
                "end_seconds": start + length,
                "region": min(2, index * 3 // len(centers)),
                "timeline": "native source-file PTS; no assumed continuous clock between split files",
            }
        )
    return plans


def select_temporal_indices(timestamps, count, max_gap):
    ts = np.asarray(timestamps, float)
    if len(ts) < count or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("Insufficient/nonmonotonic native timestamps")
    if (np.diff(ts) > max_gap).any():
        raise ValueError("Native gap exceeds the permitted clip gap; no interpolation")
    # Same endpoint-inclusive uniform subsampling as the official X3D hub transform.
    indices = np.linspace(0, len(ts) - 1, count).astype(int)
    if len(set(indices)) != count:
        raise ValueError("Repeated frame is not a new observation")
    return indices


def decode_clip(plan, config):
    frames, timestamps = [], []
    with av.open(plan["source_file"]) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        container.seek(int(plan["start_seconds"] / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Decoded frame lacks PTS")
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < plan["start_seconds"]:
                continue
            if timestamp >= plan["end_seconds"]:
                break
            frames.append(frame)
            timestamps.append(timestamp)
    selected = select_temporal_indices(
        timestamps, config["frames_per_clip"], config["max_native_gap_seconds"]
    )
    images = []
    for index in selected:
        frame = frames[index]
        image = frame.to_ndarray(format="rgb24")
        quarter = plan.get("rotation_quarter_turns", display_quarter_turns(frame))
        images.append(np.ascontiguousarray(np.rot90(image, k=quarter)))
    return images, np.asarray(timestamps)[selected], len(frames)


class FaceCropper:
    def __init__(self, path, config):
        cv2.setNumThreads(2)
        self.config = config
        self.detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            score_threshold=config["face_detection_score"],
            nms_threshold=0.3,
            top_k=20,
        )

    def crop(self, frames):
        boxes = []
        dimensions = {(image.shape[0], image.shape[1]) for image in frames}
        if len(dimensions) != 1:
            raise ValueError("Frame dimensions change inside a temporal clip")
        height, width = frames[0].shape[:2]
        for image in frames:
            scale = min(1.0, 640 / max(height, width))
            small = cv2.resize(image, (round(width * scale), round(height * scale)))
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            if faces is None or len(faces) != 1:
                raise ValueError("Missing or ambiguous face; no whole-frame fallback")
            boxes.append(faces[0, :4] / scale)
        x, y, box_width, box_height = np.median(boxes, axis=0)
        side = max(box_width, box_height) * self.config["face_margin"]
        center_x, center_y = x + box_width / 2, y + box_height / 2
        left, top = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
        right, bottom = min(width, int(center_x + side / 2)), min(height, int(center_y + side / 2))
        if min(right - left, bottom - top) < 32:
            raise ValueError("Face crop too small for reliable visual input")
        crops = np.stack(
            [
                cv2.resize(image[top:bottom, left:right], (self.config["face_size"],) * 2)
                for image in frames
            ]
        )
        return crops, [left, top, right, bottom]


def prepare_recordings(manifest, output, sampling, detector_path):
    output = Path(output)
    (output / "clips").mkdir(parents=True, exist_ok=True)
    cropper = FaceCropper(detector_path, sampling)
    records, clip_audit = [], []
    for group_id, parts in manifest.groupby("recording_group_id", sort=True):
        sidecar = output / "clips" / f"{group_id}.json"
        signature = hashlib.sha256(
            json.dumps(
                {
                    "sampling": sampling,
                    "source": parts[["original_video_id", "source_sha256"]].to_dict("records"),
                    "detector_sha256": checksum(detector_path),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if saved["signature"] != signature:
                raise ValueError("Preparation resume attempted with changed source/config/detector")
            previous = saved["record"]
            if (
                previous.get("clip_cache")
                and checksum(previous["clip_cache"]) != previous["cache_sha256"]
            ):
                raise ValueError("Prepared clip cache changed")
            records.append(previous)
            clip_audit.extend(saved["clips"])
            print(f"CACHED preparation {group_id}", flush=True)
            continue
        row = {
            "recording_group_id": group_id,
            "subject_id": str(parts.subject_id.iloc[0]),
            "official_fold": int(parts.official_fold.iloc[0]),
            "label": int(parts.label.iloc[0]),
            "source_files": len(parts),
            "eligible": False,
        }
        if (
            parts.subject_id.nunique() != 1
            or parts.label.nunique() != 1
            or parts.official_fold.nunique() != 1
        ):
            raise ValueError("Mixed participant/label/fold inside one recording")
        if parts.missing.any() or parts.decoding_status.eq("invalid").any():
            row.update(exclusion_reason="missing_or_invalid_source_video")
            records.append(row)
            write_json(sidecar, {"signature": signature, "record": row, "clips": []})
            continue
        plans = plan_clips(parts, sampling)
        arrays, valid, times = [], [], []
        group_clips = []
        for plan in plans:
            info = {k: v for k, v in plan.items() if k != "source_file"}
            try:
                frames, pts, decoded = decode_clip(plan, sampling)
                crops, box = cropper.crop(frames)
                info.update(
                    valid=True, decoded_native_frames=decoded, selected_pts=pts.tolist(), box=box
                )
            except (av.error.FFmpegError, cv2.error, ValueError, OSError) as exc:
                crops = np.zeros(
                    (sampling["frames_per_clip"], sampling["face_size"], sampling["face_size"], 3),
                    np.uint8,
                )
                pts = np.full(sampling["frames_per_clip"], np.nan)
                info.update(valid=False, reason=str(exc))
            arrays.append(crops)
            valid.append(info["valid"])
            times.append(pts)
            clip_audit.append(info)
            group_clips.append(info)
        mask = np.asarray(valid, bool)
        regions = {p["region"] for p, usable in zip(plans, valid) if usable}
        eligible = int(mask.sum()) >= sampling["minimum_usable_clips"]
        if sampling["require_early_middle_late"]:
            eligible = eligible and regions == {0, 1, 2}
        target = output / "clips" / f"{group_id}.npz"
        if target.exists():
            raise FileExistsError(
                "Prepared clip cache exists; never overwrite completed extraction"
            )
        temporary = target.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, frames=np.stack(arrays), valid=mask, timestamps=np.stack(times)
            )
        temporary.replace(target)
        row.update(
            eligible=bool(eligible),
            usable_clips=int(mask.sum()),
            planned_clips=len(mask),
            valid_frame_coverage=float(mask.mean()),
            clip_cache=str(target),
            cache_sha256=checksum(target),
            exclusion_reason=""
            if eligible
            else "insufficient_clips_or_missing_early_middle_late_region",
        )
        records.append(row)
        write_json(sidecar, {"signature": signature, "record": row, "clips": group_clips})
        print(f"Prepared {group_id}: {mask.sum()}/{len(mask)} usable clips", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "recording_manifest.csv", index=False)
    write_json(output / "clip_audit.json", clip_audit)
    return frame


def fold_membership(manifest, outer_fold):
    eligible = manifest[manifest.eligible].copy()
    validation_fold = outer_fold % 5 + 1
    eligible["split"] = np.where(
        eligible.official_fold.eq(outer_fold),
        "test",
        np.where(eligible.official_fold.eq(validation_fold), "validation", "train"),
    )
    if eligible.groupby("subject_id").split.nunique().max() != 1:
        raise ValueError("Participant leakage")
    if eligible.groupby("recording_group_id").split.nunique().max() != 1:
        raise ValueError("Original-video/fragment leakage")
    if set(eligible.split) != {"train", "validation", "test"}:
        raise ValueError("Incomplete fold population; cannot claim official five-fold evaluation")
    if any(set(group.label) != {0, 5, 10} for _, group in eligible.groupby("split")):
        raise ValueError("A class is absent from an evaluation partition")
    return eligible
