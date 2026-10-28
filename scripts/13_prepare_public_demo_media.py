"""Prepare a few licensed examples from existing local data; never download or train."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "4"
import cv2  # noqa: E402
import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from IPython.display import Audio  # noqa: E402
from scipy.signal import welch  # noqa: E402
from atc_fatigue.audio.features import resample  # noqa: E402
from atc_fatigue.audio.radio import radio_style, recording_seed  # noqa: E402
from atc_fatigue.research.data import pseudonym  # noqa: E402
from atc_fatigue.research.plots import theme, COLORS  # noqa: E402
from atc_fatigue.research.runs import project_config, cached_run  # noqa: E402
from atc_fatigue.utils.runtime import checksum  # noqa: E402


def save_bytes(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Existing public example differs")
    else:
        path.write_bytes(data)


def main():
    config = project_config()
    root = Path("assets/demo_media")
    root.mkdir(parents=True, exist_ok=True)
    predi, _ = cached_run("predi_covid_radio", config)
    data = pd.read_parquet(predi.path / "features/features.parquet")
    source = data[data.split.eq("test")].sort_values("sample_id").iloc[0]
    original = Path(source.original_path)
    assert checksum(original) == source.source_checksum
    waveform, rate = sf.read(original, always_2d=True)
    clean = resample(waveform.mean(axis=1), rate, 16000).astype(np.float32)
    radio = radio_style(
        clean, 16000, config["radio_channel"], recording_seed(source.sample_id, config["seed"])
    )
    save_bytes(root / "predi_source.wav", original.read_bytes())
    save_bytes(root / "predi_clean.wav", Audio(clean, rate=16000, normalize=False).data)
    save_bytes(root / "predi_radio20.wav", Audio(radio["signal"], rate=16000, normalize=False).data)
    theme()
    time = np.arange(len(clean)) / 16000
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
    for axis, signal, title in zip(
        axes, [clean, radio["signal"]], ["Original / canonical mono16kHz", "Radio-style / SNR20dB"]
    ):
        axis.plot(time, signal, color=COLORS["audio"], lw=0.6)
        axis.set(ylabel="Amplitudo relatif", title=title)
    axes[-1].set_xlabel("Waktu recording (detik)")
    fig.suptitle("Predi-COVID REAL: satu recording sebelum/sesudah kanal / CC BY4.0")
    fig.savefig(root / "predi_waveforms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    spectra = [
        abs(librosa.stft(signal, n_fft=2048, hop_length=512)) for signal in [clean, radio["signal"]]
    ]
    reference = max(float(spectrum.max()) for spectrum in spectra)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    for axis, spectrum, title in zip(
        axes, spectra, ["Original / sebelum kanal", "Radio-style / SNR20dB"]
    ):
        image = axis.imshow(
            librosa.amplitude_to_db(spectrum, ref=reference),
            origin="lower",
            aspect="auto",
            extent=[0, len(clean) / 16000, 0, 8000],
            vmin=-80,
            vmax=0,
            cmap="magma",
        )
        axis.set(xlabel="Waktu recording (detik)", ylabel="Frekuensi (Hz)", title=title)
    fig.colorbar(image, ax=list(axes), label="Magnitude relatif referensi bersama (dB)")
    fig.suptitle("Predi-COVID REAL: STFT2048 / hop512 / satu reference / CC BY4.0")
    fig.savefig(root / "predi_spectrograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5.3), layout="constrained")
    for signal, label, color in [
        (clean, "Original / canonical mono16k", COLORS["secondary"]),
        (radio["signal"], "Radio-style20dB", COLORS["audio"]),
    ]:
        frequency, density = welch(signal, fs=16000, nperseg=2048)
        axis.plot(frequency, 10 * np.log10(np.maximum(density, 1e-14)), label=label, color=color)
    axis.axvspan(*config["radio_channel"]["band_hz"], color=COLORS["audio"], alpha=0.06)
    axis.set(
        xlabel="Frekuensi (Hz)",
        ylabel="PSD (dB amplitudo²/Hz)",
        xlim=(0, 8000),
        title="Predi-COVID REAL: pergeseran energi kanal / CC BY4.0",
    )
    axis.legend(loc="lower left")
    fig.savefig(root / "predi_frequency_energy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    ravdess, _ = cached_run("ravdess_emotion", config)
    pairs = pd.read_parquet(ravdess.path / "features/features.parquet")
    pair = pairs[pairs.split.eq("test")].sort_values("sample_id").iloc[0]
    assert checksum(Path(pair.original_path)) == pair.source_checksum
    save_bytes(root / "ravdess_paired_example.mp4", Path(pair.original_path).read_bytes())
    capture = cv2.VideoCapture(str(root / "ravdess_paired_example.mp4"))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, 500)
        success, frame = capture.read()
        if not success:
            raise ValueError("Cannot decode selected genuine AV frame")
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("JPEG encoding failed")
        save_bytes(root / "ravdess_frame.jpg", encoded.tobytes())
    finally:
        capture.release()
    audio_notebook = nbformat.read("notebooks/01_audio_analysis.ipynb", 4)
    audio_cell = next(
        cell for cell in audio_notebook.cells if "display(Audio(original_example" in cell.source
    )
    policy = {
        "date_utc": "2026-10-23",
        "authorization": "User explicitly revised prior no-public-media preference; research smoke-test examples",
        "scope": "one Predi-COVID test recording and one genuine paired RAVDESS test recording; no full dataset/cache/models",
        "selection": "first test sample sorted by sample_id; identical to existing notebook demonstrations",
        "sources": {
            "predi_covid": {
                "record": "https://zenodo.org/records/5937844",
                "doi": "10.5281/zenodo.5937844",
                "title": "Audio recordings of COVID-19 positive individuals from the prospective Predi-COVID cohort study with their fatigue status",
                "creators": [
                    "Abir Elbéji",
                    "Lu Zhang",
                    "Eduardo Higa",
                    "Aurélie Fischer",
                    "Vladimir Despotovic",
                    "Petr V. Nazarov",
                    "Gloria A. Aguayo",
                    "Guy Fagherazzi",
                ],
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "sample": pseudonym(source.sample_id),
                "source_sha256": source.source_checksum,
                "source_run": predi.manifest["run_id"],
                "source_rate_hz": rate,
                "source_channels": waveform.shape[1],
                "duration_seconds": len(clean) / 16000,
                "modifications": "source WAV unchanged under generic filename; clean is mono/resampled16k PCM16; radio adds configured voice-channel simulation; plots derive from same source",
                "radio_parameters": config["radio_channel"],
                "radio_qc": radio["qc"],
            },
            "ravdess": {
                "record": "https://zenodo.org/records/1188976",
                "doi": "10.5281/zenodo.1188976",
                "title": "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)",
                "creators": ["Steven R. Livingstone", "Frank A. Russo"],
                "license": "CC-BY-NC-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "sample": pseudonym(pair.sample_id),
                "source_sha256": pair.source_checksum,
                "source_run": ravdess.manifest["run_id"],
                "duration_seconds": pair.duration_seconds,
                "stream_offset_seconds": pair.stream_offset_seconds,
                "modifications": "AV MP4 unchanged under generic filename; frame sampled at500ms, JPEG encoding, no crop or image alteration",
            },
        },
        "excluded_sources": {
            "atcosim": "technical report prohibits third-party redistribution",
            "drozy": "written approval required for redistribution; image publication restricted",
            "mrl_eye": "no explicit redistribution license verified; current official page timeout",
            "cogbeacon": "software MIT does not establish a clear redistribution grant for biometric media",
        },
        "assets": [],
        "notebook_audio_cells": [
            {
                "notebook": "01_audio_analysis.ipynb",
                "cell_id": audio_cell.id,
                "code_sha256": hashlib.sha256(audio_cell.source.encode()).hexdigest(),
                "dataset": "predi_covid",
                "license": "CC-BY-4.0",
                "audio_sha256": [
                    checksum(root / "predi_clean.wav"),
                    checksum(root / "predi_radio20.wav"),
                ],
            }
        ],
    }
    for path in sorted(root.iterdir()):
        if path.suffix not in {".wav", ".mp4", ".jpg", ".png"}:
            continue
        dataset = "ravdess" if path.name.startswith("ravdess") else "predi_covid"
        policy["assets"].append(
            {
                "file": str(path),
                "sha256": checksum(path),
                "bytes": path.stat().st_size,
                "dataset": dataset,
                "license": policy["sources"][dataset]["license"],
            }
        )
    Path("reports/media_publication.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Prepared",
        len(policy["assets"]),
        "licensed examples;",
        sum(row["bytes"] for row in policy["assets"]),
        "bytes; no downloads or training",
    )


if __name__ == "__main__":
    main()
