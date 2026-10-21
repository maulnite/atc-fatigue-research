from pathlib import Path

import pytest

from atc_fatigue.cli import main

CONFIG = str(Path("configs/smoke.yaml").resolve())


def test_report_cli_does_not_require_a_model_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report.md"
    main(["--config", CONFIG, "report", "--output", str(output)])
    assert output.is_file()
    assert "belum tersedia" in output.read_text()


def test_train_cli_accepts_models_argument_and_reports_missing_input(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(
            [
                "--config",
                CONFIG,
                "train-mrl",
                "--manifest",
                str(tmp_path / "missing.parquet"),
                "--output",
                str(tmp_path / "metrics"),
                "--models",
                str(tmp_path / "models"),
            ]
        )
    assert result.value.code == 1  # Runtime missing input, not argparse's unknown-option code 2.
