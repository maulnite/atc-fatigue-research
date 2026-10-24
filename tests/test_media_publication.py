"""Synthetic output fixtures: explicit media exceptions cannot waive scientific guards."""

import base64
import copy
import hashlib
import os
import runpy
from pathlib import Path

import nbformat
import pytest

from atc_fatigue.research import publication
from atc_fatigue.research.runs import notebook_source_hash


def fixture_cell():
    cell = nbformat.v4.new_code_cell("display(fixture_audio)", id="reviewed-fixture")
    payload = b"RIFF-engineering-fixture-only"
    cell.metadata["tags"] = ["private-media"]
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "text/html": '<audio controls src="data:audio/wav;base64,'
                + base64.b64encode(payload).decode()
                + '"></audio>'
            },
        )
    ]
    policy = {
        "assets": [],
        "notebook_audio_cells": [
            {
                "notebook": "fixture.ipynb",
                "cell_id": cell.id,
                "code_sha256": hashlib.sha256(cell.source.encode()).hexdigest(),
                "license": "CC-BY-4.0",
                "audio_sha256": [hashlib.sha256(payload).hexdigest()],
            }
        ],
    }
    return cell, policy


def test_only_exact_reviewed_cell_and_audio_bytes_can_be_published():
    cell, policy = fixture_cell()
    assert publication.approved_audio_cell(cell, "fixture.ipynb", policy)
    assert not publication.approved_audio_cell(cell, "other.ipynb", policy)
    tampered = copy.deepcopy(cell)
    tampered.outputs[0].data["text/html"] = (
        tampered.outputs[0].data["text/html"].replace("UklGR", "QUFBQ")
    )
    with pytest.raises(ValueError, match="Embedded audio"):
        publication.approved_audio_cell(tampered, "fixture.ipynb", policy)
    tampered = copy.deepcopy(cell)
    tampered.source += "\nchange_training()"
    with pytest.raises(ValueError, match="source changed"):
        publication.approved_audio_cell(tampered, "fixture.ipynb", policy)
    cell.outputs[0].data["image/png"] = "unreviewed-fixture"
    with pytest.raises(ValueError, match="Only the reviewed"):
        publication.approved_audio_cell(cell, "fixture.ipynb", policy)


def test_publication_profile_is_opt_in_and_keeps_private_original(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cell, policy = fixture_cell()
    notebook = nbformat.v4.new_notebook(cells=[cell])
    with monkeypatch.context() as context:
        context.setattr(os, "environ", dict(os.environ))
        context.chdir(root)
        context.setattr(publication, "publication_policy", lambda: policy)
        export = runpy.run_path(str(root / "scripts/10_execute_notebook_v3.py"))["publication_copy"]
        default = export(notebook)
        assert "<audio" not in str(default)
        approved = export(notebook, notebook_name="fixture.ipynb", publish_licensed_media=True)
        assert "<audio" in str(approved)
        assert notebook_source_hash(notebook) == notebook_source_hash(approved)
        assert len(notebook.cells[0].outputs) == 1


def test_markdown_revisions_never_allow_modified_training_code():
    frozen = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Old explanation"),
            nbformat.v4.new_code_cell("model.fit(X_train,y_train)"),
        ]
    )
    expected = notebook_source_hash(frozen)
    current = copy.deepcopy(frozen)
    current.cells[0].source = "Revised explanation and licensed media links"
    assert publication.verify_presentation_only_change(current, frozen, expected) is False
    current.cells[1].source = "model.fit(X_test,y_test)"
    with pytest.raises(ValueError, match="Scientific code"):
        publication.verify_presentation_only_change(current, frozen, expected)
