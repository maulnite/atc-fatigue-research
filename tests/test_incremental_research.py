"""Engineering fixtures for privacy, time-label boundaries and actor independence."""
import os
import runpy
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from atc_fatigue.audio.features import extract_signal
from atc_fatigue.io.validation import validate_target
from atc_fatigue.research.data import (
    actor_metric_table, cumulative_event_increments, parse_cogbeacon_session,
    parse_predi_filename, private_media_enabled, temporal_audio_representations,
)
from atc_fatigue.research.runs import notebook_source_hash


def test_private_export_removes_all_media_but_preserves_local_evidence(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as context:
        context.setattr(os, 'environ', dict(os.environ))
        context.chdir(root)
        export = runpy.run_path(str(root / 'scripts/execute_notebooks.py'))['publication_copy']
        sensitive = nbformat.v4.new_code_cell('plot_actual_audio()',metadata={'tags':['private-media']})
        sensitive.outputs = [nbformat.v4.new_output('display_data',data={'image/png':'PRIVATE',
                            'text/html':'<audio src="private">'})]
        safe = nbformat.v4.new_code_cell('plot_synthetic_fixture()')
        safe.outputs = [nbformat.v4.new_output('display_data',data={'image/png':'SAFE'})]
        original = nbformat.v4.new_notebook(cells=[sensitive,safe])
        public = export(original)
        assert 'PRIVATE' not in str(public) and '<audio' not in str(public)
        assert public.cells[1].outputs[0].data['image/png']=='SAFE'
        assert original.cells[0].outputs[0].data['image/png']=='PRIVATE'
        assert notebook_source_hash(original)==notebook_source_hash(public)


def test_real_audio_visualization_matches_canonical_extractor():
    # Synthetic fixture validates code consistency only, never reported as fatigue data.
    rate=22050
    time=np.arange(rate)/rate
    mono=.2*np.sin(2*np.pi*170*time)+.03*np.sin(2*np.pi*1100*time)
    stereo=np.column_stack([mono,mono*.8])
    config={'sample_rate':16000,'n_mfcc':13,'min_duration_seconds':.5,'silence_rms':1e-5}
    representations=temporal_audio_representations(stereo,rate,config)
    actual=extract_signal(stereo.mean(axis=1),rate,config)
    for stat,values in [('mean',representations['mfcc'].mean(axis=1)),('std',representations['mfcc'].std(axis=1))]:
        np.testing.assert_allclose(values,[actual[f'mfcc_{i:02d}_{stat}'] for i in range(13)],rtol=1e-6)
    assert len(representations['prepared_signal'])==16000


def test_private_media_requires_explicit_valid_opt_in(monkeypatch):
    monkeypatch.delenv('ATC_NOTEBOOK_PRIVATE_MEDIA',raising=False)
    assert not private_media_enabled()
    monkeypatch.setenv('ATC_NOTEBOOK_PRIVATE_MEDIA','1')
    assert private_media_enabled()
    monkeypatch.setenv('ATC_NOTEBOOK_PRIVATE_MEDIA','yes')
    with pytest.raises(ValueError):
        private_media_enabled()


def test_predi_publisher_identity_and_daily_label_are_parsed_without_relabeling():
    result=parse_predi_filename('Predi-COVID_0067_20200610140840_1_m4a_M_0.wav')
    assert result['subject_id']=='0067' and result['session_id']=='0067_20200610'
    assert result['target_value']==0 and result['audio_task']==1
    with pytest.raises(ValueError):
        parse_predi_filename('Predi-COVID_0067_20209910140840_1_m4a_M_0.wav')
    validate_target('predi_covid','clinical_fatigue_binary')
    with pytest.raises(ValueError):
        validate_target('predi_covid','ATC_fatigue_binary')


def test_cogbeacon_second_day_is_same_subject_not_another_person():
    first=parse_cogbeacon_session('user_1_v_m')
    second=parse_cogbeacon_session('user_1b_v_m')
    assert first['subject_id']==second['subject_id']
    assert first['session_id']!=second['session_id']
    np.testing.assert_array_equal(cumulative_event_increments([0,0,1,1,3]),[0,0,1,0,2])
    validate_target('cogbeacon',None)
    with pytest.raises(ValueError):
        validate_target('cogbeacon','fatigue_binary')


@pytest.mark.parametrize('values',[[0,1,0],[-1,0],[0,.5],[],[np.nan]])
def test_cogbeacon_rejects_invalid_cumulative_annotation(values):
    with pytest.raises(ValueError):
        cumulative_event_increments(values)


def test_loao_and_variation_preserve_actor_as_independent_unit():
    fixture=pd.DataFrame({'model':['fixture']*8,'subject_id':['1']*4+['2']*4,
                          'truth':[0,1,0,1]*2,'prediction':[0,1,1,1]*2})
    for training,test in LeaveOneGroupOut().split(fixture,groups=fixture.subject_id):
        assert not set(fixture.iloc[training].subject_id)&set(fixture.iloc[test].subject_id)
    summary=actor_metric_table(fixture,[0,1])
    assert len(summary)==2 and list(summary.recordings)==[4,4]
    assert summary.macro_f1.between(0,1).all()


def test_predi_subset_selection_is_label_blind_and_keeps_earliest_readings():
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/12_prepare_predi_subset.py'))
    files = []
    for subject in range(140):
        for day,task in [(3,1),(1,1),(2,1),(1,2)]:
            files.append({'key':f'Predi-COVID_{subject:04d}_2020060{day}140840_{task}_m4a_M_0.wav',
                          'size':1000,'checksum':'md5:engineering_fixture','links':{'self':'https://example.invalid/fixture'}})
    selected = api['select_subset'](files)
    assert selected.subject_id.nunique()==120 and len(selected)==240
    assert selected.audio_task.eq(1).all()
    assert set(selected.recording_timestamp.str[6:8])=={'01','02'}
    changed = [{**row,'key':row['key'].replace('_M_0.wav','_M_1.wav')} for row in files]
    other = api['select_subset'](changed)
    assert list(selected.subject_id)==list(other.subject_id)
    assert list(selected.recording_timestamp)==list(other.recording_timestamp)


def test_all_three_notebook_sources_parse_before_expensive_execution():
    import ast

    root=Path(__file__).resolve().parents[1]
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        for cell in nbformat.read(path,4).cells:
            if cell.cell_type=='code':
                ast.parse(cell.source,filename=path.name)
