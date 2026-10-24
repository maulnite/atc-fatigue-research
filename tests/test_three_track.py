"""Engineering fixtures: channel physics, train-only representations and publication."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atc_fatigue.audio.radio import frequency_response, radio_style, recording_seed
from atc_fatigue.research.representations import inspect_model_input, binary_ranking_metrics

CHANNEL={'band_hz':[300,3400],'filter_order':4,'gain_db':-3.,'snr_db':20.,'peak_limit':.98}


def fixture_signal():
    time=np.arange(16000)/16000
    return .2*np.sin(2*np.pi*1000*time)+.1*np.sin(2*np.pi*6000*time)


def test_channel_preserves_duration_measured_snr_and_determinism():
    signal=fixture_signal()
    actual=radio_style(signal,16000,CHANNEL,recording_seed('fixture',42))
    second=radio_style(signal,16000,CHANNEL,recording_seed('fixture',42))
    np.testing.assert_array_equal(actual['signal'],second['signal'])
    assert actual['signal'].shape==signal.shape
    assert actual['qc']['measured_snr_db']==pytest.approx(20.,abs=1e-5)
    assert actual['qc']['radio_clipping_fraction']==0
    assert actual['qc']['radio_inband_energy_fraction']>actual['qc']['source_inband_energy_fraction']
    assert recording_seed('fixture',42)!=recording_seed('other-fixture',42)


def test_channel_rejects_silence_and_inadequate_bandwidth():
    with pytest.raises(ValueError,match='No usable'):
        radio_style(np.zeros(16000),16000,CHANNEL,42)
    with pytest.raises(ValueError,match='filter'):
        radio_style(fixture_signal(),6000,CHANNEL,42)
    frequencies,db=frequency_response(16000,CHANNEL)
    assert db[np.argmin(abs(frequencies-1000))]>-1
    assert db[np.argmin(abs(frequencies-6000))]<-40


def test_representation_uses_fitted_training_statistics_and_excludes_ids():
    train=pd.DataFrame({'sample_id':['a','b','c','d'],'x':[0.,1.,np.nan,3.],'z':[1.,2.,4.,8.],'target_value':[0,0,1,1]})
    test=pd.DataFrame({'sample_id':['e'],'x':[1000.],'z':[1000.],'target_value':[1]})
    model=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression())
    columns=['x','z']
    model.fit(train[columns],train.target_value)
    before=model[1].mean_.copy()
    view=inspect_model_input(model,train,test,columns)
    np.testing.assert_array_equal(model[1].mean_,before)
    assert view['X_train_shape']==(4,2) and view['raw_missing_cells']==1
    assert view['example_vector'].transformed.max()>100
    with pytest.raises(ValueError,match='leakage'):
        inspect_model_input(model,train,test,['subject_id'])
    with pytest.raises(ValueError,match='declared training'):
        inspect_model_input(model,train.iloc[:3],test,columns)


def test_binary_ranking_scores_need_not_be_probabilities():
    result=binary_ranking_metrics([0,0,1,1],[-5,-3,2,8])
    assert result=={'roc_auc':1.,'average_precision':1.}


@pytest.mark.parametrize('payload',[
    {'audio/wav':'engineering-fixture-not-audio'},
    {'text/html':'<audio controls src="data:audio/wav;base64,fixture"></audio>'},
    {'text/html':'<video controls src="fixture.mp4"></video>'},
])
def test_publication_removes_untagged_media_without_mutating_local_evidence(monkeypatch,payload):
    import os
    import runpy
    from pathlib import Path
    import nbformat
    from atc_fatigue.research.runs import notebook_source_hash

    root=Path(__file__).resolve().parents[1]
    with monkeypatch.context() as context:
        context.setattr(os,'environ',dict(os.environ))
        context.chdir(root)
        api=runpy.run_path(str(root/'scripts/10_execute_notebook_v3.py'))
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('fixture = True')])
        notebook.cells[0].outputs=[nbformat.v4.new_output('display_data',data=payload)]
        public=api['publication_copy'](notebook)
        assert public.cells[0].outputs[0].output_type=='stream'
        assert notebook.cells[0].outputs[0].data==payload
        assert notebook_source_hash(notebook)==notebook_source_hash(public)


def test_raw_source_denial_is_process_local_and_allows_cache(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    raw=tmp_path/'data/raw'
    raw.mkdir(parents=True)
    cache=tmp_path/'runs'
    cache.mkdir()
    (raw/'fixture.txt').write_text('engineering fixture')
    (cache/'fixture.txt').write_text('engineering fixture')
    script="""
from pathlib import Path
from atc_fatigue.research.data import deny_raw_media_reads
deny_raw_media_reads()
assert Path('runs/fixture.txt').read_text() == 'engineering fixture'
try:
    Path('data/raw/fixture.txt').read_text()
except PermissionError:
    pass
else:
    raise AssertionError('Raw source access was allowed')
"""
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=env,check=True,capture_output=True,timeout=30)
    assert (raw/'fixture.txt').read_text()=='engineering fixture'
