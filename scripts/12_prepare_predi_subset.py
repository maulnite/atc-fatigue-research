"""Audit/select Predi-COVID; download only the explicitly authorized bounded subset.

Default is metadata audit only. Review rights and obtain dataset-download permission
before supplying --download-approved-subset. This script never runs from a notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atc_fatigue.research.data import parse_predi_filename
from atc_fatigue.utils.runtime import checksum

API = "https://zenodo.org/api/records/5937844"
LIMIT = 384 * 1024**2


def select_subset(files: list[dict]) -> pd.DataFrame:
    """Seed42 participants; task1; earliest two each; selection ignores labels/model scores."""
    rows = []
    for item in files:
        metadata = parse_predi_filename(item['key'])
        rows.append({**metadata, 'key': item['key'], 'bytes': item['size'],
                     'publisher_md5': item['checksum'].removeprefix('md5:'),
                     'source_url': item['links']['self']})
    frame = pd.DataFrame(rows)
    subjects = np.random.default_rng(42).choice(sorted(frame.subject_id.unique()),120,replace=False)
    return (frame[frame.subject_id.isin(subjects) & frame.audio_task.eq(1)]
            .sort_values(['subject_id','recording_timestamp']).groupby('subject_id',sort=False).head(2)
            .reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-approved-subset', action='store_true')
    args = parser.parse_args()
    response = requests.get(API, timeout=(15,90))
    response.raise_for_status()
    record = response.json()
    if record['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Publisher license changed; re-audit before acquisition')
    subset = select_subset(record['files'])
    if len(subset)!=207 or subset.subject_id.nunique()!=120 or subset.bytes.sum()>LIMIT:
        raise ValueError('Publisher contents changed; existing subset approval no longer sufficient')
    print(json.dumps({'recordings':len(subset),'subjects':120,'bytes':int(subset.bytes.sum()),
                      'download_enabled':args.download_approved_subset}))
    if not args.download_approved_subset:
        return
    manifest = Path('data/manifests/predi_covid_subset.parquet')
    if manifest.exists():
        raise FileExistsError('Preserve existing manifest; verify/use its sources rather than overwrite')
    root = Path('data/raw/predi_covid/approved_subset_seed42')
    root.mkdir(parents=True,exist_ok=True)
    # Count any failed payloads already on disk toward the transfer budget.
    used = sum(path.stat().st_size for path in root.glob('*.part'))
    rows = []
    for item in subset.to_dict('records'):
        path = root/item.pop('key')
        if not path.exists():
            part = path.with_suffix('.wav.part')
            with requests.get(item['source_url'],stream=True,timeout=(15,90)) as stream:
                stream.raise_for_status()
                with part.open('xb') as handle:
                    for block in stream.iter_content(128*1024):
                        used += len(block)
                        if used>LIMIT:
                            raise RuntimeError('Authorized transfer budget exhausted; partial file preserved')
                        handle.write(block)
            if part.stat().st_size!=item['bytes'] or hashlib.md5(part.read_bytes()).hexdigest()!=item['publisher_md5']:
                raise ValueError('Publisher checksum mismatch; partial file preserved')
            part.rename(path)
        if path.stat().st_size!=item['bytes'] or hashlib.md5(path.read_bytes()).hexdigest()!=item['publisher_md5']:
            raise ValueError('Existing source differs from publisher')
        rows.append({**item,'sample_id':path.stem,'dataset_id':'predi_covid',
                     'target_name':'clinical_fatigue_binary','label_source':'publisher_filename_same_day_clinical_self_report',
                     'original_path':str(path),'source_checksum':checksum(path),'is_synthetic_pair':False})
    manifest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(manifest,index=False)
    print(json.dumps({'verified_sources':len(rows),'transferred_bytes':used,'max_transfer_bytes':LIMIT}))


if __name__=='__main__':
    main()
