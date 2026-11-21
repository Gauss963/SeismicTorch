"""Run the published dense native path on a full held-out event recording."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import numpy as np
from obspy import read

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/report-continuous'

def main():
    OUT.mkdir(exist_ok=True)
    evidence=json.loads((BASE/'report-data/evidence.json').read_text())
    records=evidence['fresh_comparison']['clips']
    phase=[r for r in records if r['id']=='IHP-003' and r['label'] in (0,1)]
    assert len(phase)==2
    row=phase[0];paths=[ROOT/p for p in row['paths']]
    binary=ROOT/'build/native/seismictorch-infer'
    version=subprocess.check_output([str(binary),'--version'],text=True).strip()
    expected=evidence['selected_training']['checkpoint_sha256']
    assert f'model={expected}' in version
    destination=OUT/'IHP-003.csv'
    subprocess.run([str(binary),'--sac',*map(str,paths),'--threads','4','--output',str(destination)],check=True)
    stream=[read(str(p))[0] for p in paths]
    raw=np.stack([tr.data for tr in stream],axis=1)
    with destination.open() as f:predictions=list(csv.DictReader(f))
    probabilities=np.array([[float(r[k]) for k in ('P','S','Noise')] for r in predictions])
    assert len(probabilities)==len(raw)
    finite=np.isfinite(probabilities).all(1)
    assert np.allclose(probabilities[finite].sum(1),1,atol=2e-6)
    assert np.isnan(probabilities[~finite]).all()
    snapshot=json.loads((BASE/'report-data/test-predictions.json').read_text())
    checks=[]
    for pick in phase:
        reference=next(r for r in snapshot['records'] if r['id']==pick['id'] and r['label']==pick['label'])
        error=float(np.max(np.abs(probabilities[pick['center_sample']]-reference['probabilities'][1])))
        assert error<2e-6
        checks.append({'phase':pick['phase'],'center_sample':pick['center_sample'],'max_probability_error':error})
    baseline=np.median(raw[100:700],axis=0)
    np.savez_compressed(BASE/'report-data/continuous-event.npz',raw=raw,probabilities=probabilities,baseline=baseline)
    info={'id':row['id'],'site':row['site'],'station':row['station'],'event':row['event'],
          'origin_utc':row['origin_utc'],'start_utc':row['start_utc'],'origin_offset':row['origin_offset'],
          'sample_rate':100,'npts':len(raw),'paths':row['paths'],
          'waveform_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
          'checkpoint_sha256':expected,'binary_version':version,'stride':1,'threads':4,
          'finite_centers':int(finite.sum()),'nan_centers':int((~finite).sum()),
          'reference_times_after_origin':{p['phase']:p['onset_center'] for p in phase},
          'reference_checks':checks,'plot_excerpt_after_origin':[-12,70],
          'scope':'Full-record preprocessing and dense 100 Hz inference; only the two reviewed phase clips have labels. This is a qualitative display, not continuous detection accuracy.'}
    (BASE/'report-data/continuous-event.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
