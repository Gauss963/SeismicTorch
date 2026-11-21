"""Snapshot report evidence and recompute all 47 held-out clips at fixed offsets."""
from pathlib import Path
import hashlib
import json
import runpy
from collections import Counter

import numpy as np
import torch
from scipy.signal import welch
from seismictorch.data.preprocessing import prepared_waveform, window
from seismictorch.data.sac import load_xyz
from seismictorch.models.waveform import EarthquakeCNN

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'report-data'

def read(path):
    return json.loads((ROOT / path).read_text())

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    sources = {
        'selected_training': 'models/qsis/current/training.json',
        'selected_evaluation': 'models/qsis/current/evaluation-ies-confirmation.json',
        'selected_model': 'models/qsis/current/model.pth',
        'stead_metrics': 'runs/waveform-2026-11-12/stead/metrics.json',
        'qsis_baseline_metrics': 'runs/waveform-2026-11-12/qsis/metrics.json',
        'cwa_metrics': 'runs/qsis-benchmark/cwa-pretrain-42/metrics.json',
        'fresh_comparison': 'runs/qsis-ies-hybrid/comparison.json',
        'prior_comparison': 'runs/qsis-prospective-gravity/comparison.json',
        'supplemental_comparison': 'runs/qsis-prospective-gravity/array-followup/supplemental-comparison.json',
        'archived_comparison': 'runs/qsis-ccul-extra/record-orientation-generalization-controlled.json',
        'verification': 'runs/qsis-ies-hybrid/verification.json',
        'cpp_parity': 'runs/cpp-inference/parity.json',
        'cpp_release': 'runs/cpp-inference/release.json',
    }
    snapshot = {k: read(p) for k, p in sources.items() if k != 'selected_model'}
    for stage in ('stead', 'qsis'):
        path = f'runs/waveform-2026-11-12/{stage}/history.jsonl'
        sources[f'{stage}_history'] = path
        snapshot[f'{stage}_history'] = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
    training = snapshot['selected_training']
    model = EarthquakeCNN().eval()
    model.load_state_dict(torch.load(ROOT / sources['selected_model'], weights_only=True, map_location='cpu'))
    cases = runpy.run_path(str(ROOT / 'tests/test_verified_model.py'))['reference_cases']()
    supplemental = snapshot['supplemental_comparison']['clips']
    fresh = snapshot['fresh_comparison']['clips']
    assert len(cases) == 47
    records, plot_arrays, shapes = [], {}, []
    for layer in (model.conv1, model.bn1, model.conv2, model.bn2, model.conv3, model.bn3, model.fc1, model.fc2, model.fc3):
        shapes.append({'name': next(n for n, m in model.named_modules() if m is layer),
                       'parameters': sum(p.numel() for p in layer.parameters())})
    for index, row in enumerate(cases):
        source = supplemental[index] if index < len(supplemental) else fresh[index - len(supplemental)]
        label = source['label']
        stream = load_xyz([ROOT / p for p in row['paths']], min_samples=200)
        all_probs = []
        center = row['center_sample']
        center_clip = None
        for j, offset in enumerate((-20, 0, 20)):
            raw, filtered, orientation = prepared_waveform(stream, center + offset, training['waveform_filter'], training['orientation'])
            clip = window(raw, filtered, center + offset)
            assert clip is not None
            with torch.inference_mode():
                probs = model(torch.from_numpy(clip[None])).softmax(1)[0].numpy()
            assert int(probs.argmax()) == row['predictions'][j], row['id']
            all_probs.append(probs.tolist())
            if offset == 0:
                center_clip = clip[0]
                plot_arrays[f'clip_{index}'] = center_clip
                plot_arrays[f'raw_{index}'] = raw[:, center-100:center+100].T
                plot_arrays[f'context_{index}'] = filtered[:, center-400:center+400].T
        record = {'id': row['id'], 'label': label, 'cohort': 'prior' if index < 20 else 'fresh',
                  'source': source.get('source', 'primary_frozen'), 'site': source.get('site', 'IES'),
                  'day': source.get('day', source.get('origin_utc', '')[:10]),
                  'center_sample': center, 'paths': row['paths'],
                  'waveform_sha256': [digest(p) for p in row['paths']],
                  'predictions': row['predictions'], 'probabilities': all_probs}
        records.append(record)
    sets = {}
    for name, subset in [('fresh', [r for r in records if r['cohort']=='fresh']),
                         ('primary', [r for r in records if r['cohort']=='fresh' or r['source']=='primary_frozen']),
                         ('supplemental', records)]:
        results = []
        for j, offset in enumerate((-.2, 0, .2)):
            matrix = np.zeros((3,3), dtype=int)
            for r in subset:
                matrix[r['label'], r['predictions'][j]] += 1
            n = int(matrix.sum()); correct = int(matrix.trace())
            recall = matrix.diagonal() / matrix.sum(1)
            precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(3), where=matrix.sum(0)!=0)
            f1 = 2 * precision * recall / (precision + recall)
            z = 1.959963984540054; p = correct/n
            middle = (p+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            results.append({'offset': offset, 'matrix': matrix.tolist(), 'n': n, 'correct': correct,
                            'accuracy': p, 'balanced_accuracy': float(recall.mean()),
                            'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(),
                            'macro_f1': float(f1.mean()), 'wilson95_clip_iid': [middle-half, middle+half]})
        sets[name] = results
    for key, name in [('fresh_test','fresh'),('combined_primary','primary'),('including_supplemental_context','supplemental')]:
        for expected, actual in zip(snapshot['selected_evaluation'][key], sets[name], strict=True):
            assert expected['confusion'] == actual['matrix']
    # Fixed, date-clustered paired bootstrap: preserve within-date P/S/noise dependence.
    dates = sorted({r['day'] for r in records})
    assert len(dates) == 21
    counts = np.zeros((len(dates),3,2),int)
    for r in records:
        k=dates.index(r['day'])
        for j in range(3): counts[k,j] += (int(r['label']==r['predictions'][j]),1)
    rng=np.random.default_rng(42)
    draws=counts[rng.integers(len(dates),size=(10000,len(dates)))].sum(1)
    scores=draws[:,:,0]/draws[:,:,1]
    bootstrap={'unit':'UTC date','dates':len(dates),'replicates':10000,'seed':42,
               'intervals':np.quantile(scores,[.025,.975],axis=0).T.tolist(),
               'note':'Descriptive percentile interval on selected references; cannot resolve unseen classes/sites or coverage bias.'}
    save('evidence.json', snapshot)
    save('test-predictions.json', {'checkpoint_sha256': training['checkpoint_sha256'], 'offsets':[-.2,0,.2],
         'records': records, 'sets': sets, 'date_bootstrap':bootstrap, 'layer_parameters': shapes})
    np.savez_compressed(OUT/'waveform-panels.npz', **plot_arrays)
    provenance={'source_files':{k:{'path':p,'sha256':digest(p)} for k,p in sources.items()},
                'model_parameter_count':sum(p.numel() for p in model.parameters()),
                'training_class_counts':training['training_classes'],
                'training_orientation_counts':dict(Counter(training['orientation_by_source'])),
                'test_clips':47,'test_evaluations':141,'test_dates':len(dates),
                'test_predictions_recomputed':True,'matrix_rows':'reference','matrix_columns':'prediction'}
    save('provenance.json', provenance)
    print(json.dumps({'sets':sets,'date_bootstrap':bootstrap,'parameters':provenance['model_parameter_count']},indent=2))

if __name__=='__main__': main()
