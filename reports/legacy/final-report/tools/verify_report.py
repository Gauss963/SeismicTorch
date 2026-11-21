"""Check the delivered report, its vector figures, matrices and replay provenance."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
DATA=BASE/'report-data'

def main():
    test=json.loads((DATA/'test-predictions.json').read_text())
    replay=json.loads((DATA/'final-training-replay.json').read_text())
    assert len(test['records'])==47
    assert len({r['day'] for r in test['records']})==21
    assert test['sets']['supplemental'][1]['matrix']==[[13,1,0],[0,12,0],[0,0,21]]
    assert test['sets']['supplemental'][2]['matrix']==[[12,2,0],[0,12,0],[0,0,21]]
    assert len(replay['history'])==20 and replay['all_state_tensors_equal']
    assert replay['state_tensor_count']==27
    assert replay['checkpoint_sha256']==test['checkpoint_sha256']
    assert hashlib.sha256((ROOT/'models/qsis/current/model.pth').read_bytes()).hexdigest()==test['checkpoint_sha256']
    figures=list((BASE/'figures').glob('*.pdf'))
    assert len(figures)==21
    for path in figures:
        reader=PdfReader(path)
        assert len(reader.pages)==1,path
        assert not list(reader.pages[0].images),f'Expected vector-only figure: {path}'
    tex=(BASE/'Project_Report.tex').read_text()
    included=re.findall(r'\\fig\[[^\]]*\]\{([^}]+)\}',tex)
    assert len(included)==20
    assert all((BASE/'figures'/f'{name}.pdf').is_file() for name in included)
    assert len((BASE/'reference_rows.tex').read_text().splitlines())==47
    sensitivity=json.loads((DATA/'sensitivity.json').read_text())
    protocol=sensitivity['protocol']
    integrity=json.loads((DATA/'sensitivity-input-integrity.json').read_text())
    assert integrity['passed'] and integrity['unique_centered_inputs']==226
    assert not integrity['cross_fold_identical_inputs']
    assert sensitivity['completed_fits']==60 and sensitivity['completed_optimization_epochs']==4800
    assert sensitivity['released_model_unchanged'] and sensitivity['confirmation_references_not_used']
    assert len(sensitivity['summaries'])==20 and protocol['groups']==58
    assert hashlib.sha256((json.dumps(protocol,indent=2,allow_nan=False)+'\n').encode()).hexdigest()==sensitivity['protocol_sha256']
    assert len({(r['configuration'],r['seed'],r['fold']) for r in sensitivity['runs']})==60
    for fold in protocol['folds']:
        assert not set(fold['train_indices']) & set(fold['validation_indices'])
        assert sorted(fold['train_indices']+fold['validation_indices'])==list(range(226))
    for run in sensitivity['runs']:
        assert run['protocol_sha256']==sensitivity['protocol_sha256']
        assert [c['epoch'] for c in run['checkpoints']]==[5,10,20,40,80]
    labels=[r['label'] for r in json.loads((DATA/'evidence.json').read_text())['selected_training']['sources']]
    group_lists=[set(f['validation_groups']) for f in protocol['folds']]
    assert sum(map(len,group_lists))==len(set.union(*group_lists))==58
    for summary in sensitivity['summaries']:
        acc=[];balanced=[]
        for seed in summary['seeds']:
            matrices=[np.zeros((3,3),int) for _ in range(3)]
            runs=[r for r in sensitivity['runs'] if r['configuration']==summary['configuration'] and r['seed']==seed['seed']]
            assert len(runs)==5
            for run in runs:
                ids=protocol['folds'][run['fold']]['validation_indices']
                cp=next(c for c in run['checkpoints'] if c['epoch']==summary['epoch'])
                for j in range(3):
                    predicted=cp['validation'][j]['predictions']
                    assert len(predicted)==len(ids)
                    for i,pred in zip(ids,predicted,strict=True):matrices[j][labels[i],pred]+=1
            for matrix,stored in zip(matrices,seed['offsets'],strict=True):
                assert matrix.tolist()==stored['confusion'] and int(matrix.sum())==226
            matrix=matrices[1]
            acc.append(matrix.trace()/matrix.sum())
            balanced.append(np.mean(matrix.diagonal()/matrix.sum(1)))
        assert abs(np.mean(acc)-summary['accuracy_mean'])<1e-12
        assert abs(np.mean(balanced)-summary['balanced_accuracy_mean'])<1e-12
    assert len((BASE/'sensitivity_rows.tex').read_text().splitlines())==20
    continuous=json.loads((DATA/'continuous-event.json').read_text())
    assert continuous['npts']==30001 and continuous['finite_centers']==29802
    dense=np.load(DATA/'continuous-event.npz')
    assert dense['raw'].shape==dense['probabilities'].shape==(30001,3)
    finite=np.isfinite(dense['probabilities']).all(1)
    assert int(finite.sum())==29802
    assert np.allclose(dense['probabilities'][finite].sum(1),1,atol=2e-6)
    assert np.isnan(dense['probabilities'][~finite]).all()

    assert continuous['checkpoint_sha256']==test['checkpoint_sha256'] and continuous['stride']==1
    assert max(r['max_probability_error'] for r in continuous['reference_checks'])<2e-6
    ci=json.loads((DATA/'arm64-ci.json').read_text())
    assert ci['conclusion']=='success'
    assert all(j['conclusion']=='success' for j in ci['jobs'] if j['name']!='release')
    assert next(j for j in ci['jobs'] if j['name']=='linux-arm64')['conclusion']=='success'
    pdf=PdfReader(BASE/'Project_Report.pdf')
    text='\n'.join(p.extract_text() for p in pdf.pages)
    assert '??' not in text,'Unresolved TeX reference'
    for token in ['97.87%','95.74%','70.97%','33,795','References','GPH-015','IHP-017','96.90%','Raspberry Pi 5']:
        assert token in text,token
    for i,p in enumerate(pdf.pages):
        assert len(p.extract_text().strip())>50,f'Unexpected empty page: {i+1}'
    result={'report_pages':len(pdf.pages),'manuscript_figures':20,'standalone_matrix_figures':1,
            'all_generated_figures_vector_pdf':True,'test_clips':47,'offset_evaluations':141,
            'center_confusion':test['sets']['supplemental'][1]['matrix'],
            'center_accuracy':46/47,'positive_offset_accuracy':45/47,
            'training_pool_unique_centered_inputs':226,'sensitivity_fits':60,'sensitivity_aggregate_epochs':4800,'sensitivity_independent_confirmation_unused':True,
            'dense_event_samples':30001,'native_arm64_ci_passed':True,'physical_pi5_tested':False,
            'exact_replay_epochs':20,'all_27_state_tensors_equal':True,
            'model_sha256':test['checkpoint_sha256'],
            'report_sha256':hashlib.sha256((BASE/'Project_Report.pdf').read_bytes()).hexdigest(),
            'figure_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figures)},
            'checks':'Source receipts, regenerated predictions, source-weight identity, vector PDF files, manuscript references and nonempty rendered document'}
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
