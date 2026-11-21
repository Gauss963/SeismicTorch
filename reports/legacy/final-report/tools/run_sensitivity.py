"""Frozen training-pool depth/epoch sensitivity; independent confirmation is never read."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from seismictorch.data.preprocessing import prepared_waveform, window
from seismictorch.data.sac import load_xyz
from seismictorch.models.waveform import EarthquakeCNN
from seismictorch.training.pilot import augment_batch
from seismictorch.training.runtime import run_epoch

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/report-sensitivity'
SNAPSHOT=BASE/'report-data/sensitivity.json'
EPOCHS=[5,10,20,40,80]
SEEDS=[42,43,44]

class DepthCNN(nn.Module):
    def __init__(self, depth):
        super().__init__()
        channels={2:[16,64],4:[16,32,64,64]}[depth]
        self.blocks=nn.ModuleList()
        previous=1
        for i,width in enumerate(channels):
            padding=(1,0) if depth==4 and i==3 else 0
            # Preserve the original three temporal downsamplings in the deep
            # variant; the new block operates before the final pooling stage.
            pool=nn.Identity() if depth==4 and i==2 else nn.MaxPool2d((3,1))
            self.blocks.append(nn.Sequential(nn.Conv2d(previous,width,(3,1),padding=padding),
                nn.BatchNorm2d(width),nn.GELU(),pool,nn.Dropout(.1)))
            previous=width
        self.fc1=nn.Linear(384,64)
        self.fc2=nn.Linear(64,16)
        self.fc3=nn.Linear(16,3)
        self.activation=nn.GELU()
    def forward(self,x):
        for block in self.blocks:x=block(x)
        if x.shape[2]!=2:x=nn.functional.adaptive_avg_pool2d(x,(2,3))
        x=x.flatten(1)
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def scores(truth,pred):
    matrix=np.zeros((3,3),dtype=int)
    np.add.at(matrix,(truth,pred),1)
    recall=matrix.diagonal()/matrix.sum(1)
    precision=np.divide(matrix.diagonal(),matrix.sum(0),out=np.zeros(3),where=matrix.sum(0)!=0)
    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(3),where=(recall+precision)!=0)
    return {'accuracy':float(matrix.trace()/matrix.sum()),'balanced_accuracy':float(recall.mean()),
            'macro_f1':float(f1.mean()),'recall':recall.tolist(),'confusion':matrix.tolist()}

def main():
    OUT.mkdir(exist_ok=True)
    receipt_path=ROOT/'models/qsis/current/training.json'
    receipt=json.loads(receipt_path.read_text())
    init=ROOT/'runs/qsis-benchmark/cwa-pretrain-42/best.pth'
    released=ROOT/'models/qsis/current/model.pth'
    assert digest(init)==receipt['init_sha256'] and digest(released)==receipt['checkpoint_sha256']
    assert torch.backends.mps.is_available()
    device=torch.device('mps')
    rows=receipt['sources']
    # Union all origin/start/clip dates in a recording so midnight boundaries
    # cannot split one earthquake, recording, or background day between folds.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(b)]=find(a)
    groups=[];views=[[],[],[]]
    for row in rows:
        stream=load_xyz([ROOT/p for p in row['paths']],min_samples=200)
        assert stream is not None
        group_day=row['group'][:10]
        dates=[group_day,str(stream[0].stats.starttime.date),
               str((stream[0].stats.starttime+row['center_sample']/100).date)]
        for day in dates:join(dates[0],day)
        groups.append(dates[0])
        for j,offset in enumerate((-20,0,20)):
            c=row['center_sample']+offset
            raw,filtered,_=prepared_waveform(stream,c,receipt['waveform_filter'],receipt['orientation'])
            clip=window(raw,filtered,c)
            if clip is None:raise ValueError(f'Unusable sensitivity view {row["id"]} / {offset}')
            views[j].append(clip)
    groups=[find(x) for x in groups]
    x=[torch.from_numpy(np.stack(v)) for v in views]
    y=torch.tensor([r['label'] for r in rows]);truth=y.numpy()
    assert hashlib.sha256(x[1].numpy().tobytes()).hexdigest()==json.loads((BASE/'report-data/final-training-replay.json').read_text())['training_input_sha256']
    # Freeze balanced grouped folds before any fitting or prediction.
    # Restarts optimize only class/size balance; no model result is consulted.
    unique=sorted(set(groups));best=None
    for trial in range(200):
        rng=np.random.default_rng(20261121+trial)
        order=list(rng.permutation(unique))
        counts={g:np.bincount(truth[np.array(groups)==g],minlength=3) for g in unique}
        order.sort(key=lambda g:float(np.max(counts[g]/np.bincount(truth,minlength=3))),reverse=True)
        totals=np.zeros((5,3),int);assignment={}
        for g in order:
            costs=[]
            for f in range(5):
                proposed=totals.copy();proposed[f]+=counts[g]
                costs.append(np.std(proposed/np.bincount(truth,minlength=3),axis=0).mean()+.1*np.std(proposed.sum(1)/len(rows)))
            f=int(np.argmin(costs));totals[f]+=counts[g];assignment[g]=f
        if (totals==0).any():continue
        cost=np.std(totals/np.bincount(truth,minlength=3),axis=0).mean()
        if best is None or cost<best[0]:best=(cost,assignment,totals)
    assert best is not None
    assignment=best[1]
    folds=[]
    for f in range(5):
        valid=[i for i,g in enumerate(groups) if assignment[g]==f]
        train=sorted(set(range(len(rows)))-set(valid))
        assert not {groups[i] for i in train}&{groups[i] for i in valid}
        folds.append({'fold':f,'train_indices':train,'validation_indices':valid,
                      'validation_groups':sorted({groups[i] for i in valid}),
                      'validation_classes':np.bincount(truth[valid],minlength=3).tolist()})
    input_hashes=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in x[1]]
    duplicates={h:[i for i,v in enumerate(input_hashes) if v==h] for h in set(input_hashes) if input_hashes.count(h)>1}
    fold_for={i:f['fold'] for f in folds for i in f['validation_indices']}
    cross=[ids for ids in duplicates.values() if len({fold_for[i] for i in ids})>1]
    assert not cross,'Identical normalized inputs cross validation folds'
    save(BASE/'report-data/sensitivity-input-integrity.json',{
        'sources':226,'unique_centered_inputs':len(set(input_hashes)),
        'repeated_input_groups':list(duplicates.values()),'cross_fold_identical_inputs':cross,
        'passed':True,'training_sources_only':True})
    configs=[{'name':f'scratch-conv{d}','depth':d,'initialization':'random'} for d in (2,3,4)]
    configs.append({'name':'cwa-conv3','depth':3,'initialization':'CWA'})
    protocol={'purpose':'Post-release development sensitivity on fitting sources only; not independent test accuracy',
        'training_receipt_sha256':digest(receipt_path),'released_model_sha256':digest(released),
        'init_sha256':digest(init),'source_count':len(rows),'groups':len(unique),
        'group_definition':'Union of origin, SAC start and centered-sample UTC dates',
        'folds':folds,'seeds':SEEDS,'epoch_checkpoints':EPOCHS,'configurations':configs,
        'fc_dimensions':[384,64,16,3],'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.01},
        'batch_size':16,'augmentation':'Original reflected +/-8-sample shift and +/-30 degree horizontal rotation',
        'criterion':'Reciprocal training-class-frequency weighted CE; validation uses unweighted CE',
        'epochs_scope':'QSIS fitting updates only; CWA initializer has six earlier adaptation epochs and STEAD ancestry',
        'depth_scope':'2: channels16/64 with two max pools then adaptive average pool to2; 3: exact original16/32/64; 4: add64-channel padded convolution before final max pool. Depth comparison includes its stated pooling change.',
        'independent_confirmation_accessed':False,'device':'mps','torch_version':torch.__version__}
    protocol_file=OUT/'protocol.json'
    if protocol_file.exists():assert json.loads(protocol_file.read_text())==protocol,'Existing protocol differs'
    else:save(protocol_file,protocol)
    protocol_hash=digest(protocol_file)
    start=time.monotonic();all_runs=[]
    for cfg in configs:
        for seed in SEEDS:
            for fold in folds:
                path=OUT/f"{cfg['name']}-seed{seed}-fold{fold['fold']}.json"
                if path.exists():
                    result=json.loads(path.read_text());assert result['protocol_sha256']==protocol_hash
                    all_runs.append(result);continue
                torch.manual_seed(seed)
                model=EarthquakeCNN() if cfg['depth']==3 else DepthCNN(cfg['depth'])
                if cfg['initialization']=='CWA':model.load_state_dict(torch.load(init,weights_only=True,map_location='cpu'))
                model.to(device)
                train,valid=fold['train_indices'],fold['validation_indices']
                loader=DataLoader(TensorDataset(x[1][train],y[train]),batch_size=16,shuffle=True,collate_fn=augment_batch)
                weights=y[train].bincount(minlength=3).float().reciprocal().to(device)
                criterion=nn.CrossEntropyLoss(weight=weights)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
                validation=[v[valid].to(device) for v in x]
                train_x=x[1][train].to(device)
                checkpoints=[]
                run_start=time.monotonic()
                for epoch in range(1,max(EPOCHS)+1):
                    training=run_epoch(model,loader,device,criterion,optimizer)
                    if epoch not in EPOCHS:continue
                    state=torch.get_rng_state();mps_state=torch.mps.get_rng_state()
                    model.eval()
                    with torch.inference_mode():
                        logits=[model(v).cpu() for v in validation]
                        pred=[l.argmax(1).numpy() for l in logits]
                        train_logits=model(train_x).cpu()
                    torch.set_rng_state(state);torch.mps.set_rng_state(mps_state)
                    clean_train=scores(truth[train],train_logits.argmax(1).numpy())
                    checkpoints.append({'epoch':epoch,'validation':[
                        {**scores(truth[valid],p),'predictions':p.tolist(),
                         'loss':float(nn.functional.cross_entropy(l,y[valid]))}
                        for p,l in zip(pred,logits,strict=True)],
                        'training_clean':{**clean_train,'loss':float(nn.functional.cross_entropy(train_logits,y[train]))},
                        'training_augmented':training})
                result={'protocol_sha256':protocol_hash,'configuration':cfg['name'],'depth':cfg['depth'],
                    'initialization':cfg['initialization'],'seed':seed,'fold':fold['fold'],
                    'parameters':sum(p.numel() for p in model.parameters()),'checkpoints':checkpoints,
                    'elapsed_seconds':time.monotonic()-run_start}
                save(path,result);all_runs.append(result)
                print(f"{len(all_runs):02}/60 {path.stem}: center BA at80 {checkpoints[-1]['validation'][1]['balanced_accuracy']:.3f}; {result['elapsed_seconds']:.1f}s",flush=True)
    summaries=[]
    for cfg in configs:
        for epoch in EPOCHS:
            seeds=[]
            for seed in SEEDS:
                predictions=np.full((3,len(rows)),-1,int);train_metrics=[]
                for run in all_runs:
                    if run['configuration']!=cfg['name'] or run['seed']!=seed:continue
                    cp=next(c for c in run['checkpoints'] if c['epoch']==epoch)
                    fold=folds[run['fold']]
                    for j in range(3):predictions[j,fold['validation_indices']]=cp['validation'][j]['predictions']
                    train_metrics.append(cp['training_clean'])
                assert (predictions>=0).all()
                seeds.append({'seed':seed,'offsets':[scores(truth,p) for p in predictions],
                    'mean_clean_training_accuracy':float(np.mean([m['accuracy'] for m in train_metrics]))})
            center=np.array([s['offsets'][1]['accuracy'] for s in seeds]);ba=np.array([s['offsets'][1]['balanced_accuracy'] for s in seeds])
            worst=np.array([min(s['offsets'][j]['balanced_accuracy'] for j in range(3)) for s in seeds])
            summaries.append({'configuration':cfg['name'],'depth':cfg['depth'],'initialization':cfg['initialization'],
                'epoch':epoch,'parameters':next(r['parameters'] for r in all_runs if r['configuration']==cfg['name']),
                'seeds':seeds,'accuracy_mean':float(center.mean()),'accuracy_sd':float(center.std(ddof=1)),
                'balanced_accuracy_mean':float(ba.mean()),'balanced_accuracy_sd':float(ba.std(ddof=1)),
                'worst_offset_balanced_accuracy_mean':float(worst.mean()),
                'clean_train_accuracy_mean':float(np.mean([s['mean_clean_training_accuracy'] for s in seeds]))})
    assert digest(released)==protocol['released_model_sha256']
    payload={'protocol':protocol,'protocol_sha256':protocol_hash,'runs':all_runs,'summaries':summaries,
             'completed_fits':len(all_runs),'completed_optimization_epochs':len(all_runs)*max(EPOCHS),
             'released_model_unchanged':True,'confirmation_references_not_used':True,
             'wall_seconds_this_invocation':time.monotonic()-start}
    save(OUT/'results.json',payload);save(SNAPSHOT,payload)
    print('COMPLETED',len(all_runs),'fits;',len(all_runs)*max(EPOCHS),'epochs; release unchanged')

if __name__=='__main__':main()
