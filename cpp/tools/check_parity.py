"""Compare native SAC preprocessing plus inference against Python on all held-out windows."""
from pathlib import Path
from collections import defaultdict
import argparse,json,runpy,subprocess
import numpy as np
import torch
from seismictorch.data.preprocessing import prepared_waveform,window
from seismictorch.data.sac import load_xyz
from seismictorch.models.waveform import EarthquakeCNN

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,default=ROOT/'build/native/seismictorch-parity-probe')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/cpp-inference/parity.json')
    args=parser.parse_args()
    cases=runpy.run_path(str(ROOT/'tests/test_verified_model.py'))['reference_cases']()
    grouped=defaultdict(list)
    for row in cases:
        for j,d in enumerate((-20,0,20)):grouped[tuple(row['paths'])].append((row,row['center_sample']+d,row['predictions'][j]))
    candidate=ROOT/'models/qsis/current';receipt=json.loads((candidate/'training.json').read_text())
    model=EarthquakeCNN().eval();model.load_state_dict(torch.load(candidate/'model.pth',weights_only=True,map_location='cpu'))
    results=[]
    for names,rows in grouped.items():
        paths=[ROOT/p for p in names]
        process=subprocess.run([str(args.probe),*[str(p) for p in paths],*[str(center) for _,center,_ in rows]],capture_output=True,text=True)
        if process.returncode:raise RuntimeError(process.stderr)
        native=np.array([[float(x) for x in line.split()] for line in process.stdout.splitlines()])
        assert native.shape==(len(rows),3)
        stream=load_xyz(paths,min_samples=700)
        for (row,center,label),actual in zip(rows,native,strict=True):
            raw,signal,_=prepared_waveform(stream,center,receipt['waveform_filter'],receipt['orientation'])
            clip=window(raw,signal,center)
            with torch.inference_mode():expected=model(torch.from_numpy(clip[None])).softmax(-1).numpy()[0]
            error=float(np.max(np.abs(actual-expected)))
            assert error<2e-4,(row['id'],center,error,actual,expected)
            assert actual.argmax()==label,(row['id'],center,actual,label)
            results.append({'id':row['id'],'center_sample':center,'expected_label':label,'max_probability_error':error})
    report={'checkpoint_sha256':receipt['checkpoint_sha256'],'cases':len(results),'recordings':len(grouped),'same_predictions':True,'max_probability_error':max(r['max_probability_error'] for r in results),'scope':'Native SAC decoding, full-record detrend/high-pass, gravity alignment, Fourier resampling and embedded CNN against unchanged Python checkpoint','records':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='records'})

if __name__=='__main__':main()
