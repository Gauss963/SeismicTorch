"""Generate committed native parity fixtures from Python and the verified model."""
from pathlib import Path
import hashlib,json,runpy,struct
import numpy as np
from obspy import Stream,Trace,UTCDateTime
from scipy.signal import butter,detrend,resample,sosfilt,sosfilt_zi
from scipy.spatial.transform import Rotation
import torch
from seismictorch.data.preprocessing import prepared_waveform,window
from seismictorch.data.sac import load_xyz
from seismictorch.models.waveform import EarthquakeCNN

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'cpp/tests/fixtures'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=EarthquakeCNN().eval();model.load_state_dict(torch.load(ROOT/'models/qsis/current/model.pth',weights_only=True,map_location='cpu'))
    receipt=json.loads((ROOT/'models/qsis/current/training.json').read_text())
    def prepare(stream,center):
        raw,filtered,_=prepared_waveform(stream,center,receipt['waveform_filter'],receipt['orientation'])
        return window(raw,filtered,center)
    def probabilities(clip):
        with torch.inference_mode():return model(torch.from_numpy(clip[None])).softmax(-1).numpy()[0]
    cases=runpy.run_path(str(ROOT/'tests/test_verified_model.py'))['reference_cases']()
    entries=[]
    with (OUT/'model-cases.bin').open('wb') as output:
        output.write(b'NNC1'+struct.pack('<I',len(cases)*3))
        for row in cases:
            stream=load_xyz([ROOT/p for p in row['paths']],min_samples=700)
            for j,d in enumerate((-20,0,20)):
                clip=prepare(stream,row['center_sample']+d);probs=probabilities(clip)
                assert int(probs.argmax())==row['predictions'][j]
                output.write(clip.astype('<f4').tobytes());output.write(probs.astype('<f4').tobytes())
                entries.append({'id':row['id'],'offset_samples':d,'predicted':int(probs.argmax())})
    n=1400;t=np.arange(n)/100;rng=np.random.default_rng(20261121)
    xyz=rng.integers(-3,4,(3,n)).astype(np.float64)
    xyz[0]+=980
    for a in range(3):
        xyz[a]+=np.rint((a+1)*12*np.sin(2*np.pi*(5+a)*t)*np.exp(-((t-9)/.3)**2))
        xyz[a]+=np.rint((3-a)*20*np.sin(2*np.pi*(3+a)*t)*np.exp(-((t-10.8)/.5)**2))
    start=UTCDateTime('2024-01-01T00:00:00Z')
    stream=Stream([Trace(axis.astype(np.int32),header={'sampling_rate':100.,'starttime':start,'network':'XX','station':'TEST','channel':'HL'+component}) for axis,component in zip(xyz,'XYZ')])
    for tr,a in zip(stream,'xyz'):
        tr.write(str(OUT/f'{a}.sac'),format='SAC',byteorder='<')
        tr.write(str(OUT/f'{a}-big.sac'),format='SAC',byteorder='>')
    stream.write(str(OUT/'steim2.mseed'),format='MSEED',encoding='STEIM2',reclen=512)
    float_stream=stream.copy()
    for tr,component in zip(float_stream,'ENZ'):
        tr.data=tr.data.astype(np.float32);tr.stats.channel='HN'+component
    float_stream.write(str(OUT/'float32.mseed'),format='MSEED',encoding='FLOAT32',reclen=512)
    centers=[100,500,799,800,900,1000,1300]
    with (OUT/'pipeline.bin').open('wb') as output:
        output.write(b'PRE1'+struct.pack('<IdqI',n,100.,int(start.ns),len(centers)))
        output.write(xyz.T.astype('<f8').tobytes())
        for center in centers:
            clip=prepare(stream,center);probs=probabilities(clip)
            output.write(struct.pack('<I',center));output.write(clip.astype('<f4').tobytes());output.write(probs.astype('<f4').tobytes())
    for rate in (40,50,200,1000):
        count=14*rate
        source=xyz[:,np.minimum((np.arange(count)*100/rate).astype(int),n-1)]
        coefficients=butter(4,.5,btype='highpass',fs=rate,output='sos')
        values=detrend(source,axis=-1,type='linear')
        signal,_=sosfilt(coefficients,values,axis=-1,zi=sosfilt_zi(coefficients)[:,None,:]*values[None,:,:1])
        blocks=np.median(source[:,rate:7*rate].reshape(3,6,rate),axis=-1)
        up=np.median(blocks,axis=-1);up=up/np.linalg.norm(up)
        rotation=Rotation.align_vectors([[0.,0.,1.]],up[None])[0].as_matrix()
        checks=[rate,5*rate,8*rate-1,8*rate,9*rate,10*rate,13*rate]
        with (OUT/f'pipeline-{rate}.bin').open('wb') as output:
            output.write(b'PRE1'+struct.pack('<IdqI',count,float(rate),int(start.ns),len(checks)))
            output.write(source.T.astype('<f8').tobytes())
            for center in checks:
                segment=signal[:,center-rate:center+rate]
                if center>=8*rate:segment=rotation@segment
                normalized=resample(segment,80,axis=-1).T
                clip=(normalized/np.abs(normalized).max())[None].astype(np.float32)
                output.write(struct.pack('<I',center));output.write(clip.astype('<f4').tobytes());output.write(probabilities(clip).astype('<f4').tobytes())
    (OUT/'provenance.json').write_text(json.dumps({'checkpoint_sha256':receipt['checkpoint_sha256'],'source':'47 reviewed waveform cases at three offsets; integer synthetic end-to-end reader fixture','model_cases':entries,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ('.bin','.sac','.mseed')}},indent=2)+'\n')
    print('Generated',len(entries),'real model cases and',len(centers),'complete preprocessing cases')

if __name__=='__main__':main()
