"""Generate all report figures as vector PDF from frozen, auditable evidence."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy.signal import welch, stft

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'report-data'
FIG=BASE/'figures'
E=json.loads((DATA/'evidence.json').read_text())
T=json.loads((DATA/'test-predictions.json').read_text())
H=json.loads((DATA/'final-training-replay.json').read_text())
W=np.load(DATA/'waveform-panels.npz')
LABELS=['P','S','Noise']
COLORS=['#2475a7','#d46a28','#32836b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.bbox':'tight','axes.axisbelow':True})
FIG.mkdir(exist_ok=True)

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',metadata={'Title':name.replace('_',' '),'Author':'SeismicTorch project','CreationDate':None,'ModDate':None})
    plt.close(fig)

def matrix(ax,m,title):
    m=np.array(m); fractions=m/m.sum(1)[:,None]
    ax.pcolormesh(np.arange(4),np.arange(4),fractions,cmap='Blues',vmin=0,vmax=1,edgecolors='white',linewidth=2)
    ax.set(xlim=(0,3),ylim=(3,0),xticks=np.arange(3)+.5,yticks=np.arange(3)+.5,
           xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted class',ylabel='Reference class',title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j+.5,i+.5,f'{m[i,j]}\n{100*fractions[i,j]:.1f}%',ha='center',va='center',
                    color='white' if fractions[i,j]>.55 else '#17283c',fontsize=11)
    ax.set_aspect('equal');ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)

fig,axes=plt.subplots(1,3,figsize=(10.1,3.25),layout='constrained')
for ax,key,title in zip(axes,['fresh','primary','supplemental'],['Fresh IES: 27/27 (100%)','Primary cohorts: 44/44 (100%)','With supplemental: 46/47 (97.87%)']):
    matrix(ax,T['sets'][key][1]['matrix'],title)
save(fig,'confusion_matrix_test')
fig,ax=plt.subplots(figsize=(4.7,4.6),layout='constrained')
matrix(ax,T['sets']['supplemental'][1]['matrix'],'QSIS confirmation: 46/47 (97.87%)')
save(fig,'confusion_matrix_qsis_47')
fig,axes=plt.subplots(1,3,figsize=(10.1,3.25),layout='constrained')
for ax,row in zip(axes,T['sets']['supplemental']):matrix(ax,row['matrix'],f"Offset {row['offset']:+.1f} s: {row['correct']}/{row['n']} ({row['accuracy']:.2%})")
save(fig,'confusion_matrix_offsets')
fig,axes=plt.subplots(1,2,figsize=(8.4,3.8),layout='constrained')
for ax,key,title in zip(axes,['stead_metrics','qsis_baseline_metrics'],['Prepared STEAD test: 94.14%','Early QSIS archive test: 70.97%']):matrix(ax,E[key]['test']['confusion_matrix'],title)
save(fig,'confusion_matrix_historical')

for key,filename,title in [('stead','training_stead','Prepared STEAD pretraining'),('qsis','training_qsis_baseline','Early QSIS archive fine-tuning')]:
    rows=E[f'{key}_history'];best=E['stead_metrics' if key=='stead' else 'qsis_baseline_metrics']['best_epoch']
    fig,axs=plt.subplots(1,3,figsize=(10,3.15),layout='constrained')
    epochs=[r['epoch'] for r in rows]
    for split,col,label in [('train',COLORS[0],'Training'),('validation',COLORS[1],'Validation')]:
        axs[0].plot(epochs,[r[split]['loss'] for r in rows],color=col,label=label,lw=1.5)
        axs[1].plot(epochs,[100*r[split]['accuracy'] for r in rows],color=col,label=label,lw=1.5)
    axs[0].set(ylabel='Cross-entropy',title='Recorded loss')
    axs[1].set(ylabel='Accuracy (%)',title='Recorded accuracy')
    axs[2].step(epochs,[r['lr'] for r in rows],where='post',color=COLORS[2]);axs[2].set(yscale='log',ylabel='Learning rate',title='StepLR schedule')
    for ax in axs:
        ax.axvline(best,color='#666666',ls='--',lw=1,label=f'Selected epoch {best}');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axs[0].legend();axs[1].legend(loc='lower right')
    fig.suptitle(title,fontsize=12)
    save(fig,filename)

fig,axs=plt.subplots(1,2,figsize=(8.6,3.1),layout='constrained')
rows=E['cwa_metrics']['history'];epochs=[r['epoch'] for r in rows]
for split,col in [('train',COLORS[0]),('validation',COLORS[1])]:
    axs[0].plot(epochs,[r[split]['loss'] for r in rows],'o-',color=col,label=split.capitalize())
    axs[1].plot(epochs,[100*r[split]['balanced_accuracy'] for r in rows],'o-',color=col,label=split.capitalize())
axs[0].set(ylabel='Cross-entropy',title='CWA strong-motion adaptation loss')
axs[1].set(ylabel='Balanced accuracy (%)',title='CWA event-held-out validation')
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
save(fig,'training_cwa')

fig,axs=plt.subplots(1,3,figsize=(10,3.25),layout='constrained');rows=H['history'];epochs=[r['epoch'] for r in rows]
for split,col,label in [('train_augmented',COLORS[0],'Augmented training'),('train_clean',COLORS[1],'Clean training (eval mode)')]:
    axs[0].plot(epochs,[r[split]['loss'] for r in rows],color=col,label=label)
    axs[1].plot(epochs,[100*r[split]['accuracy'] for r in rows],color=col,label=label)
    axs[2].plot(epochs,[100*r[split]['balanced_accuracy'] for r in rows],color=col,label=label)
axs[0].set(ylabel='Loss (different reductions)',title='Training objective / clean CE')
axs[1].set(ylabel='Accuracy (%)',title='Fitting-clip accuracy')
axs[2].set(ylabel='Balanced accuracy (%)',title='Mean class recall')
axs[0].legend(loc='upper right',fontsize=7)
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.set_xticks([1,5,10,15,20])
fig.suptitle('Released model: exact MPS replay, 226 fitting clips; no test monitoring',fontsize=11)
save(fig,'training_final')

fig,axs=plt.subplots(1,2,figsize=(8.8,3.4),layout='constrained')
counts=[int(E['selected_training']['training_classes'][str(i)]) for i in range(3)]
axs[0].bar(LABELS,counts,color=COLORS)
for i,n in enumerate(counts):axs[0].text(i,n+3,str(n),ha='center')
axs[0].set(title='Final fitting sources (N=226)',ylabel='Two-second source windows',ylim=(0,170))
values=np.array([[9,6,12],[3,5,9],[2,1,0]])
bottom=np.zeros(3)
for i,label in enumerate(LABELS):
    axs[1].bar(['Fresh IES','Prior primary','Supplemental'],values[:,i],bottom=bottom,color=COLORS[i],label=label)
    for j,n in enumerate(values[:,i]):
        if n:axs[1].text(j,bottom[j]+n/2,str(n),ha='center',va='center',color='white')
    bottom+=values[:,i]
axs[1].set(title='Distinct confirmation clips (N=47)',ylabel='Reference clips',ylim=(0,32));axs[1].legend(ncol=3,loc='upper right')
save(fig,'dataset_composition')

fig,ax=plt.subplots(figsize=(8.2,3.1),layout='constrained')
labels=['P phase slots','S phase slots','Background candidates'];valid=np.array([9,6,12]);unresolved=np.array([19,22,2]);excluded=np.array([1,1,0]);y=np.arange(3)
for value,left,col,name in [(valid,np.zeros(3),COLORS[2],'Agreed/scored'),(unresolved,valid,'#c6ced7','Unresolved'),(excluded,valid+unresolved,'#a75954','Raw QC excluded')]:
    ax.barh(y,value,left=left,color=col,label=name)
    for j,v in enumerate(value):
        if v:ax.text(left[j]+v/2,j,str(v),ha='center',va='center',fontsize=9)
ax.set(yticks=y,yticklabels=labels,xlabel='Enrolled slots or candidates',xlim=(0,32),title='Fresh IES reference coverage: 58 phase slots and 14 backgrounds')
ax.invert_yaxis();ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,-.24),fontsize=9)
save(fig,'reference_coverage')

fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for key,col,label in [('fresh',COLORS[0],'Fresh IES (27)'),('primary',COLORS[2],'Primary (44)'),('supplemental',COLORS[1],'With supplemental (47)')]:
    rows=T['sets'][key];axs[0].plot([r['offset'] for r in rows],[100*r['accuracy'] for r in rows],'o-',label=label,color=col)
axs[0].axhline(95,color='#777777',ls='--',lw=1);axs[0].set(xlabel='Raw-window center offset (s)',ylabel='Accuracy (%)',xticks=[-.2,0,.2],ylim=(93.8,101.2),title='Fixed timing sensitivity');axs[0].legend(loc='lower left')
for j,(key,label) in enumerate([('fresh','Fresh IES'),('primary','Primary'),('supplemental','With supplemental')]):
    r=T['sets'][key][1];lo,hi=r['wilson95_clip_iid'];a=r['accuracy']
    axs[1].errorbar(100*a,j,xerr=[[max(0,100*(a-lo))],[max(0,100*(hi-a))]],fmt='o',color=COLORS[j],capsize=4)
    axs[1].text(100*a+.4,j,f"{r['correct']}/{r['n']}",va='center',fontsize=8)
axs[1].set(yticks=[0,1,2],yticklabels=['Fresh IES','Primary','With supplemental'],xlabel='Centered accuracy (%)',xlim=(84,104),ylim=(2.6,-.6),title='95% Wilson intervals (clip-iid illustration)');axs[1].axvline(95,color='#777777',ls='--')
for ax in axs:ax.grid(alpha=.2)
save(fig,'timing_and_uncertainty')

indices=[next(i for i,r in enumerate(T['records']) if r['cohort']=='fresh' and r['label']==k) for k in range(3)]
fig,axs=plt.subplots(2,3,figsize=(10,5.4),layout='constrained')
for col,i in enumerate(indices):
    row=T['records'][i];raw=W[f'raw_{i}'];clip=W[f'clip_{i}'];tt=np.arange(200)/100-1
    raw=raw-np.median(raw,axis=0)
    for a in range(3):
        axs[0,col].plot(tt,raw[:,a],color=COLORS[a],lw=.9,label='XYZ'[a])
        axs[1,col].plot(np.arange(80)/40-1,clip[:,a],color=COLORS[a],lw=1,label='XYZ'[a])
    axs[0,col].set(title=f"{LABELS[row['label']]} / {row['id']}",ylabel='Raw deviation (gal)' if col==0 else '')
    axs[1,col].set(xlabel='Time relative to window center (s)',ylabel='Normalized amplitude' if col==0 else '',ylim=(-1.12,1.12))
    probs=row['probabilities'][1];axs[1,col].set_title('P/S/Noise = '+ '/'.join(f'{x:.3f}' for x in probs),fontsize=9)
    for ax in axs[:,col]:ax.axvline(0,color='#777777',lw=.7,ls='--');ax.grid(alpha=.15)
axs[0,0].legend(ncol=3,fontsize=8)
save(fig,'waveform_examples')

bad=[i for i,r in enumerate(T['records']) if r['label']!=r['predictions'][1]]
bad+=[i for i,r in enumerate(T['records']) if r['label']!=r['predictions'][2] and i not in bad]
fig,axs=plt.subplots(2,2,figsize=(9.2,6),layout='constrained')
for c,i in enumerate(bad):
    row=T['records'][i];context=W[f'context_{i}']
    for a in range(3):axs[0,c].plot(np.arange(len(context))/100-4,context[:,a],color=COLORS[a],lw=.85,label='XYZ'[a])
    axs[0,c].axvspan(-1,1,color='#2475a7',alpha=.09);axs[0,c].axvline(0,color='#777777',ls='--',lw=.8)
    axs[0,c].set(title=f"{row['id']} / {row['site']} / reference {LABELS[row['label']]}",xlabel='Time relative to reference center (s)',ylabel='Filtered acceleration (gal)')
    probs=np.array(row['probabilities'])
    for a in range(3):axs[1,c].plot([-.2,0,.2],probs[:,a],'o-',color=COLORS[a],label=LABELS[a])
    axs[1,c].set(xticks=[-.2,0,.2],xlabel='Window-center offset (s)',ylabel='Softmax probability',ylim=(-.03,1.03))
    axs[1,c].legend(ncol=3,loc='upper center',fontsize=8)
    for ax in axs[:,c]:ax.grid(alpha=.15)
axs[0,0].legend(ncol=3,fontsize=8)
save(fig,'error_cases')

fig,axs=plt.subplots(1,2,figsize=(9.1,3.7),layout='constrained')
freq=None
for label,col in enumerate(COLORS):
    power=[]
    for i,r in enumerate(T['records']):
        if r['label']==label:
            freq,p=welch(W[f'clip_{i}'],fs=40,nperseg=80,axis=0);power.append(p.sum(1))
    power=np.array(power);med=np.median(power,0);low,high=np.quantile(power,[.25,.75],axis=0)
    axs[0].plot(freq[1:],10*np.log10(med[1:]+1e-12),color=col,label=LABELS[label]);axs[0].fill_between(freq[1:],10*np.log10(low[1:]+1e-12),10*np.log10(high[1:]+1e-12),color=col,alpha=.12)
axs[0].set(xlabel='Frequency (Hz)',ylabel='Normalized PSD (dB/Hz)',title='Class median / interquartile range',xlim=(.5,20));axs[0].legend();axs[0].grid(alpha=.2)
i=indices[0];context=W[f'context_{i}'];freq,time,z=stft(np.linalg.norm(context,axis=1),fs=100,nperseg=64,noverlap=56,boundary=None)
sel=(freq>=.5)&(freq<=20);axs[1].pcolormesh(time-4,freq[sel],20*np.log10(np.abs(z[sel])+1e-5),shading='auto',cmap='magma');axs[1].set(xlabel='Time relative to P reference (s)',ylabel='Frequency (Hz)',title='Diagnostic STFT of vector magnitude')
save(fig,'spectral_diagnostics')

fig,ax=plt.subplots(figsize=(10,3.6),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,3.6);ax.axis('off')
boxes=[(.2,2.25,1.6,.85,'XYZ waveform\n100 Hz acquisition'),(2.2,2.25,2.15,.85,'QC, detrend, HP 0.5 Hz\ngravity alignment if valid'),(4.75,2.25,2,.85,'2 s cut, Fourier 40 Hz\njoint peak normalization'),(7.2,2.25,2.5,.85,'Input [1, 80, 3]\nRaw waveform, no spectrogram'),(.2,.5,2.35,.95,'Conv 16 / 32 / 64\nEach: BN, GELU, pool, dropout'),(3,.5,2.15,.95,'Time: 80 → 26 → 8 → 2\nFinal tensor: [64, 2, 3]'),(5.6,.5,1.85,.95,'Flatten 384\nFC 64 → 16 → 3'),(7.9,.5,1.8,.95,'Softmax\nP / S / Noise')]
for j,(x,y,w,h,label) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor='#eef3f8' if j<4 else '#eaf3ee',edgecolor='#75899a',lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
for row in [boxes[:4],boxes[4:]]:
    for a,b in zip(row[:-1],row[1:]):ax.add_patch(FancyArrowPatch((a[0]+a[2]+.05,a[1]+a[3]/2),(b[0]-.05,b[1]+b[3]/2),arrowstyle='->',mutation_scale=12,color='#586b7e'))
ax.plot([8.4,8.4,1.35,1.35],[2.15,1.85,1.85,1.65],color='#586b7e',lw=1)
ax.annotate('',xy=(1.35,1.52),xytext=(1.35,1.7),arrowprops={'arrowstyle':'->','color':'#586b7e'})
save(fig,'pipeline_architecture')

fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),layout='constrained')
for label,col in enumerate(COLORS):
    selected=[r for r in T['records'] if r['label']==label]
    vals=[r['probabilities'][1][label] for r in selected]
    xx=label+np.linspace(-.16,.16,len(vals))
    axs[0].scatter(xx,vals,color=col,s=23,alpha=.8)
    for x,r,v in zip(xx,selected,vals):
        if r['predictions'][1]!=r['label']:axs[0].scatter(x,v,facecolors='none',edgecolors='#a92a30',s=110,lw=1.5)
axs[0].set(xticks=[0,1,2],xticklabels=LABELS,ylabel='Probability assigned to reference class',ylim=(-.03,1.03),title='47 centered reference clips (not calibration)');axs[0].grid(alpha=.2)
errors=np.array([r['max_probability_error'] for r in E['cpp_parity']['records']]);axs[1].hist(np.log10(np.maximum(errors,1e-12)),bins=18,color=COLORS[0],edgecolor='white');axs[1].set(xlabel='log10(max absolute probability error)',ylabel='Number of windows',title='Native C++ vs Python: 141 evaluations');axs[1].axvline(np.log10(errors.max()),color=COLORS[1],ls='--',label=f'Max {errors.max():.2g}');axs[1].legend();axs[1].grid(alpha=.15)
save(fig,'confidence_and_native_parity')

# Machine-generated appendix tables: the same predictions used in every matrix.
lines=[]
for r in T['records']:
    preds=' & '.join(LABELS[x] for x in r['predictions'])
    lines.append(f"{r['id']} & {LABELS[r['label']]} & {r['day']} & {r['site']} & {preds} \\\\")
(BASE/'reference_rows.tex').write_text('\n'.join(lines)+'\n')
files=sorted(p.name for p in FIG.glob('*.pdf'))
print(f'Generated {len(files)} PDF figures: '+', '.join(files))
