"""Vector PDF waveform/inference comparisons and grouped-CV sensitivity panels."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'report-data'
FIG=BASE/'figures'
COLORS=['#2475a7','#d46a28','#32836b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.bbox':'tight','axes.axisbelow':True})

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',metadata={'Title':name.replace('_',' '),'Author':'SeismicTorch project','CreationDate':None,'ModDate':None})
    plt.close(fig)

def continuous():
    m=json.loads((DATA/'continuous-event.json').read_text())
    d=np.load(DATA/'continuous-event.npz')
    t=np.arange(m['npts'])/m['sample_rate']-m['origin_offset']
    raw=d['raw']-d['baseline'];prob=d['probabilities']
    for name,limits in [('waveform_inference_event',m['plot_excerpt_after_origin']),
                        ('waveform_inference_full_record',[t[0],t[-1]])]:
        fig,axs=plt.subplots(4,1,figsize=(9,8.5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,1,1,1.35]})
        use=(t>=limits[0])&(t<=limits[1])
        for a in range(3):
            axs[a].plot(t[use],raw[use,a],color=COLORS[a],lw=.65)
            axs[a].set_ylabel(f"{'XYZ'[a]} (gal)")
            axs[a].grid(alpha=.17)
        for a,label in enumerate(('P','S','Noise')):
            axs[3].plot(t[use],prob[use,a],color=COLORS[a],lw=1,label=label)
        for phase,col in [('P',COLORS[0]),('S',COLORS[1])]:
            center=m['reference_times_after_origin'][phase]
            for ax in axs:
                ax.axvline(center,color=col,ls='--',lw=1)
                ax.axvspan(center-1,center+1,color=col,alpha=.08)
            axs[0].annotate(f'{phase} reference {center:.2f} s',xy=(center,.98),xycoords=('data','axes fraction'),
                xytext=(-5 if phase=='P' else 5,-3),textcoords='offset points',va='top',ha='right' if phase=='P' else 'left',fontsize=9,
                bbox={'boxstyle':'round,pad=.15','facecolor':'white','alpha':.85,'edgecolor':'none'})
        axs[0].set_title(f"IES {m['station']} / IHP-003 / {m['origin_utc'][:19]} UTC",fontsize=12)
        axs[3].set(xlabel='Seconds relative to catalog origin',ylabel='Class probability',ylim=(-.04,1.04),xlim=limits)
        axs[3].legend(loc='upper right',ncol=3);axs[3].grid(alpha=.17)
        fig.suptitle('Acquired-axis waveform (fixed baseline removed for display) and dense CPU inference',fontsize=12)
        save(fig,name)

def sensitivity():
    if not (DATA/'sensitivity.json').exists():
        print('Sensitivity still running; waveform PDFs generated')
        return
    data=json.loads((DATA/'sensitivity.json').read_text())
    rows=data['summaries'];configs=data['protocol']['configurations'];epochs=data['protocol']['epoch_checkpoints']
    names={'scratch-conv2':'2 conv / random init','scratch-conv3':'3 conv / random init',
           'scratch-conv4':'4 conv / random init','cwa-conv3':'3 conv / CWA init'}
    fig,axs=plt.subplots(3,1,figsize=(8.8,7.5),layout='constrained')
    scratch=[c for c in configs if c['initialization']=='random']
    for a,metric in enumerate(('balanced_accuracy_mean','accuracy_mean','worst_offset_balanced_accuracy_mean')):
        values=np.array([[next(r for r in rows if r['configuration']==c['name'] and r['epoch']==e)[metric] for e in epochs] for c in scratch])
        lo=.7;hi=1
        mesh=axs[a].pcolormesh(np.arange(6),np.arange(4),values,cmap='Blues',vmin=lo,vmax=hi,edgecolors='white',linewidth=2)
        for i in range(3):
            for j in range(5):
                axs[a].text(j+.5,i+.5,f'{100*values[i,j]:.1f}',ha='center',va='center',fontsize=10,
                    color='white' if values[i,j]>.87 else '#17283c')
        axs[a].set(xticks=np.arange(5)+.5,xticklabels=epochs,yticks=np.arange(3)+.5,yticklabels=['2 conv','3 conv','4 conv'],
            xlabel='QSIS fine-tuning epochs',ylim=(3,0),title=['Centered balanced accuracy (%)','Centered accuracy (%)','Worst-offset balanced accuracy (%)'][a])
        axs[a].tick_params(length=0)
    fig.suptitle('Depth sensitivity: 5 date-grouped folds, 3 paired seeds; random initialization for all depths',fontsize=12)
    save(fig,'sensitivity_depth')

    fig,grid=plt.subplots(2,2,figsize=(9.5,6.4),layout='constrained');axs=grid.ravel()
    for name,col in [('scratch-conv3',COLORS[0]),('cwa-conv3',COLORS[1])]:
        selected=[next(r for r in rows if r['configuration']==name and r['epoch']==e) for e in epochs]
        mean=np.array([r['balanced_accuracy_mean'] for r in selected]);sd=np.array([r['balanced_accuracy_sd'] for r in selected])
        axs[0].errorbar(epochs,100*mean,yerr=100*sd,fmt='o-',capsize=3,color=col,label=names[name])
        axs[1].plot(epochs,[100*r['clean_train_accuracy_mean'] for r in selected],ls='--',color=col,label=names[name]+' training')
        axs[1].plot(epochs,[100*r['accuracy_mean'] for r in selected],'o-',color=col,label=names[name]+' OOF')
        axs[2].plot(epochs,[100*r['worst_offset_balanced_accuracy_mean'] for r in selected],'o-',color=col,label=names[name])
    axs[0].set(title='Centered balanced accuracy',ylabel='OOF balanced accuracy (%)')
    axs[1].set(title='Generalization gap',ylabel='Accuracy (%)')
    axs[2].set(title='Timing robustness',ylabel='Worst-offset balanced accuracy (%)')
    for ax in axs:
        ax.set(xlabel='QSIS fine-tuning epochs',xscale='log',xticks=epochs,xticklabels=epochs)
        ax.grid(alpha=.2);ax.minorticks_off()
    axs[0].legend(fontsize=9);axs[1].legend(fontsize=8.5,loc='lower right')
    cwa=[next(r for r in rows if r['configuration']=='cwa-conv3' and r['epoch']==e) for e in epochs]
    for k,(label,col) in enumerate(zip(['P','S','Noise'],COLORS,strict=True)):
        axs[3].plot(epochs,[100*np.mean([seed['offsets'][1]['recall'][k] for seed in r['seeds']]) for r in cwa],'o-',color=col,label=label)
    axs[3].set(title='CWA start: centered class recall',ylabel='OOF recall (%)');axs[3].legend(ncol=3,fontsize=9)
    fig.suptitle('Epoch sensitivity of the original 3+3 CNN; error bars are seed SD, not confidence intervals',fontsize=12)
    save(fig,'sensitivity_epochs')

    fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained')
    for cfg,col in zip(configs,[*COLORS,'#8664a1'],strict=True):
        selected=[next(r for r in rows if r['configuration']==cfg['name'] and r['epoch']==e) for e in epochs]
        axs[0].plot(epochs,[100*r['balanced_accuracy_mean'] for r in selected],'o-',color=col,label=names[cfg['name']])
        axs[1].plot([r['parameters']/1000 for r in selected],[100*r['balanced_accuracy_mean'] for r in selected],'o',color=col)
        first,last=selected[0],selected[-1]
        axs[1].annotate(f"{cfg['depth']} conv / "+('CWA' if cfg['initialization']=='CWA' else 'random'),
            (last['parameters']/1000,100*last['balanced_accuracy_mean']),xytext={'scratch-conv2':(-15,-17),'scratch-conv3':(5,-28),'scratch-conv4':(-90,8),'cwa-conv3':(5,10)}[cfg['name']],textcoords='offset points',fontsize=9)
    axs[0].set(xscale='log',xticks=epochs,xticklabels=epochs,xlabel='QSIS fine-tuning epochs',ylabel='OOF balanced accuracy (%)',title='All matched development trajectories');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Trainable parameters (thousands)',ylabel='OOF balanced accuracy (%)',title='Capacity versus validation performance')
    for ax in axs:ax.grid(alpha=.2);ax.minorticks_off()
    save(fig,'sensitivity_capacity')

    lines=[]
    for cfg in configs:
        for e in epochs:
            r=next(r for r in rows if r['configuration']==cfg['name'] and r['epoch']==e)
            lines.append(f"{cfg['depth']} & {'CWA' if cfg['initialization']=='CWA' else 'Random'} & {e} & {r['parameters']:,} & "
                f"{100*r['accuracy_mean']:.2f} & {100*r['balanced_accuracy_mean']:.2f} $\\pm$ {100*r['balanced_accuracy_sd']:.2f} & "
                f"{100*r['worst_offset_balanced_accuracy_mean']:.2f} \\\\")
    (BASE/'sensitivity_rows.tex').write_text('\n'.join(lines)+'\n')
    print('Generated all continuous and sensitivity PDFs')

if __name__=='__main__':continuous();sensitivity()
