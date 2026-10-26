"""Regenerate monochrome profiles from frozen observations without changing costs."""
from pathlib import Path
import sys, json, hashlib, shutil, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import argparse
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'generated' / 'profiles')
args = parser.parse_args()
BENCH = ROOT / 'frozen'
METHODS = ['UAD','NAG','RNAG','AC-FGM','GD','CUG','AdaNAG','A-QNPE']
# Distinct dash patterns and sparse markers remain identifiable in print.
MONO_STYLES = {
    'UAD': ('-', None),
    'NAG': ((0, (4, 2)), '^'),
    'RNAG': ((0, (5, 2, 1, 2)), 's'),
    'AC-FGM': ((0, (1, 2)), 'o'),
    'GD': ((0, (7, 3)), 'D'),
    'AdaNAG': ((0, (4, 2, 1, 2, 1, 2)), 'v'),
    'CUG': ((0, (2, 2)), 'x'),
    'A-QNPE': ('-', '+'),
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def label(m,suite):
    if m=='UAD': return 'UAG'
    if m=='AdaNAG': return 'AdaNAG (Alg. 1)'
    return 'A-QNPE*' if m=='A-QNPE' and suite=='cutest' else m

rows=json.loads((BENCH/'results.json').read_text())
protocol=json.loads((BENCH/'protocol.json').read_text())
analysis=json.loads((BENCH/'analysis.json').read_text())
frozen={p.name:sha(p) for p in [BENCH/'results.json',BENCH/'protocol.json',BENCH/'analysis.json',BENCH/'results.csv']}
assert frozen['results.json']=='074fdffe86254e4bf485f9bfd681b34cb8dd7b7997cd5dd165994cf7a82c6f5a'
out=args.output.resolve()
if BENCH.resolve() == out or BENCH.resolve() in out.parents:
    raise SystemExit('Refusing to write inside frozen observations.')
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'medium','pdf.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white',
    'axes.linewidth':1.5,'xtick.major.width':1.5,'ytick.major.width':1.5})
profile_checks=[]
for suite in ['specific','cutest']:
    sr=[r for r in rows if r['Suite']==suite]
    problems=list(dict.fromkeys(r['Problem'] for r in sr))
    methods=[m for m in METHODS if any(r['Method']==m for r in sr)]
    lookup={(r['Problem'],r['Method']):r for r in sr}
    stats=analysis['summary'][suite]['methods']
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.2),sharey=True)
    for ax,metric in zip(axes,['FG','Seconds']):
        costs=np.array([[float(lookup[p,m][metric]) if lookup[p,m]['Success'] else np.inf for m in methods] for p in problems])
        best=np.min(costs,axis=1)
        ratio=np.full_like(costs,np.inf)
        valid=np.isfinite(best)
        ratio[valid]=costs[valid]/best[valid,None]
        finite=ratio[np.isfinite(ratio)]
        xmax=2.**math.ceil(math.log2(max(8.,float(finite.max()) if len(finite) else 8.)))
        for j,m in enumerate(methods):
            for i,p in enumerate(problems):
                old=analysis['summary'][suite]['ratios'][metric][p][m]
                assert (old is None and not np.isfinite(ratio[i,j])) or old==ratio[i,j]
            values=np.sort(ratio[np.isfinite(ratio[:,j]),j])
            x=np.r_[1.,np.unique(values),xmax]
            y=np.array([np.sum(ratio[:,j]<=t*(1.+1e-12))/len(problems) for t in x])
            style,marker=MONO_STYLES[m]
            interior=np.flatnonzero((x>1.) & (x<xmax))
            marker_indices=interior[np.unique(np.linspace(0,len(interior)-1,min(6,len(interior))).round().astype(int))].tolist() if len(interior) else []
            ax.step(x,y,where='post',color='black',linestyle=style,
                marker=marker,markevery=marker_indices,markersize=4.8,
                markerfacecolor='white',markeredgewidth=1.5,
                linewidth=2.8 if m=='UAD' else 1.6,zorder=10 if m=='UAD' else 2,
                label=f'{label(m,suite)} ({stats[m]["solved"]}/{len(problems)})')
            profile_checks.append({'suite':suite,'metric':metric,'method':m,'ratios_equal':True})
        ax.set_xscale('log',base=2);ax.set_xlim(1.,xmax);ax.set_ylim(0.,1.025)
        ax.set_xlabel('FG Oracle Calls / best' if metric=='FG' else 'Single-run time / best')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.grid(axis='y',color='.88',linewidth=1.5);ax.set_title('FG performance profile' if metric=='FG' else 'Time performance profile')
    axes[0].set_ylabel('Fraction of problems')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),fontsize=9.0,ncol=4,frameon=False,handlelength=3.0)
    fig.suptitle('Nine concrete problems' if suite=='specific' else 'CUTEst: 55 fixed problems',y=.99,fontsize=13)
    note='One observation per method and problem. Unsuccessful runs remain in the denominator.'
    if suite=='cutest':note+='  * A-QNPE uses the online-scale extension.'
    # Protocol and extension notes are stated in the manuscript.
    fig.subplots_adjust(left=.105,right=.99,bottom=.34,top=.81,wspace=.09)
    fig.savefig(out/(suite+'_profiles.pdf'),dpi=190)
    plt.close(fig)


assert {name:sha(BENCH/name) for name in frozen}==frozen
(root_audit:=out/'profile_layout_audit.json').write_text(json.dumps({'frozen_data_sha256':frozen,'profile_checks':profile_checks,'new_optimization_runs':0,'figsize_inches':[7.4,3.2],'legend_fontsize':9.0,'figure_sha256':{p.name:sha(p) for p in out.glob('*profiles.pdf')}},indent=2),encoding='utf-8')
print('Two profiles regenerated; all original cost ratios preserved.')
