"""Analyze complete new repetitions using the preregistered R/R success rule."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE=Path(__file__).resolve().parent
METHODS=['UAD','NAG','RNAG','AC-FGM','GD','CUG','AdaNAG','A-QNPE']
STYLES={'UAD':('-',None),'NAG':((0,(4,2)),'^'),'RNAG':((0,(5,2,1,2)),'s'),
'AC-FGM':((0,(1,2)),'o'),'GD':((0,(7,3)),'D'),'AdaNAG':((0,(4,2,1,2,1,2)),'v'),
'CUG':((0,(2,2)),'x'),'A-QNPE':('-','+')}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    return x
def save(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def finite(x):return float(x) if np.isfinite(x) else None
def label(m,s):return 'UAG' if m=='UAD' else 'AdaNAG (Alg. 1)' if m=='AdaNAG' else 'A-QNPE*' if m=='A-QNPE' and s=='cutest' else m

def draw_profile(records,suite,out,suffix,extra):
    methods=[m for m in METHODS if any(r['Method']==m for r in records)]
    problems=list(dict.fromkeys(r['Problem'] for r in records))
    lookup={(r['Problem'],r['Method']):r for r in records}
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.2),sharey=True)
    ratios={}
    for ax,metric in zip(axes,['FG','Seconds']):
        costs=np.array([[lookup[p,m][metric] if lookup[p,m]['Success'] else np.inf for m in methods] for p in problems])
        best=costs.min(axis=1);r=np.full_like(costs,np.inf);valid=np.isfinite(best)
        r[valid]=costs[valid]/best[valid,None]
        values=r[np.isfinite(r)];xmax=2.**math.ceil(math.log2(max(8.,float(values.max()) if len(values) else 8.)))
        ratios[metric]={p:{m:finite(r[i,j]) for j,m in enumerate(methods)} for i,p in enumerate(problems)}
        for j,m in enumerate(methods):
            x=np.r_[1.,np.unique(r[np.isfinite(r[:,j]),j]),xmax]
            y=np.array([np.mean(r[:,j]<=t*(1+1e-12)) for t in x])
            interior=np.flatnonzero((x>1.)&(x<xmax))
            marks=interior[np.unique(np.linspace(0,len(interior)-1,min(6,len(interior))).round().astype(int))].tolist() if len(interior) else []
            style,mark=STYLES[m];solved=sum(lookup[p,m]['Success'] for p in problems)
            ax.step(x,y,where='post',color='black',linestyle=style,marker=mark,markevery=marks,
                markersize=4.8,markerfacecolor='white',markeredgewidth=1.5,
                linewidth=2.8 if m=='UAD' else 1.6,zorder=10 if m=='UAD' else 2,
                label=f'{label(m,suite)} ({solved}/{len(problems)})')
        ax.set_xscale('log',base=2);ax.set_xlim(1.,xmax);ax.set_ylim(0.,1.025)
        ax.set_xlabel('FG Oracle Calls / best' if metric=='FG' else 'Median time / best' if not suffix else 'Single-run time / best')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.grid(axis='y',color='.88',linewidth=1.5)
        ax.set_title('FG performance profile' if metric=='FG' else 'Time performance profile')
    axes[0].set_ylabel('Fraction of problems' if not suffix else 'Fraction of problem-repetitions')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.01),fontsize=9,ncol=4,frameon=False,handlelength=3.)
    title='Nine concrete problems' if suite=='specific' else 'CUTEst: 55 fixed problems'
    if suffix:title+=' (per repetition)'
    fig.suptitle(title,y=.99,fontsize=13)
    fig.subplots_adjust(left=.105,right=.99,bottom=.34,top=.81,wspace=.09)
    path=out/(suite+'_profiles'+suffix+'.pdf');fig.savefig(path,dpi=190);plt.close(fig)
    return ratios

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();source=a.campaign.resolve();out=(a.output or source/'analysis').resolve()
    protected=(HERE/'frozen').resolve()
    if out==protected or protected in out.parents:raise ValueError('Frozen observations are read-only')
    out.mkdir(parents=True,exist_ok=True)
    run_protocol=read(source/'protocol.json')
    if run_protocol.get('diagnostic', True):
        raise ValueError('Refusing to treat diagnostic or unlabelled observations as a full manuscript campaign')
    rows=read(source/'results.json');protocol=read(HERE/'frozen/protocol.json')
    keys={(suite,c['name'],m) for suite in ['specific','cutest'] for c in protocol[suite] for m in protocol['methods_'+suite]}
    repeats=sorted(set(r['Repeat'] for r in rows));complete=[];partial=[]
    for repeat in repeats:
        rr=[r for r in rows if r['Repeat']==repeat]
        seen={(r['Suite'],r['Problem'],r['Method']) for r in rr}
        if len(rr)==len(seen)==len(keys) and seen==keys:complete.append(repeat)
        else:partial.append(dict(repeat=repeat,observations=len(rr),expected=len(keys)))
    if not complete:raise ValueError('No complete repetitions across all 457 combinations')
    R=len(complete);used=[r for r in rows if r['Repeat'] in complete]
    lookup={(r['Repeat'],r['Suite'],r['Problem'],r['Method']):r for r in used}
    aggregated=[];detail=[];summary={};sensitivity={}
    for suite in ['specific','cutest']:
        for c in protocol[suite]:
            for m in protocol['methods_'+suite]:
                group=[lookup[r,suite,c['name'],m] for r in complete]
                successful=sum(bool(r['Success']) for r in group)
                d=dict(Suite=suite,Problem=c['name'],Method=m,Repetitions=R,Successes=successful,Success=successful==R,
                    StatusCounts=dict(Counter(r['Status'] for r in group)))
                for metric in ['FG','Seconds']:
                    v=np.array([r[metric] for r in group],float)
                    d[metric]=float(np.median(v));d[metric+'_q25']=float(np.quantile(v,.25));d[metric+'_q75']=float(np.quantile(v,.75));d[metric+'_min']=float(v.min());d[metric+'_max']=float(v.max())
                d['FG_distinct']=sorted(set(int(r['FG']) for r in group))
                d['FG_non_time_limit_distinct']=sorted(set(int(r['FG']) for r in group if r['Status']!='time_limit'))
                detail.append(d);aggregated.append({k:d[k] for k in ['Suite','Problem','Method','FG','Seconds','Success']})
        ss=[d for d in detail if d['Suite']==suite]
        stats={m:dict(solved=sum(d['Success'] for d in ss if d['Method']==m),cases=len(protocol[suite]),
            sum_per_problem_median_FG=sum(d['FG'] for d in ss if d['Method']==m),
            sum_per_problem_median_Seconds=sum(d['Seconds'] for d in ss if d['Method']==m),
            all_observation_statuses=dict(Counter(r['Status'] for r in used if r['Suite']==suite and r['Method']==m))) for m in protocol['methods_'+suite]}
        common={};lm={(d['Problem'],d['Method']):d for d in ss}
        for m in protocol['methods_'+suite]:
            if m=='UAD':continue
            both=[c['name'] for c in protocol[suite] if lm[c['name'],'UAD']['Success'] and lm[c['name'],m]['Success']]
            common[m]=dict(common_successes=len(both),UAG_only_successes=sum(lm[c['name'],'UAD']['Success'] and not lm[c['name'],m]['Success'] for c in protocol[suite]),comparator_only_successes=sum(not lm[c['name'],'UAD']['Success'] and lm[c['name'],m]['Success'] for c in protocol[suite]))
            for metric in ['FG','Seconds']:
                rr=[lm[name,'UAD'][metric]/lm[name,m][metric] for name in both]
                common[m][metric+'_ratio_geomean']=float(np.exp(np.log(rr).mean())) if rr else None
                common[m][metric+'_UAG_wins']=sum(v<1 for v in rr)
        uag=[d for d in ss if d['Method']=='UAD']
        relative_iqr=[(d['Seconds_q75']-d['Seconds_q25'])/d['Seconds'] for d in uag if d['Seconds']>0]
        summary[suite]=dict(methods=stats,comparisons=common,UAG_time_relative_IQR=dict(min=min(relative_iqr),median=float(np.median(relative_iqr)),max=max(relative_iqr)),total_cost_scope='Sum of per-problem median observed costs, including unsuccessful runs; not the median of suite totals.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'medium','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white','axes.linewidth':1.5,'xtick.major.width':1.5,'ytick.major.width':1.5})
    for suite in ['specific','cutest']:
        summary[suite]['ratios']=draw_profile([r for r in aggregated if r['Suite']==suite],suite,out,'',R)
        for method in protocol['methods_'+suite]:
            for metric in ['FG','Seconds']:
                summary[suite]['methods'][method][metric+'_best_cases']=sum(v[method] is not None and v[method]<=1+1e-12 for v in summary[suite]['ratios'][metric].values())
        sr=[dict(r,Problem=f'{r["Problem"]}__r{r["Repeat"]}') for r in used if r['Suite']==suite]
        sensitivity[suite]=draw_profile(sr,suite,out,'_by_repetition',R)
    save(out/'aggregated_results.json',aggregated);save(out/'repetition_details.json',detail)
    with (out/'repetition_details.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    allstatuses=dict(Counter(r['Status'] for r in used));vary=[d for d in detail if len(d['FG_distinct'])>1]
    vary_without_time_limits=[d for d in detail if len(d['FG_non_time_limit_distinct'])>1]
    set_serialization=json.dumps(sorted(keys),ensure_ascii=True,separators=(',',':')).encode('ascii')
    set_hash=hashlib.sha256(set_serialization).hexdigest()
    trace_audit=read(source/'trace_audit.json') if (source/'trace_audit.json').exists() else None
    if trace_audit is not None:
        assert trace_audit['passed'] and trace_audit['results_sha256']==hashlib.sha256((source/'results.json').read_bytes()).hexdigest()
    report=dict(complete_repetitions=complete,R=R,partial_repetitions_excluded=partial,used_observations=len(used),
        success_rule='All R/R observations must succeed; otherwise cost is infinite.',cost_rule='Median of all R costs for solved combinations; IQR uses numpy.quantile linear interpolation.',
        total_solver_seconds=sum(r['Seconds'] for r in used),all_statuses=allstatuses,combinations_with_FG_variation=len(vary),
        combinations_with_FG_variation_after_excluding_time_limits=len(vary_without_time_limits),
        expected_case_method_set_sha256=set_hash,case_method_set_hash_format='sha256 of sorted [suite,problem,method] JSON arrays, ASCII, separators comma/colon',
        first_target_trace_audit=trace_audit,summary=summary)
    save(out/'analysis.json',report);save(out/'sensitivity_ratios.json',sensitivity);save(out/'FG_variation.json',vary);save(out/'FG_variation_excluding_time_limits.json',vary_without_time_limits)
    save(out/'unsuccessful_runs.json',[r for r in used if not r['Success']])
    save(out/'input_sha256.json',{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ['results.json','protocol.json']})
    print(json.dumps({k:report[k] for k in ['R','complete_repetitions','used_observations','total_solver_seconds','all_statuses','combinations_with_FG_variation']},indent=2))

if __name__=='__main__':main()
