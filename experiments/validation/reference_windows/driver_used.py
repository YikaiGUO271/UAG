"""Run new complete repetitions without modifying any saved campaign.

By default use the portable reference benchmark implementation in this repo.
--historical-root explicitly selects a separately retained original campaign;
its runner, timers and native evaluators remain unchanged.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, value):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(value,indent=2,default=serial)+'\n',encoding='utf-8');tmp.replace(p)
def serial(value):
    if hasattr(value,'tolist'): return value.tolist()
    raise TypeError(type(value).__name__)
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--historical-root',type=Path)
    p.add_argument('--clock',choices=['portable','windows-native'],default='portable',help='Use the portable clock or exact historical Windows QPC clock')
    p.add_argument('--stop-before-utc', help='Stop before starting the next observation at this ISO UTC time')
    p.add_argument('--suite',choices=['specific','cutest','all'],default='all')
    p.add_argument('--problems',nargs='+')
    p.add_argument('--methods',nargs='+')
    p.add_argument('--max-fg',type=int,help='Diagnostic override; not the manuscript campaign')
    p.add_argument('--seconds',type=float,help='Diagnostic override; not the manuscript campaign')
    a=p.parse_args()
    deadline=datetime.fromisoformat(a.stop_before_utc) if a.stop_before_utc else None
    if deadline is not None and deadline.tzinfo is None: p.error('Deadline requires UTC offset')
    if a.repeats<1: p.error('--repeats must be positive')
    if a.max_fg is not None and a.max_fg<1:p.error('--max-fg must be positive')
    if a.seconds is not None and a.seconds<=0:p.error('--seconds must be positive')
    if a.clock=='windows-native' and sys.platform!='win32':p.error('The Windows native clock requires Windows')
    if a.historical_root and a.clock!='portable':p.error('--historical-root already selects its original clock')
    out=a.output.resolve()
    protected=[(HERE/d).resolve() for d in ('frozen','reference','cases','manifests','third_party')]
    if any(out==d or d in out.parents for d in protected): p.error('Refusing protected output directory')
    if a.historical_root:
        old=a.historical_root.resolve()
        if out==old or old in out.parents:p.error('Output must be outside the historical directory')
    out.mkdir(parents=True,exist_ok=False)
    (out/'driver_used.py').write_bytes(Path(__file__).read_bytes())
    ref=(a.historical_root.resolve() if a.historical_root else HERE/'reference')
    sys.path.insert(0,str(ref))
    import environment
    import numpy as np
    import numba
    from threadpoolctl import threadpool_limits,threadpool_info
    if not a.historical_root and a.clock=='windows-native':
        import_file('native_api',ref/'native_api_windows.py')
    R=import_file('_uag_repetition_reference',ref/('run_campaign.py' if a.historical_root else 'runner.py'))
    from adanag_theory import make_adanag_theory_solver
    R.ADANAG=make_adanag_theory_solver(R.API.query)
    protocol=load(HERE/'frozen/protocol.json')
    source_files=list(ref.glob('*.py'))+list((ref/'optimized_uad').glob('*.py'))+list((ref/'aqnpe').glob('*.py'))
    source_hashes={str(f.relative_to(ref)):digest(f) for f in sorted(source_files)}
    frozen_before=load(HERE/'frozen/SHA256.json')
    if a.historical_root:
        assert digest(ref/'protocol.json')==frozen_before['protocol.json']
        for f,h in protocol['source_sha256'].items(): assert digest(environment.ROOT/f)==h,f
    methods=a.methods or list(R.METHODS)
    if set(methods)-set(R.METHODS):p.error('Unknown reference methods')
    started=datetime.now(timezone.utc).isoformat()
    effective=dict(source_protocol_sha256=frozen_before['protocol.json'],repeats=a.repeats,
        suites=a.suite,problems=a.problems,methods=methods,config=protocol['config'],
        max_fg_override=a.max_fg,seconds_override=a.seconds,diagnostic=bool(a.max_fg or a.seconds or a.problems or a.methods or a.suite!='all'),
        numerical_threads=1,solver_parallelism=1,source_hashes=source_hashes,
        reference_runtime='original unchanged Windows/native campaign' if a.historical_root else 'repository reference; exact historical Windows QPC clock' if a.clock=='windows-native' else 'portable reference; monotonic Python clock bridge',
        clock='original historical' if a.historical_root else a.clock,
        method_order='For zero-based repetition r and within-suite case index i rotate methods by (i+r) modulo m, reverse when (i+r) is odd.',
        timing=protocol['timing'].replace('One observation;', 'One observation per repetition;'),driver_sha256=digest(__file__),created_utc=started,stop_before_utc=a.stop_before_utc,
        aggregation='Use only complete repetitions shared by every method/problem. Solved requires R/R successes; medians of all R costs are used only for solved entries, otherwise infinity. Report all statuses and IQR; also report problem-by-repeat profiles as sensitivity analysis.')
    dump(out/'protocol.json',effective)
    rows=[];setup=[];warm=[];stopped=False
    with threadpool_limits(limits=1):
        t=time.perf_counter()
        case=dict(x0=np.array([2.,-1.,.5,-2.]),native_fg=R.validation_quadratic,L1=4.,L2=1.)
        for method in methods:
            row,*_=R.measure(method,case,1e-12 if method=='CUG' else 0.,16,np.inf,False)
            if row['Error']:raise RuntimeError(row)
            warm.append(dict(method=method,FG=row['FG'],status=row['Status']))
        if 'A-QNPE' in methods:
            row,*_=R.measure('A-QNPE',dict(case,L1=None),0.,16,np.inf,False)
            if row['Error']:raise RuntimeError(row)
            warm.append(dict(method='A-QNPE-online-scale',FG=row['FG'],status=row['Status']))
        dump(out/'warmup.json',dict(seconds=time.perf_counter()-t,total_FG=sum(r['FG'] for r in warm),records=warm))
        from formal_cases import load_cases
        cases=[]
        if a.suite in ('all','specific'):
            names=[c['name'] for c in protocol['specific'] if not a.problems or c['name'] in a.problems]
            for c in load_cases(names):cases.append(('specific',c))
        if a.suite in ('all','cutest'):
            for entry in protocol['cutest']:
                if a.problems and entry['name'] not in a.problems:continue
                t=time.perf_counter()
                if a.historical_root:c=R.load_cutest(entry)
                else:
                    mod=import_file('cutest_native_'+entry['name'],HERE/'cases/cutest_generated'/('cutest_native_'+entry['name']+'.py'))
                    c=dict(name=entry['name'],family='CUTEst',x0=np.ascontiguousarray(mod.x0).copy(),native_fg=mod.fg,independent=mod.fg,L1=None,L2=None)
                setup.append(dict(suite='cutest',problem=entry['name'],construction_seconds=time.perf_counter()-t,FG=1 if a.historical_root else 0))
                cases.append(('cutest',c))
        if not cases:raise ValueError('No selected problems')
        if a.problems and set(a.problems)-{c['name'] for _,c in cases}:raise ValueError('Unknown selected problems')
        for suite,c in cases:
            t=time.perf_counter();f,g=c['native_fg'](c['x0']);c['initial_gradient']=float(np.linalg.norm(g))
            setup.append(dict(suite=suite,problem=c['name'],FG=1,seconds=time.perf_counter()-t))
        dump(out/'setup_cost.json',setup)
        dump(out/'environment.json',dict(python=sys.version,numpy=np.__version__,numba=numba.__version__,platform=platform.platform(),threadpools=threadpool_info(),numerical_threads=1))
        total=a.repeats*sum(len([m for m in methods if not (suite=='cutest' and m=='CUG')]) for suite,_ in cases)
        if total==0:raise ValueError('No selected method is applicable to the selected suite')
        for repeat in range(a.repeats):
            if stopped: break
            for suite in ('specific','cutest'):
                if stopped: break
                suite_cases=[c for s,c in cases if s==suite]
                for index,c in enumerate(suite_cases):
                    if stopped: break
                    selected=[m for m in methods if not (suite=='cutest' and m=='CUG')]
                    if not selected:continue
                    shift=(index+repeat)%len(selected);order=selected[shift:]+selected[:shift]
                    if (index+repeat)%2:order.reverse()
                    cfg=protocol['config'][suite];eps=cfg.get('eps',cfg.get('relative_eps',0)*max(1.,c['initial_gradient']))
                    fg=a.max_fg or cfg['max_fg'];seconds=a.seconds or cfg['seconds']
                    for method in order:
                        if deadline is not None and datetime.now(timezone.utc) >= deadline:
                            stopped=True;break
                        stem=f'r{repeat+1:02d}__{suite}__{c["name"]}__{method}'.replace('-','_')
                        trace_dir=out/'traces';trace_dir.mkdir(exist_ok=True)
                        marker=out/(stem+'.started.json');dump(marker,dict(started_utc=datetime.now(timezone.utc).isoformat()))
                        row,history,x,counts,events=R.measure(method,c,float(eps),int(fg),float(seconds))
                        trace=trace_dir/(stem+'.npz');np.savez_compressed(trace,history=history,x=x,counts=counts,events=events)
                        row.update(Repeat=repeat+1,Suite=suite,Problem=c['name'],Family=c['family'],Dimension=len(c['x0']),InitialGradient=c['initial_gradient'],Trace=str(trace.relative_to(out)))
                        rows.append(row);dump(out/'results.json',rows)
                        with (out/'results.csv').open('w',newline='',encoding='utf-8') as stream:
                            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                        dump(marker,dict(completed=True,status=row['Status'],FG=row['FG']))
                        print(json.dumps(dict(completed=len(rows),total=total,repeat=repeat+1,suite=suite,problem=c['name'],method=method,seconds=row['Seconds'],FG=row['FG'],status=row['Status'])),flush=True)
        if not stopped: assert len(rows)==total
    assert source_hashes=={str(f.relative_to(ref)):digest(f) for f in sorted(source_files)}
    for f,h in frozen_before.items():assert digest(HERE/'frozen'/f)==h
    per_repeat=total//a.repeats
    completed_repeats=[r for r in range(1,a.repeats+1) if sum(row['Repeat']==r for row in rows)==per_repeat]
    dump(out/'complete.json',dict(completed=not stopped,deadline_stopped=stopped,completed_repeats=completed_repeats,observations=len(rows),started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),source_hashes_unchanged=True,frozen_files_unchanged=True))

if __name__=='__main__':main()
