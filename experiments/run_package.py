"""Run the importable UAG package on selected paper problems; new observations only."""
import argparse
import json
import importlib.util
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(HERE/'reference'))
import environment
import numpy as np
from threadpoolctl import threadpool_limits
from uag import minimize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['specific','cutest'],default='specific')
    p.add_argument('--problems',nargs='+')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-evals',type=int)
    p.add_argument('--time-limit',type=float)
    p.add_argument('--backend',choices=['python','numba'],default='numba')
    a=p.parse_args();out=a.output.resolve()
    if a.problems is None:a.problems=['rosenbrock_d200'] if a.suite=='specific' else ['ARWHEAD']
    if a.max_evals is not None and a.max_evals<1:p.error('--max-evals must be positive')
    if a.time_limit is not None and a.time_limit<=0:p.error('--time-limit must be positive')
    if out==HERE.resolve() or HERE.resolve() in out.parents and out.parts[len(HERE.parts)] in ('frozen','reference','cases','manifests','third_party'):p.error('Protected output directory')
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'frozen/protocol.json').read_text());cfg=protocol['config'][a.suite]
    entries={e['name']:e for e in protocol[a.suite]}
    if set(a.problems)-set(entries):p.error('Unknown problems for selected suite')
    rows=[]
    with threadpool_limits(limits=1):
        if a.suite=='specific':
            from formal_cases import load_cases
            cases=load_cases(a.problems)
        else:
            cases=[]
            for name in a.problems:
                path=HERE/'cases/cutest_generated'/('cutest_native_'+name+'.py')
                spec=importlib.util.spec_from_file_location('cutest_native_'+name,path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
                cases.append(dict(name=name,x0=mod.x0,native_fg=mod.fg,independent=mod.fg))
        for c in cases:
            _,g=c['native_fg'](c['x0']);initial=float(np.linalg.norm(g))
            eps=cfg.get('eps',cfg.get('relative_eps',0)*max(1.,initial))
            result=minimize(c['native_fg'],c['x0'],tol=eps,max_evals=a.max_evals or cfg['max_fg'],time_limit=a.time_limit or cfg['seconds'],backend=a.backend,record_history=True)
            f,g=c['independent'](result.x);verified=float(np.linalg.norm(g))
            success=bool(result.success and verified<=eps*(1+1e-7))
            np.savez_compressed(out/(c['name']+'.npz'),history=result.history,x=result.x,counts=result.work.counts)
            rows.append(dict(problem=c['name'],method='UAG-package',backend=a.backend,success=success,status=result.status,message=result.message,FG=result.nfev,seconds=result.elapsed_seconds,setup_seconds=result.setup_seconds,gradient=result.grad_norm,verified_gradient=verified,tolerance=eps,external_initial_FG=1,external_endpoint_FG=1))
            (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(rows[-1]),flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(source='importable UAG package; separate from historical timing campaign',suite=a.suite,problems=a.problems,config=cfg,max_evals_override=a.max_evals,time_limit_override=a.time_limit,numerical_threads=1),indent=2)+'\n')

if __name__=='__main__':main()
