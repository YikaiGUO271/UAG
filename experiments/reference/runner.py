"""Historical solver recurrences on the common portable FG meter.

Only the platform clock differs from the 2026-10-24 measurement runtime.
Do not compare the new timing observations with the frozen table as repetitions.
"""
import gc
import json
import time
import numpy as np
from numba import njit
import native_api as API
import baselines_native as BL
from cug_native import cug_native
from aqnpe.native import make_native_solver
from adanag_theory import make_adanag_theory_solver
from optimized_uad import core
from optimized_uad.secant import PrivateState
METHODS = ('UAD','NAG','RNAG','AC-FGM','GD','CUG','AdaNAG','A-QNPE')
STATUSES = {0:'unexpected_end',1:'stationarity',2:'query_budget',3:'nonfinite_oracle',4:'time_limit',5:'nonfinite_iterate',6:'oracle_error',7:'numerical_failure',8:'nonpositive_convexity_residual'}
AQNPE = make_native_solver(API.query)
ADANAG = make_adanag_theory_solver(API.query)
def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)
def emit(value): print(json.dumps(value,default=serial),flush=True)
def configure(fg,n,eps,maxfg):
    meter=API.Meter(fg,maxfg,eps)
    meter.history=np.empty((maxfg,4),dtype=np.float64)
    run=API.Runtime(1.,eps,maxfg,n,meter)
    run.events=np.zeros(128,dtype=np.int64)
    return meter,run,PrivateState(n),core.Journal(0)


def invoke(method,case,run,private,journal):
    x=case['x0'].copy()
    if method=='UAD': return core.solve(x,run,private,journal,True)
    if method in ('NAG','RNAG','GD'):
        return BL.nag_native(x,run,{'NAG':2,'RNAG':0,'GD':3}[method],True)
    if method=='AC-FGM': return BL.acfgm_native(x,run,1.)
    if method=='CUG': return cug_native(x,run,float(case['L1']),float(case['L2']))
    if method=='AdaNAG': return ADANAG(x,run,case['initial_offset'])
    if method=='A-QNPE':
        return AQNPE(x,run,float(case.get('L1') or 1.),0,case.get('L1') is None)
    raise KeyError(method)


def measure(method,case,eps,maxfg,seconds,verify=True):
    meter,run,private,journal=configure(case['native_fg'],len(case['x0']),eps,maxfg)
    case=dict(case,initial_offset=np.random.default_rng(0).uniform(0.,1.,len(case['x0'])))
    gc.collect()
    gc.disable()
    error=''; result=None
    meter.started=API.now()
    meter.deadline=meter.started+seconds
    try:
        result=invoke(method,case,run,private,journal)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if not run.status: run.status=7
    finally:
        elapsed=API.now()-meter.started
        gc.enable()
    history=meter.history[:meter.calls].copy()
    valid=np.isfinite(history[:,2]) & np.isfinite(history[:,3])
    gradient=float(np.min(history[valid,3])) if valid.any() else np.inf
    point=run.minimum
    endpoint=point[0].copy()
    verified=gradient; objective=float(point[1])
    if verify:
        try:
            f,g=case['independent'](endpoint)
            verified=float(np.linalg.norm(g)); objective=float(f)
        except Exception as exc:
            error+=' Endpoint verification: '+repr(exc)
            verified=np.inf
    accounting=(meter.calls==run.calls==int(run.counts.sum()))
    if not accounting: error+=' FG counter mismatch.'
    success=bool(not error and accounting and run.status==1 and gradient<=eps
                 and np.isfinite(verified) and verified<=eps*(1.+1e-7))
    status='stationarity' if success else STATUSES.get(int(meter.status or run.status),'unknown')
    if run.status==1 and not success: status='endpoint_verification_failure'
    diagnostics={}
    if method=='A-QNPE' and result is not None:
        diagnostics=dict(zip(('iterations','metric_updates','scale_resets','L1_final',
                             'rank_peak','metric_state_bytes_peak','CR_products','SEP_products'),result[1:]))
    if method=='UAD':
        diagnostics={'counts':run.counts.copy(),'events':run.events.copy(),'final_shared_scale':run.ell}
    row=dict(Method=method,FG=int(meter.calls),Seconds=float(elapsed),Objective=objective,
             Gradient=gradient,VerifiedGradient=verified,Tolerance=eps,Status=status,Success=success,
             Error=error,AQNPEVariant=('online scale extension' if case.get('L1') is None else 'supplied L1') if method=='A-QNPE' else '',
             Diagnostics=json.dumps(diagnostics,default=serial),EndpointChecks=int(verify))
    return row,history,endpoint,run.counts.copy(),run.events.copy()


@njit(API.FG_SIGNATURE,cache=False)
def validation_quadratic(x):
    g=np.empty_like(x)
    f=0.
    for j in range(len(x)):
        g[j]=(j+1.)*x[j]
        f+=.5*x[j]*g[j]
    return f,g


def precompile():
    case=dict(x0=np.array([2.,-1.,.5,-2.]),native_fg=validation_quadratic,L1=4.,L2=1.)
    records=[]
    for method in METHODS:
        start=time.perf_counter()
        row,*_=measure(method,case,1e-12 if method=='CUG' else 0.,16,np.inf,False)
        record=dict(method=method,FG=row['FG'],status=row['Status'],error=row['Error'],seconds=time.perf_counter()-start)
        records.append(record);emit(dict(stage='precompile',**record))
        if row['Error']: raise RuntimeError(record)
    row,*_=measure('A-QNPE',dict(case,L1=None),0.,16,np.inf,False)
    records.append(dict(method='A-QNPE-online-scale',FG=row['FG'],status=row['Status'],error=row['Error']))
    if row['Error']: raise RuntimeError(records[-1])
    return records


