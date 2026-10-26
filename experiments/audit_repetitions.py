"""Audit every saved repeated-run trace after the campaign has stopped."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args();root=a.campaign.resolve()
    if not (root/'complete.json').exists():raise ValueError('Wait until the campaign has completed or stopped at its deadline')
    rows=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256((root/'driver_used.py').read_bytes()).hexdigest()==protocol['driver_sha256']
    archives={};lookup={}
    for path in sorted(root.glob('traces*.zip')):
        archive=zipfile.ZipFile(path);archives[path.name]=archive
        for name in archive.namelist():
            base=name.replace('\\','/').rsplit('/',1)[-1]
            assert base not in lookup,base
            lookup[base]=(archive,name)
    trace_hashes={};keys=set()
    for row in rows:
        key=(row['Repeat'],row['Suite'],row['Problem'],row['Method']);assert key not in keys;keys.add(key)
        cfg=protocol['config'][row['Suite']]
        expected_target=cfg.get('eps',cfg.get('relative_eps',0)*max(1.,row['InitialGradient']))
        assert row['Tolerance']==expected_target,('target mismatch',key)
        assert row['FG']<=int(protocol.get('max_fg_override') or cfg['max_fg'])
        name=row['Trace'].replace('\\','/').rsplit('/',1)[-1];local=root/'traces'/name
        if local.exists():content=local.read_bytes()
        else:
            archive,entry=lookup[name];content=archive.read(entry)
        trace_hashes[name]=hashlib.sha256(content).hexdigest()
        with np.load(io.BytesIO(content),allow_pickle=False) as saved:
            h=saved['history'];assert len(h)==row['FG']==int(saved['counts'].sum())
            assert np.array_equal(h[:,0],np.arange(1,len(h)+1))
            assert np.all(np.diff(h[:,1])>=0) and np.all(h[:,1]>=0)
            if len(h):assert h[-1,1]<=row['Seconds']+1e-8
            finite=np.isfinite(h[:,2])&np.isfinite(h[:,3]);best=float(h[finite,3].min()) if finite.any() else np.inf
            assert best==row['Gradient']
            hits=np.flatnonzero(finite&(h[:,3]<=row['Tolerance']))
            if row['Success']:
                assert row['Status']=='stationarity'
                assert len(hits)==1 and hits[0]==len(h)-1
                assert row['VerifiedGradient']<=row['Tolerance']*(1+1e-7) and not row['Error']
            else:assert len(hits)==0 or row['Status']=='endpoint_verification_failure'
    for archive in archives.values():archive.close()
    report=dict(passed=True,observations=len(rows),trace_checks=len(trace_hashes),successful_first_target_checks=sum(r['Success'] for r in rows),all_recorded_FG_counters_verified=True,targets_match_protocol=True,statuses=dict(Counter(r['Status'] for r in rows)),driver_sha256=protocol['driver_sha256'],results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest())
    (root/'trace_hashes.json').write_text(json.dumps(trace_hashes,indent=2)+'\n')
    (root/'trace_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
