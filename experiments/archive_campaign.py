"""Package a stopped campaign with exact NPZ bytes, one uncompressed ZIP per repeat."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();source=a.campaign.resolve();out=a.output.resolve()
    if not (source/'complete.json').exists():raise ValueError('Campaign has not stopped')
    if source==out or source in out.parents:raise ValueError('Archive output must be outside the campaign')
    for name in ('frozen','reference','cases','manifests','third_party'):
        protected=(HERE/name).resolve()
        if out==protected or protected in out.parents:raise ValueError('Protected output path')
    audit=json.loads((source/'trace_audit.json').read_text())
    if not audit['passed'] or audit['results_sha256']!=digest(source/'results.json'):
        raise ValueError('Run audit_repetitions.py on the completed results first')
    complete=json.loads((source/'complete.json').read_text())
    rows=json.loads((source/'results.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    files=['results.json','results.csv','protocol.json','environment.json','warmup.json',
           'setup_cost.json','complete.json','driver_used.py','trace_hashes.json','trace_audit.json',
           'original_sources_audit.json']
    for name in files:
        if (source/name).exists():shutil.copyfile(source/name,out/name)
    if (source/'analysis').is_dir():shutil.copytree(source/'analysis',out/'analysis')
    groups=defaultdict(list)
    for row in rows:groups[row['Repeat']].append(row)
    traces=json.loads((source/'trace_hashes.json').read_text())
    manifests=[]
    for repeat,group in sorted(groups.items()):
        is_complete=repeat in complete['completed_repeats']
        name=f'traces_r{repeat:02d}'+('' if is_complete else '_partial')+'.zip'
        path=out/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
            for row in group:
                base=row['Trace'].replace('\\','/').rsplit('/',1)[-1]
                src=source/'traces'/base
                if digest(src)!=traces[base]:raise ValueError('Trace hash changed: '+base)
                archive.write(src,arcname='traces/'+base)
        if path.stat().st_size>=100*1024*1024:
            raise ValueError('Archive exceeds GitHub individual-file limit; split it before publishing')
        manifests.append(dict(file=name,repeat=repeat,complete=is_complete,observations=len(group),
                              bytes=path.stat().st_size,sha256=digest(path),compression='ZIP_STORED'))
    (out/'archive_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
    hashes={str(f.relative_to(out)).replace('\\','/'):digest(f) for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),files=len(hashes)+1,bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),archives=manifests),indent=2))

if __name__=='__main__':main()
