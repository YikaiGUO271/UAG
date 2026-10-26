"""Fetch user-requested LIBSVM data and verify frozen signed CSR arrays.

Dataset files are deliberately not redistributed with this software. Review the
linked dataset terms before requesting downloads. Existing data is never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

HERE = Path(__file__).resolve().parent

def convert(raw, item):
    data, indices, indptr = [], [], [0]
    lines = [line.split() for line in raw.decode('ascii').splitlines() if line.strip()]
    labels = sorted(set(float(line[0]) for line in lines))
    if len(labels) != 2: raise ValueError('Expected exactly two labels')
    for row in lines:
        sign = 1. if float(row[0]) == labels[-1] else -1.
        previous = -1
        for entry in row[1:]:
            col, value = entry.split(':'); col = int(col) - 1; value = float(value)
            if not 0 <= col < item['shape'][1] or col <= previous:
                raise ValueError('Unexpected feature ordering or dimension')
            previous = col
            if value != 0.:
                data.append(-sign * value); indices.append(col)
        indptr.append(len(data))
    arrays = dict(data=np.array(data, dtype=np.float64), indices=np.array(indices, dtype=np.int32),
                  indptr=np.array(indptr, dtype=np.int32), shape=np.array(item['shape'], dtype=np.int64))
    assert len(lines) == item['shape'][0]
    for key, expected in item['arrays'].items():
        array = arrays[key]
        assert list(array.shape) == expected['shape'] and str(array.dtype) == expected['dtype']
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256'], (item['name'], key)
    return arrays

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true', help='Explicitly fetch the listed external datasets')
    p.add_argument('--raw-dir', type=Path, help='Alternatively use already acquired files named as in their source URLs')
    p.add_argument('--output', type=Path, default=HERE / 'cases/data')
    p.add_argument('--names', nargs='+', default=['a9a','australian','ionosphere','w8a'])
    a = p.parse_args()
    if not a.download and a.raw_dir is None: p.error('Supply --download or --raw-dir')
    items = json.loads((HERE / 'manifests/logistic_data.json').read_text())
    unknown = set(a.names) - {i['name'] for i in items}
    if unknown: p.error('Unknown datasets: ' + str(sorted(unknown)))
    output = a.output.resolve()
    frozen = (HERE / 'frozen').resolve()
    if frozen == output or frozen in output.parents: raise ValueError('Frozen artifacts are read-only')
    output.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item['name'] not in a.names: continue
        target = output / ('super_logistic_' + item['name'] + '.npz')
        if target.exists(): raise FileExistsError('Refusing to replace ' + str(target))
        if a.raw_dir is not None: raw = (a.raw_dir / item['url'].rsplit('/',1)[-1]).read_bytes()
        else:
            with urlopen(item['url'], timeout=60) as response: raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == item['raw_sha256'], 'Raw dataset checksum mismatch'
        arrays = convert(raw, item)
        np.savez_compressed(target, **arrays)
        print(json.dumps(dict(dataset=item['name'], arrays_match_frozen=True,
                             raw_sha256=hashlib.sha256(raw).hexdigest())), flush=True)

if __name__ == '__main__': main()
