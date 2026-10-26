"""Verify immutable observations and tabulate every failure category."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent

def audit(output, check_traces=False):
    frozen = HERE / 'frozen'
    expected = json.loads((frozen / 'SHA256.json').read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest, name
    rows = json.loads((frozen / 'results.json').read_text())
    protocol = json.loads((frozen / 'protocol.json').read_text())
    expected_keys = {(suite, case['name'], method)
                     for suite in ('specific', 'cutest') for case in protocol[suite]
                     for method in protocol['methods_' + suite]}
    actual = {(r['Suite'], r['Problem'], r['Method']) for r in rows}
    assert actual == expected_keys and len(rows) == len(actual) == 457
    trace_checks = 0
    if check_traces:
        hashes = json.loads((HERE / 'manifests/trace_hashes.json').read_text())
        with zipfile.ZipFile(frozen / 'traces.zip') as archive:
            assert set(archive.namelist()) == set(hashes)
            for row in rows:
                name = row['Trace'].replace('\\', '/').rsplit('/', 1)[-1]
                content = archive.read(name)
                assert hashlib.sha256(content).hexdigest() == hashes[name]['sha256']
                with np.load(io.BytesIO(content), allow_pickle=False) as stored:
                    h = stored['history']
                    assert len(h) == row['FG'] == int(stored['counts'].sum())
                    assert np.array_equal(h[:, 0], np.arange(1, len(h) + 1))
                    assert np.all(np.diff(h[:, 1]) >= 0)
                    finite = np.isfinite(h[:, 2]) & np.isfinite(h[:, 3])
                    best = float(h[finite, 3].min()) if finite.any() else np.inf
                    assert best == row['Gradient']
                    hits = np.flatnonzero(finite & (h[:, 3] <= row['Tolerance']))
                    if row['Success']:
                        assert len(hits) == 1 and hits[0] == len(h) - 1
                        assert row['VerifiedGradient'] <= row['Tolerance'] * (1 + 1e-7)
                        assert not row['Error']
                    else:
                        assert len(hits) == 0 or row['Status'] == 'endpoint_verification_failure'
                trace_checks += 1
    output = Path(output).resolve()
    if frozen.resolve() == output or frozen.resolve() in output.parents:
        raise ValueError('Refusing to write inside frozen observations')
    output.mkdir(parents=True, exist_ok=True)
    categories = Counter((r['Suite'], r['Method'], r['Status'], r['Success']) for r in rows)
    with (output / 'failure_categories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['suite', 'method', 'status', 'success', 'count'])
        writer.writerows((*key, value) for key, value in sorted(categories.items()))
    failures = [r for r in rows if not r['Success']]
    (output / 'unsuccessful_runs.json').write_text(json.dumps(failures, indent=2) + '\n')
    report = dict(passed=True, observations=len(rows), successful=len(rows)-len(failures),
                  unsuccessful=len(failures), trace_checks=trace_checks, optimization_runs=0,
                  statuses=dict(Counter(r['Status'] for r in rows)), frozen_files_verified=expected)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'generated/audit')
    p.add_argument('--check-traces', action='store_true')
    a = p.parse_args()
    print(json.dumps(audit(a.output, a.check_traces), indent=2))
