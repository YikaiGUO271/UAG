"""Single-threaded, repository-local reference benchmark environment."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORMAL = ROOT / 'cases'
FROZEN = ROOT / 'frozen'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '_cache' / 'numba'))
sys.dont_write_bytecode = True
