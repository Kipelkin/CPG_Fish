"""Compare outputs from two independent project checkouts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.runtime import ROOT, VALIDATION

TABLES = ['Table_1.csv','Table_2.csv','Table_3.csv','conductance_drift.csv','programming_direction.csv',
          'initialization_summary.csv','stochastic_summary.csv','predictive_tracking.csv',
          'conditional_hardware_bounds.csv','drift_brackets.csv']
EXPERIMENTS = ['predictive','original_32','stochastic','stochastic_extra','sensitivity']


def digest(path, decompress=False):
    open_file = gzip.open if decompress else Path.open
    with open_file(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(other):
    a, b = ROOT/'results', other/'results'
    rows = []
    for name in TABLES + [f'experiments/{suite}/metrics.csv' for suite in EXPERIMENTS]:
        f = Path(name) if '/' in name else Path('tables')/name
        left = pd.read_csv(a/f,float_precision='round_trip',low_memory=False)
        right = pd.read_csv(b/f,float_precision='round_trip',low_memory=False)
        non_scientific = ['runtime_s']
        left = left.drop(columns=non_scientific,errors='ignore')
        right = right.drop(columns=non_scientific,errors='ignore')
        equal = left.equals(right)
        rows.append(dict(file=str(f),number_of_rows=len(left),exact=equal))
    for path in sorted((a/'figures').glob('*.png')):
        rel = path.relative_to(a)
        rows.append(dict(file=str(rel),exact=digest(path)==digest(b/rel)))
    for path in ['validation/run_checks_summary.json','validation/run_level_checks.csv',
                 'experiments/prescribed/trajectories.csv.gz',
                 'experiments/original_32/trace_hash_check.csv',
                 'experiments/phase/results/selected_metrics.csv',
                 'experiments/phase/results/fixed_gain_state_scan.csv']:
        rows.append(dict(file=path,exact=digest(a/path,path.endswith('.gz'))==digest(b/path,path.endswith('.gz'))))
    result = dict(isolated_run=json.loads((b/'validation/pipeline.json').read_text()),
                  environment=json.loads((b/'validation/environment.json').read_text()),
                  compared_outputs=rows,all_exact=all(r['exact'] for r in rows))
    VALIDATION.mkdir(parents=True,exist_ok=True)
    (VALIDATION/'isolated_run_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_exact']:
        raise SystemExit('Some repeated outputs differ. See isolated_run_comparison.json.')
    print(f'{len(rows)}/{len(rows)} compared outputs agree exactly across checkouts.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('other',type=Path)
    main(parser.parse_args().other.resolve())
