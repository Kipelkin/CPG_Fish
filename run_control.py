"""Fresh causal-control, stochastic and imposed-drift runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import sys

import pandas as pd

from scripts.runtime import CONFIGS, DATA, WORK, command, copy, control, stochastic, sensitivity


def read_json(path):
    return json.loads(path.read_text())


def worker(spec):
    engine, path, kwargs = spec
    sys.path.insert(0, path)
    if engine == 'control':
        from memristive_robotic_control import simulate
    elif engine == 'timed':
        from timed_simulation import simulate
    elif engine == 'redesign':
        from redesign_simulation import simulate
    elif engine == 'reliability':
        from reliability_simulation import simulate
    else:
        from sensitivity_simulation import simulate
        return simulate(kwargs, force=True)
    return simulate(**kwargs, force=True)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.17g')


def run_jobs(specs, workers, destination):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(worker, specs, chunksize=1), start=1):
            rows.append(row)
            if i % 50 == 0 or i == len(specs):
                print(f'{destination.name}: {i}/{len(specs)}', flush=True)
                write_csv(destination, rows)
    return rows


def predictive(workers):
    runtime = WORK / 'predictive'
    control(runtime)
    sys.path.insert(0, str(runtime))
    import numerics
    p = read_json(CONFIGS / 'predictive.json')
    base = dict(target_index=p['main_target_state_DEP_index'], step=p['main_target_step'],
                pulse_start=p['pulse_start'], pulse_area=p['pulse_area'],
                end=p['main_end'], evaluation=p['main_eval'])
    specs = []
    for group, scene in [('main', dict(base, name='main')),
                         ('manuscript_2p9', dict(base, name='main_2p9', target_Hz=2.9))]:
        for arch in ['S', 'N']:
            for kind in ['Fixed', 'Ideal', 'Device']:
                for budget in p['physical_budgets'] if kind == 'Device' else [0]:
                    specs.append(('control', str(runtime), dict(controller=f'{arch}-{kind}', p=p,
                                 scenario=scene, budget=budget, group=group)))
    run_jobs(specs, workers, runtime / 'metrics.csv')


def original_32(workers):
    runtime = WORK / 'original_32'
    control(runtime)
    sys.path.insert(0, str(runtime))
    import numerics
    p = read_json(CONFIGS / 'predictive.json')
    scene = dict(name='main', target_index=p['main_target_state_DEP_index'], step=p['main_target_step'],
                 pulse_start=p['pulse_start'], pulse_area=p['pulse_area'],
                 end=p['main_end'], evaluation=p['main_eval'])
    jobs = [('control', str(runtime), dict(controller='N-Device', p=p, scenario=scene, budget=10,
             seed=seed, mode='independent_virtual_blocks', group='reliability_exact_replay'))
            for seed in range(31001, 31033)]
    rows = run_jobs(jobs, workers, runtime / 'metrics.csv')
    expected = read_json(CONFIGS / 'stochastic/original_32_trace_hashes.json')
    hashes = []
    for row in rows:
        trace = runtime / 'runs' / row['run_id'] / 'trace.csv.gz'
        with gzip.open(trace, 'rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        hashes.append(dict(seed=row['seed'], expected=expected[str(row['seed'])], actual=actual,
                           exact=actual == expected[str(row['seed'])]))
    write_csv(runtime / 'trace_hash_check.csv', hashes)


def run_cohort(workers, supplementary=False):
    runtime = WORK / ('stochastic_extra' if supplementary else 'stochastic')
    stochastic(runtime)
    # Build the exact same numerical kernels once before spawning workers.
    sys.path[:0] = [str(runtime), str(runtime / 'baseline')]
    import numerics
    import guard_numerics
    name = 'extra_cohort_jobs.json' if supplementary else 'cohort_jobs.json'
    frozen = read_json(CONFIGS / 'stochastic' / name)
    specs = []
    for job in frozen['jobs']:
        params = frozen['parameter_sets'][job['parameters']]
        kwargs = {key: job[key] for key in ('controller', 'scenario', 'budget', 'seed', 'mode', 'group')}
        specs.append((job['engine'], str(runtime), dict(kwargs, p=params)))
    rows = run_jobs(specs, workers, runtime / 'metrics.csv')
    for row, job in zip(rows, frozen['jobs'], strict=True):
        row['reference_run_id'] = job['reference_id']
        row['engine'] = job['engine']
    write_csv(runtime / 'metrics.csv', rows)



def cohort(workers):
    run_cohort(workers)


def extra_cohort(workers):
    run_cohort(workers, supplementary=True)


def initialization(workers):
    runtime = WORK / 'initialization'
    stochastic(runtime)
    copy(DATA / 'reference/stochastic/canonical_initial_configurations.csv.gz',
         runtime / 'canonical_initial_configurations.csv.gz')
    for name in ('initialization_stress', 'unconditioned_initialization_sensitivity',
                 'global_initialization_scan'):
        command(name, runtime / f'{name}.py')

def imposed_drift(workers):
    runtime = WORK / 'sensitivity'
    sensitivity(runtime)
    sys.path[:0] = [str(runtime / 'source')]
    import common
    import numerics
    import guard_numerics
    from sensitivity_simulation import load_linear
    load_linear()
    jobs = read_json(CONFIGS / 'NEW_JOBS.json')
    refined = pd.read_csv(DATA / 'reference/sensitivity/refined_results.csv', usecols=['run_id'])
    # The archived refinement set uses the frozen subset implemented by refine_new.py.
    refined_jobs = [dict(job, refined=True) for job in jobs
                    if job['suite'] == 'asymmetry' or job.get('level', 1) in [-.02, 0., .02]]
    assert len(refined_jobs) == len(refined)
    base_specs = [('sensitivity', str(runtime / 'source'), job) for job in jobs]
    fine_specs = [('sensitivity', str(runtime / 'source'), job) for job in refined_jobs]
    run_jobs(base_specs + fine_specs, workers, runtime / 'metrics.csv')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('suite', choices=['predictive', 'original_32', 'cohort', 'extra_cohort', 'initialization', 'imposed_drift'])
    cli.add_argument('--workers', type=int, default=4)
    args = cli.parse_args()
    if args.workers < 1:
        cli.error('workers must be positive')
    globals()[args.suite](args.workers)
