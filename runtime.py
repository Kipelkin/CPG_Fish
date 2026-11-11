"""Isolate the historical, script-oriented implementations in a run directory."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CONFIGS = ROOT / 'configs'
PROTOCOLS = ROOT / 'src/memristive_cpg/protocols'
RESULTS = ROOT / 'results'
WORK = RESULTS / 'experiments'
FIGURES = RESULTS / 'figures'
VALIDATION = RESULTS / 'validation'
TABLES = RESULTS / 'tables'
LOGS = RESULTS / 'logs'


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        shutil.copyfile(source, destination)


def populate(source, destination, names=None):
    files = [source / name for name in names] if names else source.rglob('*')
    for path in files:
        if path.is_file():
            copy(path, destination / path.relative_to(source))


def command(label, *args, cwd=ROOT):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f'{label}.log'
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(WORK / '.matplotlib'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with log.open('w', encoding='utf-8') as stream:
        completed = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{label} exited {completed.returncode}\n' + log.read_text()[-5000:])
    print(f'{label} completed', flush=True)


def control(destination):
    populate(PROTOCOLS / 'control', destination)
    populate(DATA / 'processed/control_inputs', destination / 'inputs')
    populate(CONFIGS / 'control', destination)
    populate(DATA / 'reference/control_provenance', destination / 'reference')
    copy(CONFIGS / 'predictive.json', destination / 'parameters.json')


def stochastic(destination, baseline=True):
    populate(PROTOCOLS / 'stochastic', destination)
    populate(CONFIGS / 'stochastic', destination)
    if baseline:
        control(destination / 'baseline')


def sensitivity(destination):
    populate(PROTOCOLS / 'sensitivity', destination / 'source')
    stochastic(destination / 'source/reused')
    copy(CONFIGS / 'sensitivity.json', destination / 'parameters.json')
    copy(CONFIGS / 'NEW_JOBS.json', destination / 'NEW_JOBS.json')
    (destination / 'runs').mkdir(exist_ok=True)
