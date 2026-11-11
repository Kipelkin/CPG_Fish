"""Regenerate numerical results and figures, then audit them against the manuscript."""
import argparse
from importlib.metadata import version
import json
import os
import platform
import shutil
import subprocess
import sys
from time import monotonic

from scripts.runtime import ROOT, VALIDATION, command


def preflight():
    if sys.version_info[:2] != (3,13):
        raise SystemExit('Use Python 3.13 with the pinned dependencies in requirements.txt.')
    expected = {'numpy':'2.3.5','scipy':'1.16.3','pandas':'3.0.1','matplotlib':'3.10.7'}
    for name, required in expected.items():
        installed = version(name)
        if installed != required:
            raise SystemExit(f'{name}=={required} required, found {installed}.')
    if not shutil.which('clang++'):
        raise SystemExit('clang++ with C++17 support is required for the archived RK4 kernels.')
    VALIDATION.mkdir(parents=True,exist_ok=True)
    compiler = subprocess.check_output(['clang++','--version'],text=True).splitlines()[0]
    data = dict(python=platform.python_version(),platform=platform.platform(),compiler=compiler,
                isolated_venv=sys.prefix != sys.base_prefix,libraries=expected)
    (VALIDATION/'environment.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=min(os.cpu_count() or 1,6))
    parser.add_argument('--verify-existing',action='store_true',help='Re-audit saved outputs; skip integrations.')
    parser.add_argument('--skip-waveforms',action='store_true',help='Skip the 77 full waveform read-window checks.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    os.environ['PYTHONPATH'] = str(ROOT/'src')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    preflight()
    t = monotonic()
    command('tests','-m','unittest','discover','-s','tests','-v')
    if not args.verify_existing:
        wave = ['--skip-waveforms'] if args.skip_waveforms else []
        command('raw_archive','-m','scripts.run_measurements',*wave)
        for suite in ('calibration','phase','prescribed'):
            command(f'suite_{suite}','-m','scripts.run_dynamics',suite)
        for suite in ('predictive','original_32','cohort','extra_cohort','initialization','imposed_drift'):
            command(suite,'-m','scripts.run_control',suite,'--workers',args.workers)
        command('hardware','-m','scripts.run_hardware')
    command('comparison','-m','scripts.compare_results')
    command('run_level_checks','-m','scripts.validate_runs','--workers',args.workers)
    command('figures','-m','scripts.generate_figures')
    data = dict(completed=True, mode='verify_existing' if args.verify_existing else 'fresh',
                skipped_waveforms=args.skip_waveforms, workers=args.workers,
                elapsed_seconds=monotonic()-t)
    (VALIDATION/'pipeline.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f'Completed in {data["elapsed_seconds"]:.1f} s. See results/validation/comparison.csv.')


if __name__ == '__main__':
    main()
