"""Run the recording-disjoint fit, phase study and prescribed historical replay."""
import argparse
from scripts.runtime import (CONFIGS, DATA, FIGURES, PROTOCOLS, WORK, command, copy, populate)


def calibration():
    directory = WORK / 'calibration'
    source = PROTOCOLS / 'calibration'
    raw = DATA / 'raw/read_means'
    command('calibration', source / 'calculate.py', '--raw', raw, '--out', directory)
    copy(source / 'supplement.py', directory / 'supplement.py')
    command('calibration_tables', directory / 'supplement.py')
    figure = WORK / 'figure2'
    command('figure2_numerical', source / 'recompute_data.py', '--raw', raw, '--out',
            figure / 'data', '--reference', DATA / 'reference/figure2')
    command('figure2_plot', source / 'make_figure3.py', '--out', figure)
    for ext in ('png', 'pdf'):
        copy(figure / f'Figure_3.{ext}', FIGURES / f'Figure_2.{ext}')


def phase():
    directory = WORK / 'phase'
    source = PROTOCOLS / 'phase'
    inputs = directory / 'inputs'
    populate(DATA / 'raw/read_means', inputs / 'raw')
    populate(WORK / 'calibration/data', inputs / 'section3_2_data',
             ['coefficients.csv', 'screened_prefixes_with_origin.csv', 'measured_transition_pairs.csv.gz'])
    copy(DATA / 'reference/calibration/archive_provenance.json', inputs / 'archive_provenance.json')
    command('phase_dynamics', source / 'calculate.py', '--out', directory / 'results', '--inputs', inputs)
    command('phase_unperturbed', source / 'build_figure4_alternative.py', '--data',
            directory / 'results', '--out', directory / 'figure4_alternative')
    plot = directory / 'figure4_enhanced'
    populate(source, plot, ['prepare_enhanced_data.py', 'plot_figure4_enhanced.py'])
    command('figure3_plot', plot / 'plot_figure4_enhanced.py')
    for ext in ('png', 'pdf'):
        copy(plot / f'Figure_4_enhanced.{ext}', FIGURES / f'Figure_3.{ext}')


def prescribed():
    directory = WORK / 'prescribed'
    populate(PROTOCOLS / 'prescribed', directory)
    populate(DATA / 'processed/prescribed', directory / 'reference')
    populate(DATA / 'processed/control_inputs', directory / 'reference',
             ['coefficients.csv', 'interface.json', 'selected_virtual_devices.csv',
              'state_catalog.csv', 'support_halfspaces.csv'])
    copy(CONFIGS / 'prescribed.json', directory / 'parameters.json')
    command('figure4_prescribed', directory / 'figure5.py')
    for ext in ('png', 'pdf'):
        copy(directory / f'Figure_5_reference_style.{ext}', FIGURES / f'Figure_4_prescribed.{ext}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['calibration', 'phase', 'prescribed'])
    args = parser.parse_args()
    globals()[args.suite]()
