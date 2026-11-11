"""Audit the raw archive and independently derive endurance data."""
import argparse
from memristive_cpg.data_processing import audit_archive, endurance
from scripts.runtime import DATA, WORK


def main(skip_waveforms=False):
    dest = WORK / 'measurements'
    dest.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'raw/device_archive.zip'
    records, members = audit_archive(archive, DATA / 'raw/read_means', not skip_waveforms)
    reads, summary = endurance(archive)
    common = summary[summary.n_records == reads.record.nunique()]
    # The Figure 1B source provenance records that its first four cycles were omitted.
    shown = common[common.protocol_cycle >= 5]
    for name, frame in [('archive_records', records), ('archive_members', members),
                        ('endurance_record_means', reads), ('endurance_all', summary),
                        ('endurance_plotted', shown)]:
        frame.to_csv(dest / f'{name}.csv', index=False, float_format='%.17g')
    print(f'{len(records)} records, {records.cycles.sum()} cycles, {records.transitions.sum()} transitions, '
          f'{records.waveform_checked.sum()} complete waveform checks, {reads.record.nunique()} endurance records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-waveforms', action='store_true')
    main(parser.parse_args().skip_waveforms)
