"""Create and CRC-check a private review archive of the current project."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from scripts.runtime import ROOT

IGNORED = {'.git','.venv','__pycache__','.matplotlib','build','dist'}
COMPRESSED = {'.gz','.zip','.png','.pdf'}


def main(output):
    if output.is_relative_to(ROOT):
        raise ValueError('Place the archive outside the project directory.')
    staging = output.with_suffix('.part')
    with ZipFile(staging,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        files = 0
        for source in sorted(ROOT.rglob('*')):
            rel = source.relative_to(ROOT)
            if (not source.is_file() or source.is_symlink() or
                any(part in IGNORED or part.endswith('.egg-info') for part in rel.parts) or
                source.suffix in {'.pyc','.so'} or source.name=='.DS_Store'):
                continue
            method = ZIP_STORED if source.suffix in COMPRESSED else ZIP_DEFLATED
            archive.write(source,(Path(ROOT.name)/rel).as_posix(),compress_type=method)
            files += 1
    with ZipFile(staging) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failure: {bad}')
        assert len(archive.namelist())==files
    staging.replace(output)
    print(f'Validated {files} files in {output.name} ({output.stat().st_size} bytes).')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,nargs='?',default=ROOT.parent/f'{ROOT.name}.zip')
    main(parser.parse_args().output.resolve())
