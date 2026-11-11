"""Optional bitmap provenance check with the Poppler pdfimages command."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.runtime import DATA, VALIDATION


def main(pdfimages):
    ref = DATA/'reference/manuscript'
    rows = []
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for manuscript in ('Fish_CPG.pdf','Fish_CPG-2.pdf'):
            subprocess.run([pdfimages,'-f','22','-l','25','-png',str(ref/manuscript),str(tmp/'Fish')],
                           check=True)
            images = sorted(tmp.glob('Fish-*.png'))
            assert len(images)==8, 'Unexpected PDF image layout'
            for n in range(1,5):
                with Image.open(images[2*(n-1)]) as color, Image.open(images[2*(n-1)+1]) as mask, \
                        Image.open(ref/'figures'/f'Figure_{n}.png') as supplied:
                    actual = color.convert('RGBA')
                    actual.putalpha(mask.convert('L'))
                    expected = supplied.convert('RGBA')
                    assert actual.size==expected.size
                    exact = np.array_equal(np.asarray(actual),np.asarray(expected))
                    white = Image.new('RGBA',actual.size,'white')
                    a = np.asarray(Image.alpha_composite(white,actual).convert('RGB'))
                    b = np.asarray(Image.alpha_composite(white,expected).convert('RGB'))
                    difference = float(np.any(a!=b,axis=2).mean())
                    rows.append(dict(manuscript=manuscript,figure=n,pdf_page=n+21,width=color.width,height=color.height,
                                     exact_RGBA=bool(exact),exact_composite_white=difference==0.,
                                     fraction_different_composite_pixels=difference))
    VALIDATION.mkdir(parents=True,exist_ok=True)
    (VALIDATION/'manuscript_embedded_figures.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'{sum(r["exact_RGBA"] for r in rows)}/{len(rows)} embedded figures match supplied PNGs pixel for pixel.')


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pdfimages',default='pdfimages',help='Poppler executable')
    main(cli.parse_args().pdfimages)
