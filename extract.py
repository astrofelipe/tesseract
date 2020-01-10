from astropy.io import fits
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Converts TESS official LCs to Juliet format')
parser.add_argument('File', help='Light curve file (.FITS)')

args     = parser.parse_args()
filename = args.File

data  = fits.getdata(filename)
head  = fits.getheader(filename, 0)
head2 = fits.getheader(filename, 1)

TIC  = head['TICID']
sec  = head['SECTOR']
bjdr = head2['BJDREFI']

q    = (data['QUALITY'] == 0) & np.isfinite(data['PDCSAP_FLUX'])
f    = data['PDCSAP_FLUX'][q]
med  = np.nanmedian(f)
f   /= med

t    = data['TIME'][q] + bjdr
e    = data['PDCSAP_FLUX_ERR'][q] / med

print(data['PDCSAP_FLUX'])

output = np.transpose((t,f,e))
np.savetxt('TIC%d_%02d-SC.dat' % (TIC, sec), output, fmt='%s')
