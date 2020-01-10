from astropy.io import fits
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Converts TESS official LCs to Juliet format')
parser.add_argument('File', help='Light curve file (.FITS)')

args     = parser.parse_args()
filename = args.File

data = fits.getdata(filename)
head = fits.getheader(filename)

TIC  = head['TICID']
sec  = head['SECTOR']
q    = data['QUALITY'] == 0
t    = data['TIME'][q]
f    = data['PDCSAP_FLUX'][q]
e    = data['PDCSAP_FLUX_ERR'][q]

output = np.transpose((t,f,e))
np.savetxt('TIC%d_%02d-SC.dat' % (TIC, sec), output)
