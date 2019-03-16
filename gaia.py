from astropy.io import fits
import os
import sys


sect = int(sys.argv[1])
data = fits.getdata('gaia.fits', 1)

ra  = data['ra2000']
dec = data['dec2000']


for i in range(len(ra)):
    run = 'python ticlc.py %f %f %d --noplots' % (ra[i], dec[i], sect)
    os.system(run)
