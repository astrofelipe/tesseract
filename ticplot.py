import __future__
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from photutils import MMMBackground
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.io import fits

parser = argparse.ArgumentParser(description='Plot TICs over FFIs')
parser.add_argument('File', type=str, help='FFI File')

args = parser.parse_args()

#Leer FITS
hdus = fits.open(args.File)
data = hdus[1].data
hdr  = hdus[1].header

#Donde estoy y filtro catalogo
camera = hdr['CAMERA']
chip   = hdr['CCD']
sector = args.File.split('-')[1][-3:]
cata   = '/Volumes/Felipe/TESS/all_targets_S%s_v1.csv' % sector
cata   = pd.read_csv(cata, comment='#')

mask = (cata['Camera'] == camera) & (cata['CCD'] == chip)
cata = cata[mask]
ra   = cata['RA']
dec  = cata['Dec']

#RADEC to pix
w = WCS(hdr)
x,y = w.all_world2pix(ra, dec, 0)

fig, ax = plt.subplots()
ax.matshow(np.log10(data))
ax.plot(x, y, '.r', ms=1)

fig2, ax2 = plt.subplots(nrows=4, ncols=4)
axs2 = np.ravel(ax2)
tids = np.random.choice(cata['TICID'], size=16)

for i in range(16):
    xx, yy = int(x[i]), int(y[i])
    subs   = np.ravel(data[xx-7:xx+8, yy-7:yy+8])

    sigma_clip = SigmaClip(sigma=3.)
    bkg        = MMMBackground(sigma_clip=sigma_clip)
    bkg_value  = bkg.calc_background(subs)
    axs2[i].hist(subs, bins=50)
    axs2[i].axvline(bkg_value, color='r')

plt.show()
