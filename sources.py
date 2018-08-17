import h5py
import argparse
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
from astropy.stats import sigma_clipped_stats

parser = argparse.ArgumentParser(description='Clean and align images')
parser.add_argument('File', type=str, help='File containing masterframe')
args = parser.parse_args()

File = args.File

inpt   = h5py.File(File, 'r')
mframe = inpt['mframe'][:]

mean, median, std = sigma_clipped_stats(mframe, sigma=3, iters=5)
daofind = DAOStarFinder(fwhm=1.5, threshold=3*std)
sources = daofind(mframe - median)
pos     = (sources['xcentroid'], sources['ycentroid'])

rads      = range(1,7)
apertures = [CircularAperture(pos, r=r) for r in rads]
photable  = aperture_photometry(mframe, apertures, method='exact')

print photable

import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
ax.matshow(np.log10(mframe), cmap='bone')
ax.plot(pos[0], pos[1], '.r', ms=.5)
plt.show()
