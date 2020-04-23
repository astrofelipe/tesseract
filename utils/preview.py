import matplotlib.pyplot as plt
import glob
import h5py
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import argparse
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description='Test lightcurves')
parser.add_argument('RA', type=float)
parser.add_argument('DEC', type=float)
parser.add_argument('--neigh', action='store_true')
parser.add_argument('--nonorm', action='store_true')

args = parser.parse_args()
ra0  = args.RA
dec0 = args.DEC

allfits = glob.glob('*.fits')
showimg = fits.getdata(allfits[0])
wcshdr  = fits.getheader(allfits[0], 1)
w       = WCS(wcshdr)
x0, y0  = w.all_world2pix(ra0, dec0, 1)

x, y, ra, dec = np.genfromtxt('positions.dat', unpack=True, usecols=(1,2,3,4))
f = h5py.File('supertable.hdf5', mode='r')
data = f['data']

dist = (ra - ra0)**2 + (dec - dec0)**2
dord = np.argsort(dist)
idx  = dord[0] + 2
print(ra[dord[0]], dec[dord[0]])

print('Distancia: ', dist[idx-2])
#xf, yf = w.wcs_world2pix(ra[dord[0]], dec[dord[0]], 1)
xf, yf = x[dord[0]], y[dord[0]]

figo, axo = plt.subplots(figsize=[8,6])
axo.matshow(np.log10(showimg))
axo.plot(x0, y0, 'xr')
axo.plot(xf, yf, 'c+')
axo.plot(x, y, '.b', ms=1)

qual = data[1] == 0
time = data[0][qual]
flux = data[idx,qual]

fig, ax = plt.subplots(figsize=[19,3])

fm = np.nanmedian(flux) if args.nonorm else medfilt(flux, 35)
flux /= fm

if args.neigh:
    for ii in dord[1:11]+2:
        fm = np.nanmedian(data[ii,qual]) if args.nonorm else medfilt(data[ii,qual], 35)
        ax.plot(time, data[ii,qual]/fm, '.')

ax.plot(time, flux, '.k')
#ax.plot(time, fm, '-r')

plt.show()
