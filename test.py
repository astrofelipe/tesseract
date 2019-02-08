from astropy.io import fits
from utils import detrender
import numpy as np
import matplotlib.pyplot as plt

fname = '/data/TESS/LC/official/s1/tess2018206045859-s0001-0000000038827910-0120-s_lc.fits'

print 'Abriendo'

data = fits.getdata(fname)
mask = np.isfinite(data['TIME']) & np.isfinite(data['PDCSAP_FLUX']) & (data['QUALITY'] == 0)
t    = data['TIME'][mask]
y    = data['PDCSAP_FLUX'][mask]
yerr = data['PDCSAP_FLUX_ERR'][mask]

print 'Detrending'
mu, var = detrender(t, y, yerr)

print 'Plot'

fig, ax = plt.subplots()
ax.plot(t, y, '.', ms=1)
ax.plot(t, mu, '-r')
plt.show()
