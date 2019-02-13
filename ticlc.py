import __future__
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from lightkurve.lightcurve import TessLightCurve
from autoap import generate_aperture, select_aperture
from photutils import MMMBackground, SExtractorBackground, Background2D, CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, BoxLeastSquares
from astropy.wcs import WCS
from astropy.io import fits
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('TIC', type=int, help='TIC ID')
parser.add_argument('Sector', type=int, help='Sector')

args = parser.parse_args()

cata   = '/Volumes/Felipe/TESS/all_targets_S%03d_v1.csv' % args.Sector
cata   = pd.read_csv(cata, comment='#')
target = cata[cata['TICID'] == args.TIC]
print(target)

ra  = float(target['RA'])
dec = float(target['Dec'])
coord = SkyCoord(ra, dec, unit='deg')

'''
#Offline mode
from lightkurve import KeplerTargetPixelFile
fnames = np.sort(glob.glob('*sp.fits'))
hdus   = KeplerTargetPixelFile.from_fits_images(images=fnames,
                                                position=coord,
                                                size=(21,21),
                                                target_id=str(args.TIC))
'''

#Online mode
from astroquery.mast import Tesscut
hdus  = Tesscut.get_cutouts(coord, 21)
print('Target found on ',len(hdus),' sectors')

#Data type
qual = hdus[0][1].data['QUALITY'] == 0
time = hdus[0][1].data['TIME'][qual] + hdus[0][1].header['BJDREFI']
flux = hdus[0][1].data['FLUX'][qual]
errs = hdus[0][1].data['FLUX_ERR'][qual]
bkgs = np.zeros(len(flux))
lcfl = np.zeros(len(flux))

#Star position
w = WCS(hdus[0][2].header)
x,y = w.all_world2pix(ra, dec, 0)

#Background
for i,f in enumerate(flux):
    sigma_clip = SigmaClip(sigma=2.5)
    bkg        = MMMBackground(sigma_clip=sigma_clip)
    bkgs[i]    = bkg.calc_background(f)

#DBSCAN Aperture
daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,3,5,7,9,11,13,15]]
dap  = np.array([select_aperture(d, x, y) for d in daps])

#Aperture photometry
lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

#Lightkurves
lks = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]
lkf = [lk.flatten(polyorder=2, window_length=51) for lk in lks]

#Select best
cdpp = [lk.estimate_cdpp() for lk in lkf]
bidx = np.argmin(cdpp)
print('Best lk:', bidx)


aps    = CircularAperture([(x,y)], r=2.5)

fig1, ax1 = plt.subplots(ncols=2)
ax1[0].matshow(np.log10(flux[4]))
ax1[0].matshow(dap[bidx], alpha=.2)
ax1[0].plot(x,y, '.r')
aps.plot(color='w', ax=ax1[0])
#ax1[1].matshow(bkgs[4])

fig, ax = plt.subplots(figsize=[19,4])
ax.plot(time, lkf[bidx].flux, '-ok', ms=2)
#ax.plot(time, mf, '-r', lw=1)
#for i in range(10):
#    ax.axvline(2458438.38600+i*4.05200)
ax.ticklabel_format(useOffset=False)

'''
#BLS
model       = BoxLeastSquares(time, (final_lc/mf))
durations   = np.linspace(0.05, 0.2, 50)
periodogram = model.autopower(durations, minimum_period=0.5, maximum_period=30)
idx         = np.argmax(periodogram.power)
period      = periodogram.period[idx]
t0          = periodogram.transit_time[idx]
ph          = (time-t0 + 0.5*period) % period - 0.5*period
print(period)

fig, ax = plt.subplots(figsize=[9,3])
ax.plot(ph, final_lc/mf, 'ok', ms=2)
ax.set_xlim(-.2, .2)
'''

plt.show()

inst   = np.repeat('TESS', len(time))
output = np.transpose([time, lkf[bidx].flux, lkf[bidx].flux_err, inst])
np.savetxt('TIC%d.dat' % args.TIC, output, fmt='%s')
