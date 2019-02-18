import __future__
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from everest.mathutils import SavGol
from transitleastsquares import transitleastsquares as TLS
from eveport import PLD
from lightkurve.lightcurve import TessLightCurve
from lightkurve.correctors import PLDCorrector
from lightkurve.search import search_tesscut
from lightkurve.targetpixelfile import TessTargetPixelFile
from utils import mask_planet
from autoap import generate_aperture, select_aperture
from photutils import MMMBackground, SExtractorBackground, Background2D, CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, BoxLeastSquares, BoxLeastSquares
from astropy.wcs import WCS
from astropy.io import fits

parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('TIC', type=int, help='TIC ID')
parser.add_argument('Sector', type=int, help='Sector')
parser.add_argument('--everest', action='store_true')

args = parser.parse_args()

cata   = '/Volumes/Felipe/TESS/all_targets_S%03d_v1.csv' % args.Sector
cata   = pd.read_csv(cata, comment='#')
cid    = cata['TICID'] == args.TIC
target = cata[cid]
print(target,'\n')

ra  = float(target['RA'])
dec = float(target['Dec'])
coord = SkyCoord(ra, dec, unit='deg')

if args.everest:
    from sklearn.neighbors import KDTree
    X    = np.transpose([cata['RA'], cata['Dec']])
    tree = KDTree(X)
    nd, ni = tree.query(X, k=11)
    ni = ni[:,1:]
    print(cata.iloc[ni[0]])

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

maskf = {'4':  (hdus[0][1].data['TIME'] < (2458419 - 2457000)) +
               ((hdus[0][1].data['TIME'] > (2458424 - 2457000)) *
               (hdus[0][1].data['TIME'] < 2458436.25 - 2457000)),
         }

ma = qual# & maskf['4']

time = hdus[0][1].data['TIME'][ma] + hdus[0][1].header['BJDREFI']
flux = hdus[0][1].data['FLUX'][ma]
errs = hdus[0][1].data['FLUX_ERR'][ma]
bkgs = np.zeros(len(flux))
lcfl = np.zeros(len(flux))
print(hdus[0][1].data.columns)

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
print(dap[bidx].sum(),'pixels in aperture')


'''
from eveport import rPLDYay, GetData
datita = GetData(hdus[0], args.TIC, dap[bidx])
print(datita)
emodel = rPLDYay(args.TIC, data=datita, debug=True, season=None, breakpoints=None, clobber=True)
'''

'''
aa = search_tesscut(coord).download()
print(aa)

corr = PLDCorrector(aa)
lc = corr.correct()#aperture_mask=dap[bidx])
lc = lc.flatten(polyorder=2, window_length=51)
lc.plot()
'''
#tmask = mask_planet(time, 2458327.6782456427, 4.086673341286014)

#flsa = SavGol(lkf[bidx].flux)
#med  = np.nanmedian(lkf[bidx].flux)
#MAD  = 1.4826 * np.nanmedian(np.abs(lkf[bidx].flux - med))
#tmask = np.abs(lkf[bidx].flux - med) < 10.*MAD

det_flux, det_err = PLD(time, flux, errs, lkf[bidx].flux, dap[bidx])#, mask=np.where(tmask)[0])
det_lc = TessLightCurve(time=time, flux=det_flux, flux_err=det_err)
det_lc = det_lc.flatten(polyorder=2, window_length=51)
#fig, ax = plt.subplots(figsize=[10,3])


aps    = CircularAperture([(x,y)], r=2.5)

fig1, ax1 = plt.subplots(figsize=[10,3], ncols=2)
ax1[0].matshow(np.log10(flux[4]))
ax1[0].matshow(dap[bidx], alpha=.2)
ax1[0].plot(x,y, '.r')
aps.plot(color='w', ax=ax1[0])
#ax1[1].matshow(bkgs[4])

#Phased
#pg  = lkf[bidx].remove_outliers(sigma=8).to_periodogram(method='bls', frequency_factor=25)
model  = TLS(time, det_lc.flux)
result = model.power(oversampling_factor=5)#, duration_grid_step=1.02)

'''
durations = np.linspace(0.05, 0.2, 100)# * u.day
model     = BoxLeastSquares(time, det_lc.flux)
result    = model.autopower(durations, frequency_factor=5.0, maximum_period=10.0)
idx       = np.argmax(result.power)

period = result.period[idx]
t0     = result.transit_time[idx]
dur    = result.duration[idx]
depth  = result.depth[idx]
snr    = result.depth_snr[idx]
print(period, t0, dur, depth)

ph = (time - t0 + 0.5*period) % period - 0.5*period
'''

print(result.period, result.T0, result.duration)

#ax1[1].plot(ph, det_lc.flux, 'ok', ms=2)
ax1[1].plot(result.folded_phase - .5, result.folded_y, 'ok', ms=2)
#ax1[1].set_xlim(-.2,.2)
#ax1[1].set_ylim(1.-2*result.depth, 1.001)

fig, ax = plt.subplots(figsize=[10,4])
ax.plot(time, lkf[bidx].flux, '-ok', ms=2, lw=1.5)
ax.plot(time, det_lc.flux, color='tomato', lw=1)
#ax.plot(time, mf, '-r', lw=1)
#for i in range(10):
#    ax.axvline(2458438.38600+i*4.05200)
ax.ticklabel_format(useOffset=False)


plt.show()

inst   = np.repeat('TESS', len(time))
output = np.transpose([time, lkf[bidx].flux, lkf[bidx].flux_err, inst])
np.savetxt('TIC%d.dat' % args.TIC, output, fmt='%s')
