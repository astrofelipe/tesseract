import __future__
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import everest
from everest.mathutils import SavGol
from transitleastsquares import transitleastsquares as TLS
from eveport import PLD
from lightkurve.lightcurve import TessLightCurve
from lightkurve.correctors import PLDCorrector
from lightkurve.search import search_tesscut
from lightkurve.targetpixelfile import KeplerTargetPixelFile
from utils import mask_planet, FFICut
from autoap import generate_aperture, select_aperture
from photutils import MMMBackground, SExtractorBackground, Background2D, CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, BoxLeastSquares, BoxLeastSquares
from astropy.wcs import WCS
from astropy.io import fits

parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('TIC', type=float, nargs='+', help='TIC ID or RA DEC')
parser.add_argument('Sector', type=int, help='Sector')
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--size', type=int, help='TPF size')
parser.add_argument('--mask-transit', type=float, nargs=3, default=(None, None, None), help='Mask Transits, input: period, t0')
parser.add_argument('--everest', action='store_true')
parser.add_argument('--noplots', action='store_true')

args = parser.parse_args()
iP, it0, idur = args.mask_transit

if len(args.TIC) < 2:
    args.TIC = int(args.TIC[0])
    cata   = '../TIC_5.csv'# % args.Sector
    cata   = pd.read_csv(cata, comment='#')
    cid    = cata['TICID'] == args.TIC
    target = cata[cid]
    print(target,'\n')


    ra  = float(target['RA'])
    dec = float(target['Dec'])
    #cam = int(target['Camera'])
    #ccd = int(target['CCD'])

else:
    ra, dec = args.TIC
    print(ra, dec)

coord = SkyCoord(ra, dec, unit='deg')

if args.everest:
    from sklearn.neighbors import KDTree
    X    = np.transpose([cata['RA'], cata['Dec']])
    tree = KDTree(X)
    nd, ni = tree.query(X, k=11)
    ni = ni[:,1:]
    print(cata.iloc[ni[0]])


if args.folder is not None:
    #Offline mode
    fnames  = np.sort(glob.glob(args.folder + '*s%04d-%d-%d-*ffic.fits' % (args.Sector, cam, ccd)))
    allhdus, w = FFICut(fnames, ra, dec, 21)

else:
    #Online mode
    allhdus = search_tesscut(coord, sector=args.Sector).download(cutout_size=21)
    w       = WCS(allhdus.hdu[2].header)

hdus  = allhdus.hdu

#Data type
qual = hdus[1].data['QUALITY'] == 0

maskf = {'4':  (hdus[1].data['TIME'] < (2458419 - 2457000)) +
               ((hdus[1].data['TIME'] > (2458424 - 2457000)) *
               (hdus[1].data['TIME'] < 2458436.25 - 2457000)),
         }

ma = qual# & maskf['4']

time = hdus[1].data['TIME'][ma] + hdus[1].header['BJDREFI']
flux = hdus[1].data['FLUX'][ma]
errs = hdus[1].data['FLUX_ERR'][ma]
bkgs = np.zeros(len(flux))
lcfl = np.zeros(len(flux))
#print(hdus[1].data.columns)
#print(hdus[1].data['TIME'][:10])

#Star position
x,y = w.all_world2pix(ra, dec, 0)
#print(x,y)

#Background
for i,f in enumerate(flux):
    sigma_clip = SigmaClip(sigma=3)
    bkg        = MMMBackground(sigma_clip=sigma_clip)
    bkgs[i]    = bkg.calc_background(f)

#DBSCAN Aperture
daps = [generate_aperture(flux- bkgs[:,None,None], n=i) for i in [1,3,5,7,9,11,13,15]]
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

#tmask = mask_planet(time, 2458327.6782456427, 4.086673341286014)

#flsa = SavGol(lkf[bidx].flux)
#med  = np.nanmedian(lkf[bidx].flux)
#MAD  = 1.4826 * np.nanmedian(np.abs(lkf[bidx].flux - med))
#tmask = np.abs(lkf[bidx].flux - med) < 10.*MAD

if iP is None:
    mask = np.ones(time.size).astype(bool)
else:
    mask = mask_planet(time, it0, iP, dur=idur)


det_flux, det_err = PLD(time, flux, errs, lkf[bidx].flux, dap[bidx], mask=mask)
det_lc = TessLightCurve(time=time, flux=det_flux, flux_err=det_err)
det_lc = det_lc.flatten(polyorder=2, window_length=51)


if not args.noplots:
    aps    = CircularAperture([(x,y)], r=2.5)

    fig1, ax1 = plt.subplots(figsize=[10,3], ncols=2)
    ax1[0].matshow(np.log10(flux[0]), cmap='YlGnBu_r')
    ax1[0].matshow(dap[bidx], alpha=.2)
    ax1[0].plot(x,y, '.r')
    aps.plot(color='w', ax=ax1[0])
    #ax1[1].matshow(bkgs[4])

    '''
    #Phased
    model  = TLS(time, det_lc.flux)
    result = model.power(oversampling_factor=5)#, duration_grid_step=1.02)

    print(result.period, result.T0, result.duration)

    ax1[1].plot(result.folded_phase - .5, result.folded_y, 'ok', ms=2)
    '''

    fig, ax = plt.subplots(figsize=[10,4])
    ax.plot(time, lkf[bidx].flux, '-ok', ms=2, lw=1.5)
    ax.plot(time[~mask], lkf[bidx].flux[~mask], 'oc', ms=4, alpha=.9)
    ax.plot(time, det_lc.flux, color='tomato', lw=1)
    ax.ticklabel_format(useOffset=False)


    plt.show()

inst   = np.repeat('TESS', len(time))
output = np.transpose([time, lkf[bidx].flux, lkf[bidx].flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
np.savetxt('TIC%s.dat' % (args.TIC), output, fmt='%s')

#output = np.transpose([time, det_lc.flux, det_lc.flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d_det.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
