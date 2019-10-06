import __future__
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import glob
import os
#from everest.mathutils import SavGol
from eveport import PLD
try:
    from lightkurve.lightcurve import TessLightCurve
    from lightkurve.search import search_tesscut
    from lightkurve.targetpixelfile import KeplerTargetPixelFile
except:
    os.system('rm ~/.astropy/config/*.cfg')
    from lightkurve.lightcurve import TessLightCurve
    from lightkurve.search import search_tesscut
    from lightkurve.targetpixelfile import KeplerTargetPixelFile

from utils import mask_planet, FFICut, pixel_border
from autoap import generate_aperture, select_aperture
from photutils import MMMBackground, SExtractorBackground
from astropy.utils.console import color_print
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import median_filter
from tess_stars2px import tess_stars2px_function_entry as ts2p

parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('TIC', type=float, nargs='+', help='TIC ID or RA DEC')
parser.add_argument('Sector', type=int, help='Sector')
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--size', type=int, default=21, help='TPF size')
parser.add_argument('--mask-transit', type=float, nargs=3, default=(None, None, None), help='Mask Transits, input: period, t0')
parser.add_argument('--everest', action='store_true')
parser.add_argument('--noplots', action='store_true')
parser.add_argument('--pld', action='store_true')
parser.add_argument('--psf', action='store_true')
parser.add_argument('--circ', action='store_true')
parser.add_argument('--norm', action='store_true')

args = parser.parse_args()
iP, it0, idur = args.mask_transit

if len(args.TIC) < 2:
    from astroquery.mast import Catalogs
    args.TIC = int(args.TIC[0])
    if os.path.isfile('TIC%d.dat' % args.TIC):
        import sys
        color_print('Skipping TIC %d' % args.TIC, 'lightred')
        sys.exit()
    #cata   = '../TIC_5.csv'# % args.Sector
    #cata   = pd.read_csv(cata, comment='#')
    #cid    = cata['TICID'] == args.TIC
    #target = cata[cid]
    color_print('TIC: ', 'lightcyan', '%d' % args.TIC, 'default')
    target = Catalogs.query_object('TIC %d' % args.TIC, radius=0.05, catalog='TIC')


    ra  = float(target[0]['ra'])
    dec = float(target[0]['dec'])
    #cam = int(target[0]['Camera'])
    #ccd = int(target[0]['CCD'])

else:
    ra, dec = args.TIC

color_print('RA Dec: ', 'lightcyan', '%f %f' % (ra, dec), 'default')

_, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)
cam = cam[0]
ccd = ccd[0]
#print(args.TIC, ra, dec, cam, ccd)


coord = SkyCoord(ra, dec, unit='deg')

'''
#if args.everest:
    from sklearn.neighbors import KDTree
    X    = np.transpose([cata['RA'], cata['Dec']])
    tree = KDTree(X)
    nd, ni = tree.query(X, k=11)
    ni = ni[:,1:]
    print(cata.iloc[ni[0]])
'''


if args.folder is not None:
    #Offline mode
    fnames  = np.sort(glob.glob(args.folder + '*s%04d-%d-%d*.fits' % (args.Sector, cam, ccd)))
    print(args.Sector, cam, ccd)
    print(fnames[5])
    fhdr    = fits.getheader(fnames[5], 1)
    ffis    = args.folder + 'TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, cam, ccd)

    w   = WCS(fhdr)
    x,y = w.all_world2pix(ra, dec, 0)
    print(x,y)

    '''
    fig, ax = plt.subplots()
    ax.matshow(np.log10(fits.getdata(fnames[1])))
    ax.plot(x,y,'.r')
    plt.show()
    '''

    allhdus = FFICut(ffis, y, x, args.size)

else:
    #Online mode
    color_print('Querying MAST...', 'lightcyan')
    allhdus = search_tesscut(coord, sector=args.Sector).download(cutout_size=args.size, download_dir='.')
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

#Star position
x,y = w.all_world2pix(ra, dec, 0)

#Background
for i,f in enumerate(flux):
    sigma_clip = SigmaClip(sigma=3)
    bkg        = MMMBackground(sigma_clip=sigma_clip)
    bkgs[i]    = bkg.calc_background(f)

if args.psf:
    flux_psf = np.zeros(len(flux))

    import tensorflow as tf
    from vaneska.models import Gaussian
    from tqdm import tqdm

    nstars   = 1
    flux_psf = tf.Variable(np.ones(nstars)*np.nanmax(flux[0]), dtype=tf.float64)
    bkg_psf  = tf.Variable(bkgs[0], dtype=tf.float64)
    xshift   = tf.Variable(0.0, dtype=tf.float64)
    yshift   = tf.Variable(0.0, dtype=tf.float64)

    gaussian = Gaussian(shape=flux.shape[1:], col_ref=0, row_ref=0)

    a = tf.Variable(initial_value=1., dtype=tf.float64)
    b = tf.Variable(initial_value=0., dtype=tf.float64)
    c = tf.Variable(initial_value=1., dtype=tf.float64)

    if nstars == 1:
        mean = gaussian(flux_psf, x+xshift, y+yshift, a, b, c)
    else:
        mean = [gaussian(flux[j], x+xshift, y+yshift, a, b, c) for j in range(nstars)]

    mean += bkg_psf

    psf_data = tf.placeholder(dtype=tf.float64, shape=flux[0].shape)
    bkgval   = tf.placeholder(dtype=tf.float64)

    nll = tf.reduce_sum(tf.squared_difference(mean, psf_data))

    var_list = [flux_psf, xshift, yshift, a, b, c, bkg_psf]
    grad     = tf.gradients(nll, var_list)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    var_to_bounds = {flux_psf: (0, np.infty),
                     xshift: (-1.0, 1.0),
                     yshift: (-1.0, 1.0),
                     a: (0, np.infty),
                     b: (0, np.infty),
                     c: (0, np.infty)
                    }

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(nll, var_list, method='TNC', tol=1e-4, var_to_bounds=var_to_bounds)

    fout   = np.zeros((len(flux), nstars))
    bkgout = np.zeros(len(flux))

    for i in tqdm(range(len(flux))):
        optim = optimizer.minimize(session=sess, feed_dict={psf_data:flux[i], bkgval:bkgs[i]})
        fout[i] = sess.run(flux_psf)
        bkgout[i] = sess.run(bkg_psf)

    sess.close()

    psf_flux = fout[:,0]
    psf_bkg = bkgout

    #Lightkurves
    lks = TessLightCurve(time=time, flux=psf_flux)#, flux_err=flux_psf_err)
    lkf = lks.flatten(polyorder=2, window_length=91) if args.norm else lks

else:
    #DBSCAN Aperture
    x = x - int(x) + args.size//2
    y = y - int(y) + args.size//2

    if not args.circ:
        daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,3,5,7,9,11,13,15]]
        dap  = np.array([select_aperture(d, x, y) for d in daps])
    else:
        XX, YY = np.ogrid[:args.size, :args.size]
        dap    = [np.sqrt((XX-y)**2 + (YY-x)**2) < i for i in range(1,5)]

    #Aperture photometry
    lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
    lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

    #Lightkurves
    lks = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]
    mfs = [median_filter(lk.flux, size=55) for lk in lks]
    lkf = [lk.flatten(polyorder=2, window_length=85) for lk in lks] if args.norm else lks
    #lkf = [TessLightCurve(time=time, flux=lcfl[i]/mfs[i], flux_err=lcer[i]/mfs[i]) for i in range(len(lcfl))] if args.norm else lks

    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    print('Best lk:', bidx)
    print(dap[bidx].sum(),'pixels in aperture')
    lkf  = lkf[bidx]

    #PLD?
    if args.pld:
        #det_flux, det_err = PLD(time, flux, errs, lkf[bidx].flux, dap[bidx], mask=mask, n=8)
        pld_flux = PLD(flux, dap[bidx], lkf.flux)
        lkf = TessLightCurve(time=time, flux=lkf.flux - pld_flux + np.nanmedian(lkf.flux), flux_err=lkf.flux_err)
        #det_lc = det_lc.flatten(polyorder=2, window_length=51)



if iP is None:
    mask = np.ones(time.size).astype(bool)
else:
    mask = mask_planet(time, it0, iP, dur=idur)

if not args.noplots:
    #aps    = CircularAperture([(x,y)], r=2.5)
    fig1 = plt.figure(figsize=[16,3], dpi=120)
    gs   = gridspec.GridSpec(2, 2, width_ratios=[1,5])#, height_ratios=[1,1])

    ax0 = plt.subplot(gs[1,1])
    ax0.plot(time, bkgs, '.k', ms=2)

    ax1 = plt.subplot(gs[:,0])
    ax1.matshow(np.log10(flux[0]), cmap='YlGnBu_r', aspect='equal')

    if not args.psf:
        xm, ym = pixel_border(dap[bidx])
        for xi,yi in zip(xm, ym):
            ax1.plot(xi, yi, color='lime', lw=1.25)

    #ax1[0].matshow(dap[bidx], alpha=.2)
    ax1.plot(x,y, '.r')
    #aps.plot(color='w', ax=ax1[0])
    #ax1[1].matshow(bkgs[4])

    ax = plt.subplot(gs[0,1])
    ax.plot(time, lkf.flux, '-ok', ms=2, lw=1.5)
    ax.set_ylabel(r'Flux  (e-/s)', fontweight='bold')
    ax.set_xlabel(r'BJD', fontweight='bold')
    #ax.plot(time[~mask], lkf[bidx].flux[~mask], 'oc', ms=4, alpha=.9)
    #if args.pld:
    #    ax.plot(time, det_lc.flux*np.nanmedian(lkf.flux)/np.nanmedian(det_lc.flux), color='tomato', lw=.66)
    ax.ticklabel_format(useOffset=False)


    fig1.tight_layout()
    plt.show()

inst   = np.repeat('TESS', len(time))
output = np.transpose([time, lkf.flux, lkf.flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
np.savetxt('TIC%s.dat' % (args.TIC), output, fmt='%s')

if args.folder is None:
    os.system('rm tesscut/*%s*' % ra)

print('Done!\n')
#output = np.transpose([time, det_lc.flux, det_lc.flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d_det.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
