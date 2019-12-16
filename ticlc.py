import __future__
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
from eveport import PLD, PLD2
from lightkurve.lightcurve import TessLightCurve
from utils import mask_planet, FFICut, pixel_border
from autoap import generate_aperture, select_aperture
from photutils import MMMBackground
from astropy.utils.console import color_print
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, mad_std
from astropy.wcs import WCS
from astropy.io import fits
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
parser.add_argument('--flatten', action='store_true')
parser.add_argument('--pixlcs', action='store_true')
parser.add_argument('--pngstamp', action='store_true')

args = parser.parse_args()
iP, it0, idur = args.mask_transit

plt.rc('font', family='serif')

if len(args.TIC) < 2:
    from astroquery.mast import Catalogs
    args.TIC = int(args.TIC[0])
    if os.path.isfile('TIC%d_%02d.dat' % (args.TIC, args.Sector)):
        import sys
        color_print('Skipping TIC %d' % args.TIC, 'lightred')
        sys.exit()

    color_print('TIC: ', 'lightcyan', args.TIC, 'default')
    target = Catalogs.query_object('TIC %d' % args.TIC, radius=0.05, catalog='TIC')


    ra  = float(target[0]['ra'])
    dec = float(target[0]['dec'])

else:
    ra, dec = args.TIC

color_print('RA: ', 'lightcyan', ra, 'default', '\tDec: ', 'lightcyan', dec, 'default')

_, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)
cam = cam[0]
ccd = ccd[0]


coord = SkyCoord(ra, dec, unit='deg')

#Offline mode
if args.folder is not None:
    color_print('Extracting data from FFIs...', 'lightcyan')
    fnames  = np.sort(glob.glob(args.folder + 'tess*s%04d-%d-%d*ffic.fits' % (args.Sector, cam, ccd)))
    fhdr    = fits.getheader(fnames[5], 1)
    ffis    = args.folder + 'TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, cam, ccd)

    w   = WCS(fhdr)
    x,y = w.all_world2pix(ra, dec, 0)

    hdus = FFICut(ffis, y, x, args.size).hdu

#Online mode
else:
    from lightkurve.search import search_tesscut

    color_print('Querying MAST...', 'lightcyan')
    hdus = search_tesscut(coord, sector=args.Sector).download(cutout_size=args.size, download_dir='.').hdu
    cam     = hdus[2].header['CAMERA']
    ccd     = hdus[2].header['CCD']
    w       = WCS(hdus[2].header)
    hdus[1].data['TIME'] += hdus[1].header['BJDREFI']

    x,y = w.all_world2pix(ra, dec, 0)

color_print('Sector: ', 'lightcyan', args.Sector, 'default',
            '\tCamera: ', 'lightcyan', cam, 'default',
            '\tCCD: ', 'lightcyan', ccd, 'default')

color_print('Pos X: ', 'lightcyan', x, 'default', '\tPos Y: ', 'lightcyan', y, 'default')

#Data type
ma = hdus[1].data['QUALITY'] == 0

time = hdus[1].data['TIME'][ma]
flux = hdus[1].data['FLUX'][ma]
errs = hdus[1].data['FLUX_ERR'][ma]
bkgs = np.zeros(len(flux))
berr = np.zeros(len(flux))

print(flux.shape)

#Background
for i,f in enumerate(flux):
    sigma_clip = SigmaClip(sigma=3)
    bkg        = MMMBackground(sigma_clip=sigma_clip)
    bkgs[i]    = bkg.calc_background(f)
    berr[i]    = mad_std(f)

if args.psf:
    import tensorflow as tf
    from vaneska.models import Gaussian
    from tqdm import tqdm

    nstars   = 1
    flux_psf = tf.Variable(np.ones(nstars)*np.nanmax(flux[0]), dtype=tf.float64)
    ferr_psf = tf.Variable(np.ones(nstars)*np.nanmax(errs[0]), dtype=tf.float64)
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
    psf_derr  = tf.placeholder(dtype=tf.float64, shape=errs[0].shape)
    bkgval   = tf.placeholder(dtype=tf.float64)

    #nll = tf.reduce_sum(tf.squared_difference(mean, psf_data))
    nll = tf.reduce_sum(tf.truediv(tf.squared_difference(mean, psf_data), psf_derr))

    var_list = [flux_psf, xshift, yshift, a, b, c, bkg_psf]
    grad     = tf.gradients(nll, var_list)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    var_to_bounds = {flux_psf: (0, np.infty),
                     xshift: (-1.0, 1.0),
                     yshift: (-1.0, 1.0),
                     a: (0, np.infty),
                     b: (-0.5, 0.5),
                     c: (0, np.infty)
                    }

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(nll, var_list, method='TNC', tol=1e-4, var_to_bounds=var_to_bounds)

    fout   = np.zeros((len(flux), nstars))
    ferr   = np.zeros((len(errs), nstars))
    bkgout = np.zeros(len(flux))

    for i in tqdm(range(len(flux))):
        optim = optimizer.minimize(session=sess, feed_dict={psf_data:flux[i], psf_derr:errs[i], bkgval:bkgs[i]})
        fout[i] = sess.run(flux_psf)
        ferr[i] = sess.run(ferr_psf)
        bkgout[i] = sess.run(bkg_psf)


    sess.close()

    psf_flux = fout[:,0]
    psf_ferr = ferr[:,0]
    psf_bkg  = bkgout

    #Lightkurves
    lks = TessLightCurve(time=time, flux=psf_flux)
    lkf = lks.flatten(polyorder=2, window_length=91) if args.flatten else lks

    if args.norm:
        lkf.flux = lkf.flux / np.nanmedian(lkf.flux)

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
    lkf = [lk.flatten(polyorder=2, window_length=85) for lk in lks] if args.norm else lks

    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    lkf  = lkf[bidx]

    color_print('Aperture chosen: ', 'lightcyan', str(bidx+1) + 'px radius' if args.circ else 'No. ' + str(bidx), 'default',
                '\tNumber of pixels inside: ', 'lightcyan', dap[bidx].sum(), 'default')


    #PLD?
    if args.pld:
        if iP is None:
            mask = np.ones(time.size).astype(bool)
        else:
            mask = mask_planet(time, it0, iP, dur=idur)

        #det_flux, det_err = PLD2(time, flux, errs, lkf[bidx].flux, dap[bidx], mask=mask, n=5)
        #lkf = TessLightCurve(time=time, flux=det_flux, flux_err=lkf.flux_err)
        pld_flux = PLD(flux, dap[bidx], lkf.flux)
        lkf = TessLightCurve(time=time, flux=lkf.flux - pld_flux + np.nanmedian(lkf.flux), flux_err=lkf.flux_err)
        #det_lc = det_lc.flatten(polyorder=2, window_length=51)


if not args.noplots:
    if args.pixlcs:
        pfig, pax = plt.subplots(figsize=[8,8])
        tmin = np.nanmin(time)
        tlen = np.nanmax(time) - tmin
        tnor = (time - tmin) / tlen

        pax.matshow(np.log10(np.nanmedian(flux[::10], axis=0)), cmap='Blues_r')

        theflux = flux - bkgs[:,None,None]
        for i in range(args.size):
            for j in range(args.size):
                fmin = np.nanmin(theflux[:,i,j])
                flen = np.nanmax(theflux[:,i,j]) - fmin
                fnor = (theflux[:,i,j] - fmin) / flen

                pax.plot(j+tnor-0.5, i+fnor-0.5, '-', color='lime' if dap[bidx][i,j] else 'tomato', lw=.1)

        pax.set_xticks(np.arange(-.5, args.size, 1), minor=True)
        pax.set_yticks(np.arange(-.5, args.size, 1), minor=True)
        pax.grid(which='minor', zorder=99)
        pfig.tight_layout()

    fig1 = plt.figure(figsize=[12,3], dpi=120)
    gs   = gridspec.GridSpec(2, 2, width_ratios=[1,5])#, height_ratios=[1,1])

    ax0 = plt.subplot(gs[1,1])
    ax0.errorbar(time, bkgs, yerr=berr, fmt='ok', ms=2)
    #ax0.plot(time, bkgs if not args.psf else bkgout, '.k', ms=2)
    ax0.set_title('Background')
    ax0.set_ylabel(r'Flux  (e-/s)', fontweight='bold')
    ax0.set_xlabel(r'BJD', fontweight='bold')

    ax1 = plt.subplot(gs[:,0])
    ax1.matshow(np.log10(flux[0]), cmap='YlGnBu_r', aspect='equal')

    if not args.psf:
        xm, ym = pixel_border(dap[bidx])
        for xi,yi in zip(xm, ym):
            ax1.plot(xi, yi, color='lime', lw=1.25)

    #ax1[0].matshow(dap[bidx], alpha=.2)
    #ax1.plot(x,y, '.r')
    #aps.plot(color='w', ax=ax1[0])
    #ax1[1].matshow(bkgs[4])

    ax = plt.subplot(gs[0,1], sharex=ax0)
    ax.plot(time, lkf.flux, '-ok', ms=2, lw=1.5)
    ax.set_ylabel(r'Flux  (e-/s)', fontweight='bold')
    #ax.plot(time[~mask], lkf[bidx].flux[~mask], 'oc', ms=4, alpha=.9)
    #if args.pld:
    #    ax.plot(time, det_lc.flux*np.nanmedian(lkf.flux)/np.nanmedian(det_lc.flux), color='tomato', lw=.66)
    ax.ticklabel_format(useOffset=False)
    ax.set_title('Light curve')


    fig1.tight_layout()
    plt.show()

if args.pngstamp:
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.Purples(np.arange(plt.cm.Purples.N))
    my_cmap[:,0:3] *= 0.95
    my_cmap = ListedColormap(my_cmap)

    sfig, sax = plt.subplots(figsize=[2.5,2.5])
    sax.matshow(np.log10(np.nanmedian(flux[::10], axis=0)), cmap=my_cmap)

    xm, ym = pixel_border(dap[bidx])
    for xi,yi in zip(xm, ym):
        sax.plot(xi, yi, color='#FF0043', lw=1.5)

    sax.text(0.95, 0.95, 'Sector %02d\nCCD: %d\nCam: %d' % (args.Sector, ccd, cam), ha='right', va='top', transform=sax.transAxes, color='#FF0043', size='x-large')

    plt.axis('off')
    sfig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    sfig.savefig('TIC%s_%02d_st.png' % (args.TIC, args.Sector), dpi=240)

inst   = np.repeat('TESS', len(time))
output = np.transpose([time, lkf.flux, lkf.flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
np.savetxt('TIC%s_%02d.dat' % (args.TIC, args.Sector), output, fmt='%s')

if args.folder is None:
    os.system('rm tesscut/*%s*' % ra)

color_print('\nDone!\n', 'lightgreen')
#output = np.transpose([time, det_lc.flux, det_lc.flux_err, inst])
#np.savetxt('TIC%d_s%04d-%d-%d_det.dat' % (args.TIC, args.Sector, cam, ccd), output, fmt='%s')
