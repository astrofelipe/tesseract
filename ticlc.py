import __future__
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
import astropy.units as u
from matplotlib import rcParams
from eveport import PLD, PLD2
from lightkurve.lightcurve import TessLightCurve
from utils import mask_planet, FFICut, pixel_border, dilution_factor
from autoap import generate_aperture, select_aperture
from photutils import SExtractorBackground
from astropy.utils.console import color_print
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, mad_std
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import time_support
from tess_stars2px import tess_stars2px_function_entry as ts2p

parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('TIC', type=float, nargs='+', help='TIC ID or RA DEC')
parser.add_argument('Sector', type=int, help='Sector')
parser.add_argument('--folder', type=str, default=None, help='Uses local stored FFIs (stacked in hdf5 format, see FFI2h5.py)')
parser.add_argument('--size', type=int, default=21, help='TPF size')
parser.add_argument('--mask-transit', type=float, nargs=3, default=(None, None, None), help='Mask Transits, input: period, t0')
#parser.add_argument('--everest', action='store_true')
parser.add_argument('--noplots', action='store_true', help="Doesn't show plots")
parser.add_argument('--pld', action='store_true', help='Pixel level decorrelation (Experimental)')
parser.add_argument('--psf', action='store_true', help='Experimental PSF (Eleanor)')
parser.add_argument('--pca', action='store_true', help='Removes background and some sistematics using PCA')
#parser.add_argument('--prf', action='store_true')
parser.add_argument('--circ', type=float, default=-1, help='Forces circular apertures')
parser.add_argument('--manualap', type=str, const=-1, nargs='?', help='Manual aperture input (add filename or interactive picking if not)')
parser.add_argument('--norm', action='store_true', help='Divides the flux by the median')
parser.add_argument('--flatten', action='store_true', help='Detrends and normalizes the light curve')
parser.add_argument('--window-length', type=float, default=1.5)
parser.add_argument('--cleaner', action='store_true', help='Removes saturated background times (Warning: chosen by eye)')
parser.add_argument('--pixlcs', action='store_true', help='Shows light curves per pixel')
parser.add_argument('--pngstamp', type=str, default=None, help='Saves the postage stamp as png (input "full" or "minimal")')
parser.add_argument('--pngzoom', type=float, default=1, help='Zoom for --pngstamp')
parser.add_argument('--pngtitle', type=str, default=None, help='Overrides TIC number for the title (useful for TOIs)')
parser.add_argument('--gaia', action='store_true', help='Shows Gaia sources on stamps')
parser.add_argument('--maxgaiamag', type=float, default=16, help='Maximum Gaia magnitude to consider')
parser.add_argument('--cam', type=int, default=None, help='Overrides camera number')
parser.add_argument('--ccd', type=int, default=None, help='Overrides CCD number')
parser.add_argument('--cmap', type=str, default='YlGnBu_r', help='Colormap to use')
parser.add_argument('--animation', type=str, default='none', help='Saves a movie of the cutout + lightcurve ("talk" and "outreach" options are available)')
parser.add_argument('--overwrite', action='store_false', help='Overwrites existing filename')

args = parser.parse_args()
iP, it0, idur = args.mask_transit

plt.rcParams['font.family']     = 'serif'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

if len(args.TIC) < 2:
    from astroquery.mast import Catalogs

    args.TIC = int(args.TIC[0])
    targettitle = 'TIC %d' % args.TIC if args.pngtitle is None else args.pngtitle

    if args.overwrite and os.path.isfile('TIC%d_%02d.dat' % (args.TIC, args.Sector)):
        import sys
        color_print('Skipping TIC %d' % args.TIC, 'lightred')
        sys.exit()

    color_print('TIC: ', 'lightcyan', str(args.TIC), 'default')
    target = Catalogs.query_object('TIC %d' % args.TIC, radius=0.05, catalog='TIC')


    ra  = float(target[0]['ra'])
    dec = float(target[0]['dec'])

else:
    ra, dec = args.TIC

color_print('RA: ', 'lightcyan', str(ra), 'default', '\tDec: ', 'lightcyan', str(dec), 'default')

_, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)
cam = cam[0]
ccd = ccd[0]

if args.ccd is not None:
    cam = args.cam
    ccd = args.ccd


coord = SkyCoord(ra, dec, unit='deg')

#Offline mode
if args.folder is not None:
    color_print('Extracting data from FFIs...', 'lightcyan')
    fnames  = np.sort(glob.glob(args.folder + 'tess*s%04d-%d-%d*ffic.fits' % (args.Sector, cam, ccd)))
    fhdr    = fits.getheader(fnames[0], 1)
    ffis    = args.folder + 'TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, cam, ccd)

    w   = WCS(fhdr)
    x,y = w.all_world2pix(ra, dec, 0)

    hdus = FFICut(ffis, y, x, args.size).hdu

    if args.pld:
        hdu_pld = FFICut(ffis, y, x, 2*args.size).hdu

    ex  = int(x-10.5)
    ey  = int(y-10.5)
    row = y
    column = x
    x,y = x-ex, y-ey

#Online mode
else:
    from lightkurve.search import search_tesscut

    color_print('Querying MAST...', 'lightcyan')
    hdus = search_tesscut(coord, sector=args.Sector).download(cutout_size=args.size, download_dir='.').hdu
    if args.pld:
        tpf_pld = search_tesscut(coord, sector=args.Sector).download(cutout_size=args.size, download_dir='.')
        hdus    = tpf_pld.hdu
    cam     = hdus[2].header['CAMERA']
    ccd     = hdus[2].header['CCD']
    row     = hdus[1].header['2CRV5P']
    column  = hdus[1].header['1CRV5P']
    w       = WCS(hdus[2].header)
    hdus[1].data['TIME'] += hdus[1].header['BJDREFI']

    x,y = w.all_world2pix(ra, dec, 0)
    x  += 0.5
    y  += 0.5

color_print('Sector: ', 'lightcyan', str(args.Sector), 'default',
            '\tCamera: ', 'lightcyan', str(cam), 'default',
            '\tCCD: ', 'lightcyan', str(ccd), 'default')

color_print('Pos X: ', 'lightcyan', str(x), 'default', '\tPos Y: ', 'lightcyan', str(y), 'default')
color_print('CCD Row: ', 'lightcyan', str(row+x), 'default', '\tCCD Column: ', 'lightcyan', str(column+y), 'default')

#Data type
ma = hdus[1].data['QUALITY'] == 0

time = hdus[1].data['TIME'][ma]
flux = hdus[1].data['FLUX'][ma]
errs = hdus[1].data['FLUX_ERR'][ma]
bkgs = np.zeros(len(flux))
berr = np.zeros(len(flux))

if args.manualap is not None:
    apix2, apix1        = np.genfromtxt(args.manualap, unpack=True).astype(int)
    theap               = np.zeros(flux[0].shape).astype(bool)
    theap[apix1, apix2] = True

#Background
for i,f in enumerate(flux):
    sigma_clip = SigmaClip(sigma=1)
    bkg        = SExtractorBackground(sigma_clip=sigma_clip)
    bkgs[i]    = bkg.calc_background(f) if args.manualap is None else bkg.calc_background(f[~theap])
    mad_bkg    = mad_std(f)
    berr[i]    = (3*1.253 - 2)*mad_bkg/np.sqrt(f.size)

#PRF from lightkurve
#if args.prf:
#    print(type(hdus))

#PSF routine, taken from Eleanor
#Works (?) but doesn't return errors and only fits one gaussian
if args.psf:
    import tensorflow as tf #tensorflow 1
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
    lkf = TessLightCurve(time=time, flux=psf_flux)

    if args.flatten:
        from wotan import flatten
        flat_flux = flatten(lkf.time, lkf.flux, window_length=args.window_length, method='biweight', return_trend=False)
        print(flat_flux)
        #lkf = lks.flatten(polyorder=2, window_length=51, niters=5) if args.flatten else lks
        lkf.flux = flat_flux

else:
    #DBSCAN Aperture
    #x = x - int(x) + args.size//2
    #y = y - int(y) + args.size//2

    if args.manualap == -1:
        #Choose pixels manually
        print('Coming soon!')
    elif args.manualap is not None:
        dap   = [theap]
    elif args.circ==-1:
        #K2P2 (clustering, watershed) algorithm
        daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,3,5,7,9,11,13,15]]
        dap  = np.array([select_aperture(d, x, y) for d in daps])
    else:
        #Circular aperture (no resampling)
        YY, XX = np.ogrid[:args.size, :args.size]
        caps   = [args.circ] if args.circ!=0 else range(1,6)
        dap    = [np.sqrt((XX-x+.5)**2 + (YY-y+.5)**2) <= i for i in caps]

    #Aperture photometry
    lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
    lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

    #Lightkurves
    lkf = [TessLightCurve(time=time, flux=lcfl[i], flux_err=np.sqrt(lcer[i]**2 + berr**2)) for i in range(len(lcfl))]

    #FLAT
    if args.flatten:
        from wotan import flatten
        for lk in lkf:
            lk.flux, trend = flatten(lk.time, lk.flux, window_length=args.window_length, method='rspline', return_trend=True)
            lk.flux_err    /= trend


    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    lkf  = lkf[bidx]

    color_print('\nAperture chosen: ', 'lightcyan', str(bidx+1) + 'px radius' if args.circ==0 else 'No. ' + str(bidx), 'default',
                '\tNumber of pixels inside: ', 'lightcyan', str(dap[bidx].sum()), 'default')


    #PCA
    if args.pca:
        import lightkurve
        regressors = flux[:, ~dap[bidx]]
        dm         = lightkurve.DesignMatrix(regressors, name='regressors')

        dm = dm.pca(5)
        dm = dm.append_constant()

        corrector = lightkurve.RegressionCorrector(lkf)
        corr_flux = corrector.correct(dm)
        lkf = lkf - corrector.model_lc + np.percentile(corrector.model_lc.flux, 5)
        #lkf.flux  = lkf.flux - corrector.model_lc + np.percentile(corrector.model_lc.flux, 5)

    #PLD?
    elif args.pld:
        '''
        if iP is None:
            mask = np.ones(time.size).astype(bool)
        else:
            mask = mask_planet(time, it0, iP, dur=idur)

        flux_pld -= bkgs[:,None,None]
        pldflsum = np.nansum(flux_pld, axis=0)
        pldthr   = mad_std(pldflsum)
        pldthm   = pldflsum > 3*pldthr

        pld_flux = PLD(flux_pld, pldthm, lkf.flux)
        lkf = TessLightCurve(time=time, flux=lkf.flux - pld_flux + np.nanmedian(lkf.flux), flux_err=lkf.flux_err)
        '''
        from lightkurve.correctors import PLDCorrector
        tpf_pld.hdu[1].data['FLUX'][ma] -= bkgs[:,None,None]
        tpf_pld.hdu[1].data['FLUX_ERR'][ma] -= np.sqrt(tpf_pld.hdu[1].data['FLUX_ERR'][ma]**2 + berr[:,None,None]**2)
        corr = PLDCorrector(tpf_pld)
        lkf   = corr.correct(aperture_mask = dap[bidx], pld_aperture_mask='threshold', pld_order=3, use_gp=True)


#NORM
if args.norm:
    lkf.flux_err /= np.nanmedian(lkf.flux)
    lkf.flux /= np.nanmedian(lkf.flux)

if args.cleaner:
    from cleaner import cleaner
    omask = cleaner(lkf.time, lkf.flux)
    lkf.time = lkf.time[~omask]
    lkf.flux = lkf.flux[~omask]
    lkf.flux_err = lkf.flux_err[~omask]

#Gaia sources and dilution factor
if args.gaia:
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1

    gaiawh = u.Quantity(21*args.size*np.sqrt(2)/2, u.arcsec)
    gaiar  = Gaia.cone_search_async(coord, gaiawh).get_results()
    #gaiar  = Gaia.query_object_async(coord, width=gaiawh, height=gaiawh)

    gma = gaiar['phot_rp_mean_mag'] < args.maxgaiamag*u.mag
    gra, gdec = gaiar['ra'][gma], gaiar['dec'][gma]
    grpmag    = gaiar['phot_rp_mean_mag'][gma]
    gsep      = gaiar['dist'][gma]*3600
    gaiar     = gaiar[gma]


    gx, gy = w.all_world2pix(gra, gdec, 0) + (np.ones(2)*.5)[:,None]
    print(gx,gy)

    gma2   = (gx >= 0) & (gx <= args.size) & (gy >= 0) & (gy <= args.size)
    gx, gy = gx[gma2], gy[gma2]
    gsep   = gsep[gma2]
    grpmag = grpmag[gma2]

    color_print('Nearby sources:\n', 'cyan')
    gaiaresume = gaiar#[gma2]
    gaiaresume['label'] = np.arange(len(grpmag))
    gresume = gaiaresume['label', 'designation', 'ra', 'dec', 'phot_rp_mean_mag', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'dist']
    print(gresume)

    didx = np.array((gx,gy)).astype(int)
    iidx = (dap[bidx])[didx[1], didx[0]]

    minaldi = np.argmin(np.abs(grpmag[1:] - grpmag[0]))
    minapdi = np.argmin(np.abs(grpmag[iidx][1:] - grpmag[0]))

    minalldif = grpmag[0] - grpmag[1:][minaldi]
    minapdif  = grpmag[0] - grpmag[iidx][1:][minapdi]
    minallsep = gsep[1:][minaldi]
    minapsep  = gsep[iidx][1:][minapdi]

    dfac = dilution_factor(grpmag[0], grpmag[iidx][1:], gsep[iidx][1:])
    color_print('\nAdditional sources inside aperture: ', 'cyan', str(iidx.sum()-1), 'default')
    color_print('Dilution factor: ', 'cyan', str(dfac), 'default')
    color_print('Min mag difference inside aperture: ', 'cyan', '%f mag (%f arsec)' % (minapdif, minapsep), 'default')
    color_print('Min mag difference in TPF: ', 'cyan', '%f mag (%f arsec)' % (minalldif, minallsep), 'default')

    sizes = 15/1.5**(grpmag-10)

else:
    sizes = 10*np.ones(1)

if not args.noplots:
    if args.pixlcs:
        pfig, pax = plt.subplots(figsize=[8,8])
        tmin = np.nanmin(time)
        tlen = np.nanmax(time) - tmin
        tnor = (time - tmin) / tlen

        pax.imshow(np.log10(np.nanmedian(flux[::10], axis=0)), cmap=args.cmap, extent=[0,args.size,0,args.size])

        theflux = flux - bkgs[:,None,None]
        for i in range(args.size):
            for j in range(args.size):
                fmin = np.nanmin(theflux[:,i,j])
                flen = np.nanmax(theflux[:,i,j]) - fmin
                fnor = (theflux[:,i,j] - fmin) / flen

                pax.plot(j+tnor, i+fnor, '-', color='lime' if dap[bidx][i,j] else 'tomato', lw=.1)

        pax.set_xticks(np.arange(-.5, args.size, 1), minor=True)
        pax.set_yticks(np.arange(-.5, args.size, 1), minor=True)
        pax.grid(which='minor', zorder=99)
        pfig.tight_layout()

    fig1 = plt.figure(figsize=[12,3], dpi=144)
    gs   = gridspec.GridSpec(2, 2, width_ratios=[1,5])#, height_ratios=[1,1])

    ax0 = plt.subplot(gs[1,1])
    ax0.errorbar(time, bkgs, yerr=berr, fmt='ok', ms=2)
    ax0.set_title('Background')
    ax0.set_ylabel(r'Flux  (e-/s)', fontweight='bold')
    ax0.set_xlabel(r'BJD', fontweight='bold')

    ax1 = plt.subplot(gs[:,0])
    ax1.imshow(np.log10(np.nanmedian(flux[::10], axis=0)), cmap=args.cmap, aspect='equal',
                            extent=[0,args.size,0,args.size], origin='lower')

    if not args.psf:
        xm, ym = pixel_border(dap[bidx])
        for xi,yi in zip(xm, ym):
            ax1.plot(xi, yi, color='lime', lw=1.25)

    ax1.scatter(x, y, color='#FF0043', s=sizes[0], ec=None)

    #Gaia sources
    if args.gaia:
        ax1.scatter(gx[1:], gy[1:], c='chocolate', s=sizes[1:], ec=None, zorder=9)

    ax = plt.subplot(gs[0,1], sharex=ax0)
    ax.errorbar(lkf.time.value, lkf.flux, yerr=lkf.flux_err, fmt='ok', ms=2, lw=1.5)
    ax.set_ylabel(r'Flux  (e-/s)', fontweight='bold')
    ax.ticklabel_format(useOffset=False)
    ax.set_title('Light curve')

    ax1.grid(which='minor', zorder=99, lw=3)
    ax1.set_xlim(0, args.size)
    ax1.set_ylim(0, args.size)

    fig1.tight_layout()
    plt.show()

if args.pngstamp is not None:
    from matplotlib.colors import ListedColormap
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    sfig, sax = plt.subplots(figsize=[4,3])
    stamp     = sax.imshow(np.log10(np.nanmedian(flux[::10], axis=0)),
                           cmap=args.cmap, origin='lower', aspect='equal',
                           extent=[column, column+args.size, row, row+args.size])

    xm, ym = pixel_border(dap[bidx])
    for xi,yi in zip(xm, ym):
        sax.plot(column+xi, row+yi, color='#FF0043', lw=1.5)

    sax.grid(which='minor', zorder=99)

    if args.gaia:
        sax.scatter(column+gx[1:],row+gy[1:], c='chocolate', s=sizes[1:], ec=None, zorder=9)

    if args.pngstamp == 'minimal':
        sax.text(0.95, 0.95, 'Sector %02d\nCCD: %d\nCam: %d' % (args.Sector, ccd, cam), ha='right', va='top', transform=sax.transAxes, color='#FF0043', size='large')

        plt.axis('off')

        sfig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        sfig.tight_layout()

    else:
        sax.scatter(column+x, row+y, color='#FF0043', s=sizes[0], ec=None)
        cbar = sfig.colorbar(stamp, pad=0.025)
        cbar.set_label('log(Flux)', fontsize=8)

        if args.gaia:
            sax.text(0.25, 1.15, '%s' % targettitle, transform=sax.transAxes, fontsize=8, ha='center', va='bottom')
            sax.text(0.25,1.025,'Sector: %02d\nCam:     %d\nCCD:     %d' % (args.Sector, cam, ccd), fontsize=6, transform=sax.transAxes, ha='center', va='bottom', ma='left')

            import matplotlib.patheffects as path_effects
            for i in range(1,len(gx)):
                txt = sax.text(column+gx[i]+.2, row+gy[i]+.2, i, alpha=.8, fontsize=4, ha='left', va='bottom', color='w', clip_on=True)
                txt.set_path_effects([path_effects.Stroke(linewidth=.5, foreground='gray', alpha=.8),
                           path_effects.Normal()])

            for mi in range(6,17,2):
                sax.scatter([], [], s=15/1.5**(mi-10), color='chocolate', label=mi, ec=None)
            leg = sax.legend(ncol=3, fontsize=6, loc='lower center', bbox_to_anchor=(0.85, 1), frameon=False)
            leg.set_title(r'$G_{RP}$ Magnitude', prop = {'size': 6})
        else:
            #sax.text(0.5, 0.9, 'TIC %s' % args.TIC, transform=sfig.transFigure, fontsize=8, ha='center', va='bottom')
            #sax.text(0.25,1.025,'Sector: %02d / Cam:     %d / CCD:     %d' % (args.Sector, cam, ccd), fontsize=6, transform=sax.transAxes, ha='center', va='bottom', ma='left')
            sax.set_title('%s\nSector: %02d / Cam: %d / CCD: %d' % (targettitle, args.Sector, cam, ccd), fontsize=8, pad=0)



        sax.set_xticks(column + np.arange(0, args.size, args.size//4))
        sax.set_yticks(row + np.arange(0, args.size, args.size//4))

        sax.set_xlabel('CCD Column', fontsize=8)
        sax.set_ylabel('CCD Row', fontsize=8)

        pngsize = args.size//(2*args.pngzoom)
        sax.set_xlim(column + int(x) - pngsize, column + int(x) + pngsize)
        sax.set_ylim(row + int(y) - pngsize, row + int(y) + pngsize)

    sfig.savefig('%s_%02d_%s.pdf' % (targettitle.replace(" ", ""), args.Sector, args.pngstamp), dpi=600, bbox_inches='tight')

if args.animation.lower() == 'outreach':
    from tsani import Outreach
    from matplotlib.animation import FuncAnimation

    fig     = plt.figure(figsize=[4,3])
    ax      = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor('black')
    adata   = flux - bkgs[:,None,None]
    ud      = Outreach(fig, ax, adata, lkf)
    anim    = FuncAnimation(fig, ud, frames=len(adata), init_func=ud.init, interval=60000/len(adata), save_count=len(adata))

    anim.save('movie.mp4', dpi=200)

elif args.animation.lower() == 'talk':
    from tsani import UpdateDist
    from matplotlib.animation import FuncAnimation


inst   = np.repeat('TESS', len(lkf.time))
output = np.transpose([lkf.time, lkf.flux, lkf.flux_err, inst])
np.savetxt('%s_%02d.dat' % (targettitle.replace(" ", ""), args.Sector), output, fmt='%s')

if args.folder is None:
    os.system('rm tesscut/*%.6f*' % ra)

color_print('\nDone!\n', 'lightgreen')
