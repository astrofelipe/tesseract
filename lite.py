import __future__
import h5py
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import pixel_border
from mpi4py import MPI
from astropy.utils.console import color_print
from astropy.io import fits
from astropy.wcs import WCS
from lightkurve.targetpixelfile import KeplerTargetPixelFileFactory
from astropy.stats import SigmaClip
from photutils import MMMBackground
from autoap import generate_aperture, select_aperture
from lightkurve.lightcurve import TessLightCurve
from tess_stars2px import tess_stars2px_function_entry as ts2p

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Targets', type=str)
parser.add_argument('--output', action='store_true')
parser.add_argument('--size', type=int, default=21)
parser.add_argument('--circ', action='store_true')
parser.add_argument('--sixteen', action='store_true', help='Uses 1 core per ccd, this needs to be called with mpi using exactly 16 cores')
parser.add_argument('--onlyjpg', action='store_true')

args = parser.parse_args()

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

catalog = pd.read_csv(args.Targets, names=['ID', 'ra', 'dec', 'Tmag'], skiprows=1)
tics    = np.array(catalog['ID'])
ra      = np.array(catalog['ra'])
dec     = np.array(catalog['dec'])

color_print('Trying %d targets for Sector %d' % (len(tics), args.Sector), 'lightcyan')

def FFICut(ffis, x, y, size):
    ncads  = len(ffis['FFIs'])

    x      = int(x)
    y      = int(y)

    xshape = ffis['FFIs'].shape[1]
    yshape = ffis['FFIs'].shape[2]

    x1 = np.max([0, x-size//2])
    x2 = np.min([xshape, x+size//2+1])
    y1 = np.max([0, y-size//2])
    y2 = np.min([yshape, y+size//2+1])

    aflux  = ffis['FFIs'][0:ncads, x1:x2, y1:y2]
    aerrs  = ffis['errs'][0:ncads, x1:x2, y1:y2]

    boxing = KeplerTargetPixelFileFactory(n_cadences=ncads, n_rows=aflux.shape[1], n_cols=aflux.shape[2])

    for i,f in enumerate(aflux):
        ti = ffis['data'][0,i]
        tf = ffis['data'][1,i]
        b  = ffis['data'][2,i]
        q  = ffis['data'][3,i]

        header = {'TSTART': ti, 'TSTOP': tf,
                  'QUALITY': q}

        boxing.add_cadence(frameno=i, flux=f, flux_err=aerrs[i], header=header)

    TPF = boxing.get_tpf()

    return TPF

def make_lc(tic, ra=None, dec=None, process=None):
    print('Calculando Camara y CCD')
    _, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)

    idx = (cam[0]-1)*4 + (ccd[0]-1)

    print('Leyendo Header')
    h5  = h5s[idx]
    q   = h5['data'][3] == 0
    ffi = np.array(glob.glob('/horus/TESS/FFI/s%04d/tess*-s%04d-%d-%d-*ffic.fits' % (args.Sector, args.Sector, cam, ccd)))[q][0]
    hdr = fits.getheader(ffi, 1)

    w   = WCS(hdr)
    x,y = w.all_world2pix(ra, dec, 0)

    print('Leyendo hdf5')
    allhdus = FFICut(h5, y, x, args.size)
    hdus    = allhdus.hdu

    qual = hdus[1].data['QUALITY'] == 0
    time = hdus[1].data['TIME'][q]
    flux = hdus[1].data['FLUX'][q]
    errs = hdus[1].data['FLUX_ERR'][q]
    bkgs = np.zeros(len(flux))

    print('Calculando Background')
    for i,f in enumerate(flux):
        sigma_clip = SigmaClip(sigma=3)
        bkg        = MMMBackground(sigma_clip=sigma_clip)
        bkgs[i]    = bkg.calc_background(f)

    #DBSCAN Aperture
    x = x - int(x) + args.size//2
    y = y - int(y) + args.size//2

    if not args.circ:
        daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,2,3,4,5]]
        dap  = np.array([select_aperture(d, x, y) for d in daps])
    else:
        XX, YY = np.ogrid[:args.size, :args.size]
        dap    = [np.sqrt((XX-y)**2 + (YY-x)**2) < i for i in np.arange(1,3.1,0.5)]

    #Aperture photometry
    lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
    lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

    #Lightkurves
    lkf = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]

    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    lkf  = lkf[bidx]

    #Save light curve
    inst   = np.repeat('TESS', len(time))
    output = np.transpose([time, lkf.flux, lkf.flux_err, inst])

    np.savetxt('TIC%s.dat' % tic, output, fmt='%s')
    print('LC READY!')

    #Save JPG preview
    stamp = flux - bkgs[:,None,None]
    fig, ax = plt.subplots(figsize=[4,4])
    fig.patch.set_visible(False)
    stamp = np.log10(np.nanmedian(stamp[::10], axis=0))
    stamp[np.isnan(stamp)] = np.nanmedian(stamp)

    ax.matshow(stamp, cmap='gist_gray', aspect='equal')

    xm, ym = pixel_border(dap[bidx])
    for xi,yi in zip(xm, ym):
        ax.plot(xi, yi, color='lime', lw=1.25)

    ax.plot(x, y, '.r')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig('img/TIC%s.png' % tic, dpi=72)

    plt.close(fig)
    del(fig,ax)

    return 1



if not args.sixteen:
    fs  = np.sort(glob.glob('/horus/TESS/FFI/s%04d/*.hdf5' % args.Sector))
    h5s = [h5py.File(f, 'r', libver='latest') for f in fs]

    for i in range(len(tics)):
        if os.path.isfile('TIC%d.dat' % tics[i]):
            continue

        if i%size!=rank:
            continue


        make_lc(tics[i], ra[i], dec[i])
        print(tics[i])

    for h in h5s:
        h.close()

else:
    cam = 1 + (rank // 4)
    ccd = 1 + (rank % 4)
    print('Iniciando camara %d y CCD %d' % (cam,ccd))
    h5f = '/horus/TESS/FFI/s%04d/TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Sector, cam, ccd)
    h5  = h5py.File(h5f, 'r', libver='latest')

    #for i,tic in enumerate(tqdm(tics)):
    for i,tic in enumerate(tics):
        if i%20==0:
            print('%d curvas procesadas en Camara %d / CCD %d' % (i, cam, ccd))
            sys.stdout.flush()

        if os.path.isfile('TIC%d.dat' % tic):
            continue

        _, _, _, _, ccam, cccd, _, _, _ = ts2p(tic, ra[i], dec[i], trySector=args.Sector)
        ccam = ccam[0]
        cccd = cccd[0]

        if (ccam!=cam) or (cccd!=ccd):
            continue

        q   = h5['data'][3] == 0
        ffi = np.array(glob.glob('/horus/TESS/FFI/s%04d/tess*-s%04d-%d-%d-*ffic.fits' % (args.Sector, args.Sector, cam, ccd)))[q][0]
        hdr = fits.getheader(ffi, 1)

        w   = WCS(hdr)
        x,y = w.all_world2pix(ra[i], dec[i], 0)

        allhdus = FFICut(h5, y, x, args.size)
        hdus    = allhdus.hdu

        qual = hdus[1].data['QUALITY'] == 0
        time = hdus[1].data['TIME'][q]
        flux = hdus[1].data['FLUX'][q]
        errs = hdus[1].data['FLUX_ERR'][q]
        bkgs = np.zeros(len(flux))

        print('Calculando Background')
        for i,f in enumerate(flux):
            sigma_clip = SigmaClip(sigma=3)
            bkg        = MMMBackground(sigma_clip=sigma_clip)
            bkgs[i]    = bkg.calc_background(f)

        #DBSCAN Aperture
        x = x - int(x) + flux.shape[1]//2
        y = y - int(y) + flux.shape[2]//2

        if not args.circ:
            daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,2,3,4,5]]
            dap  = np.array([select_aperture(d, x, y) for d in daps])
        else:
            XX, YY = np.ogrid[:flux.shape[1], :flux.shape[2]]
            dap    = [np.sqrt((XX-y)**2 + (YY-x)**2) < i for i in np.arange(1,3.1,0.5)]

        #Aperture photometry
        lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
        lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

        #Lightkurves
        lkf = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]

        #Select best
        cdpp = [lk.estimate_cdpp() for lk in lkf]
        bidx = np.argmin(cdpp)
        lkf  = lkf[bidx]

        #Save light curve
        inst   = np.repeat('TESS', len(time))
        output = np.transpose([time, lkf.flux, lkf.flux_err, inst])

        np.savetxt('TIC%s.dat' % tic, output, fmt='%s')

        #Save JPG preview
        stamp = flux - bkgs[:,None,None]
        fig, ax = plt.subplots(figsize=[4,4])
        fig.patch.set_visible(False)
        stamp = np.log10(np.nanmean(stamp[::len(stamp)//100], axis=0))
        stamp[np.isnan(stamp)] = np.nanmedian(stamp)

        ax.matshow(stamp, cmap='gist_gray', aspect='equal')

        xm, ym = pixel_border(dap[bidx])
        for xi,yi in zip(xm, ym):
            ax.plot(xi, yi, color='lime', lw=1.25)

        ax.plot(x, y, '.r')
        plt.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.savefig('img/TIC%s.png' % tic, dpi=72)

        plt.close(fig)
        del(fig,ax)

    h5.close()
