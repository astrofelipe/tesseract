import __future__
import h5py
import argparse
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from astropy.utils.console import color_print
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from utils import mask_planet#, FFICut
from lightkurve.targetpixelfile import KeplerTargetPixelFileFactory
from tqdm import tqdm
from astropy.stats import SigmaClip
from photutils import MMMBackground
from autoap import generate_aperture, select_aperture
from lightkurve.lightcurve import TessLightCurve
from tess_stars2px import tess_stars2px_function_entry as ts2p
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Targets', type=str)
parser.add_argument('--ncpu', type=int, default=20)
parser.add_argument('--output', action='store_true')
parser.add_argument('--size', type=int, default=21)

args = parser.parse_args()

fs  = np.sort(glob.glob('/horus/TESS/FFI/s%04d/*.hdf5' % args.Sector))
h5s = [h5py.File(f, 'r') for f in fs]

if args.Targets[-3:] == 'pkl':
    import pickle
    f = open(args.Targets, 'rb')
    d = pickle.load(f)
    tics = np.array([int(item) for item in d.keys()])

    svals = np.array([list(item.values()) for item in d.values()]).astype(bool)
    smask = svals[:,args.Sector-1]

    if args.output:
        np.savetxt('targets_s%04d.txt' % args.Sector, tics[smask], fmt='%s')


else:
    #tics, ra, dec = np.genfromtxt(args.Targets, usecols=(0,1,2), delimiter=',', skip_header=1).astype(int)
    catalog = pd.read_csv(args.Targets)
    tics    = np.array(catalog['ID'])
    ra      = np.array(catalog['ra'])
    dec     = np.array(catalog['dec'])
    #print(ra, dec)
    #_, _, _, _, cam, ccd, _, _, _ = ts2p(tics, ra, dec)#, trySector=np.ones(len(ra))*args.Sector)
    #print(cam, ccd)

color_print('Trying %d targets for Sector %d' % (len(tics), args.Sector), 'lightcyan')

def FFICut(ffis, x, y, size):
    ncads  = len(ffis['FFIs'])
    x      = int(x)
    y      = int(y)

    aflux  = ffis['FFIs'][:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]
    aerrs  = ffis['errs'][:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]

    boxing = KeplerTargetPixelFileFactory(n_cadences=ncads, n_rows=size, n_cols=size)

    #for i,f in enumerate(tqdm(aflux)):
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

def make_lc(tic, ra, dec):
    #target = Catalogs.query_object('TIC %d' % tic, radius=0.05, catalog='TIC')
    #ra     = float(target[0]['ra'])
    #dec    = float(target[0]['dec'])

    _, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)
    #print('TIC: ', 'lightred', str(tic), 'default')
    #print('Camera: ', 'lightred', str(cam[0]), 'default', ' / CCD: ', 'lightred', str(ccd[0]))
    idx = (cam[0]-1)*4 + (ccd[0]-1)

    h5  = h5s[idx]
    q   = h5['data'][3] == 0
    ffi = np.array(glob.glob('/horus/TESS/FFI/s%04d/tess*-s%04d-%d-%d-*ffic.fits' % (args.Sector, args.Sector, cam, ccd)))[q][0]
    #print('\tSolving coordinates...')
    hdr = fits.getheader(ffi, 1)

    w   = WCS(hdr)
    x,y = w.all_world2pix(ra, dec, 0)

    allhdus = FFICut(h5, y, x, args.size)
    hdus    = allhdus.hdu

    qual = hdus[1].data['QUALITY'] == 0
    time = hdus[1].data['TIME'][q] + hdus[1].header['BJDREFI']
    flux = hdus[1].data['FLUX'][q]
    errs = hdus[1].data['FLUX_ERR'][q]
    bkgs = np.zeros(len(flux))

    for i,f in enumerate(flux):
        sigma_clip = SigmaClip(sigma=3)
        bkg        = MMMBackground(sigma_clip=sigma_clip)
        bkgs[i]    = bkg.calc_background(f)

    #DBSCAN Aperture
    x = x - int(x) + args.size//2
    y = y - int(y) + args.size//2

    daps = [generate_aperture(flux - bkgs[:,None,None], n=i) for i in [1,3,5,7,9,11,13,15]]
    dap  = np.array([select_aperture(d, x, y) for d in daps])

    #Aperture photometry
    lcfl = np.einsum('ijk,ljk->li', flux - bkgs[:,None,None], dap)
    lcer = np.sqrt(np.einsum('ijk,ljk->li', np.square(errs), dap))

    #Lightkurves
    lkf = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]
    #lkf = [lk.flatten(polyorder=2, window_length=85) for lk in lks] if args.norm else lks

    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    #print('\tCalculating best lightcurve')
    #print('\t\tBest lk:', bidx)
    #print('\t\t%d pixels in aperture' % dap[bidx].sum())
    lkf  = lkf[bidx]

    inst   = np.repeat('TESS', len(time))
    output = np.transpose([time, lkf.flux, lkf.flux_err, inst])
    #print('\tSaving TIC%s.dat...' % tic)
    np.savetxt('TIC%s.dat' % tic, output, fmt='%s')

#make_lc(tics[0], ra[0], dec[0])
Parallel(n_jobs=args.ncpu)(delayed(make_lc)(tics[i], ra[i], dec[i]) for i in tqdm(range(len(tics))))
