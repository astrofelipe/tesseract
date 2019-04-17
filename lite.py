import __future__
import h5py
import argparse
import glob
import numpy as np
from astroquery.mast import Catalogs
from utils import mask_planet, FFICut
from tess_stars2px import tess_stars2px_function_entry as ts2p
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Targets', type=str)
parser.add_argument('--output', action='store_true')
parser.add_argument('--size', type=int, default=21)

args = parser.parse_args()

fs  = np.sort(glob.glob('/horus/TESS/FFI/s%04d/*.hdf5' % args.Sector))
print(fs)
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
    tics = np.genfromtxt(args.Target, usecols=(0,), delim=',')

def make_lc(tic):
    target = Catalogs.query_object('TIC %d' % tic, radius=0.05, catalog='TIC')
    ra     = float(target[0]['ra'])
    dec    = float(target[0]['dec'])

    _, _, _, _, cam, ccd, _, _, _ = ts2p(0, ra, dec, trySector=args.Sector)
    cam = cam[0]-1
    ccd = ccd[0]-1
    idx = cam*4 + ccd

    h5  = h5s[idx]
    q   = h5['data'][3] == 0
    ffi = glob.glob('/horus/TESS/FFI/TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, cam, ccd))[q][0]
    hdr = fits.getheader(ffi, 1)

    w   = WCS(hdr)
    x,y = w.all_world2pix(ra, dec, 0)

    allhdus = FFICut(ffis, y, x, args.size)
    hdus    = allhdus.hdu

    qual = hdus[1].data['QUALITY'] == 0
    time = hdus[1].data['TIME'][ma] + hdus[1].header['BJDREFI']
    flux = hdus[1].data['FLUX'][ma]
    errs = hdus[1].data['FLUX_ERR'][ma]
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
    lks = [TessLightCurve(time=time, flux=lcfl[i], flux_err=lcer[i]) for i in range(len(lcfl))]
    lkf = [lk.flatten(polyorder=2, window_length=85) for lk in lks] if args.norm else lks

    #Select best
    cdpp = [lk.estimate_cdpp() for lk in lkf]
    bidx = np.argmin(cdpp)
    print('Best lk:', bidx)
    print(dap[bidx].sum(),'pixels in aperture')
    lkf  = lkf[bidx]

    inst   = np.repeat('TESS', len(time))
    output = np.transpose([time, lkf.flux, lkf.flux_err, inst])
    np.savetxt('TIC%s.dat' % (args.TIC), output, fmt='%s')
