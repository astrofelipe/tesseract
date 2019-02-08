import glob
import argparse
import numpy as np
from utils import BLSer
from astropy.io import fits
from astropy.table import Table
from joblib import Parallel, delayed
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description='Concatenate lcs from targets in more than 1 sector')
parser.add_argument('Folder', type=str, help='Folder containing subfolders with PDCSAP lightcurves')
parser.add_argument('--ncpu', type=int, default=8, help='Number of CPUs to use in multi mode')
parser.add_argument('--target', type=int, default=None, help='Runs only on single target')
parser.add_argument('--overlaps', type=int, default=None, help='Minimum number of overlaps to consider (this will overwrite everything in that range)')
parser.add_argument('--output', type=str, default='merged_BLS.dat', help='Output filename')

args = parser.parse_args()

def mergeandBLS(tic):
    fs = glob.glob('%s*/*%016d*lc.fits' % (args.Folder, tic))
    
    t = []
    y = []
    e = []
    
    for f in fs:
        data = fits.getdata(f)
        m    = np.isfinite(data['TIME']) & np.isfinite(data['PDCSAP_FLUX']) & (data['QUALITY'] == 0)
        time = data['TIME'][m]
        flux = data['PDCSAP_FLUX'][m]
        ferr = data['PDCSAP_FLUX_ERR'][m]

        t.append(time)
        y.append(flux)
        e.append(ferr)

    t = np.concatenate(t)
    y = np.concatenate(y)
    e = np.concatenate(e)

    opt = np.transpose([t,y,e])
    np.savetxt('%smerged/%016d.dat' % (args.Folder, tic), opt, fmt='%f')
    period, t0, duration, depth, SNR = BLSer(t, y, e, maximum_period=len(fs)*40.)
    del opt,t,y,e

    return tic, period, t0, duration, depth, SNR
  

if args.target is None:
    all_lcs = glob.glob('%s*/*lc.fits' % args.Folder)
    only_id = [int(s.split('-')[-3]) for s in all_lcs]
    uni, cts = np.unique(only_id, return_counts=True)
    
    nover = args.overlaps if args.overlaps is not None else np.max(cts)
    mask  = cts == nover
    runids = uni[mask]

    results = np.array(Parallel(n_jobs=args.ncpu, verbose=10)(delayed(mergeandBLS)(tic) for tic in runids))
    order  = np.argsort(results[:,5])[::-1]

    names = ['#TIC', 'Period', 't0', 'Duration', 'Depth', 'SNR']
    opt   = Table(results[order], names=names)
    opt['#TIC'] = opt['#TIC'].astype(int)

    opt.write(args.output, format='ascii.basic', overwrite=True)
        
else:
    tic, period, t0, duration, depth, SNR = mergeandBLS(args.target)
    t, y, e = np.genfromtxt('%smerged/%016d.dat' % (args.Folder, tic), unpack=True)
    print 'TIC ', tic
    print 'Period: ', period
    print 't0    : ', t0
    print 'Dur   : ', duration
    print 'Depth : ', depth
    print 'SNR   : ', SNR

    y  = (y - np.nanmedian(y))*1e6
    yf = medfilt(y, 351)
    y  = y - yf 
    ph = (t - t0 + 0.5*period) % period - 0.5*period

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t, y, '.', ms=1, alpha=.66)
    ax[1].plot(ph, y, '.', ms=1, alpha=.66)
    plt.show()
