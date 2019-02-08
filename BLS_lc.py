import argparse
import glob
import numpy as np
#import matplotlib.pyplot as plt
from bls import BLS
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from scipy.signal import medfilt
from joblib import Parallel, delayed

''' Exploto en emu, go with joblib
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
'''

#testfile = '/data/TESS/LC/official/tess2018234235059-s0002-0000000281459670-0121-s_lc.fits'

parser = argparse.ArgumentParser(description='BLS for a folder with many FITS...')
parser.add_argument('Folder', help='Folder containing FITS files')
parser.add_argument('--output', default='BLS.dat')

args = parser.parse_args()
folder = args.Folder

allfiles = glob.glob(folder + '*.fits')

def run_BLS(testfile):
    with fits.open(testfile) as hdus:
        tic  = hdus[1].header['TICID']
        data = hdus[1].data
        t    = data['TIME']
        y    = data['PDCSAP_FLUX']
        q    = (data['QUALITY'] == 0) & np.isfinite(t) & np.isfinite(y)

    t = np.ascontiguousarray(t[q], dtype=np.float64)# * u.day
    y = np.ascontiguousarray(y[q], dtype=np.float64)
    y = (y / np.median(y) - 1)*1e6

    trend = medfilt(y, 751)
    yflat = y - trend
    
    durations = np.linspace(0.05, 0.2, 50)# * u.day
    model     = BLS(t, yflat)
    result    = model.autopower(durations, frequency_factor=5.0, maximum_period=30.0)
    idx       = np.argmax(result.power)
    
    period = result.period[idx]
    t0     = result.transit_time[idx]
    dur    = result.duration[idx]
    depth  = result.depth[idx]
    snr    = result.depth_snr[idx]

    try:    
        stats  = model.compute_stats(period, dur, t0)
        depth_even = stats['depth_even'][0]
        depth_odd  = stats['depth_odd'][0]
        depth_half = stats['depth_half'][0]
    except:
        depth_even = 0
        depth_odd  = 0
        depth_half = 0

    print tic, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half
    return tic, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half
    
    '''
    x = (t-t0 + 0.5*period) % period - 0.5*period

    fig, ax = plt.subplots(nrows=2)
    ax[0].axvline(period, alpha=0.4, lw=3)
    ax[0].plot(result.period, result.power, 'k', lw=0.5)
    ax[1].plot(x, yflat, '.')
    plt.show()
    '''

output = np.array(Parallel(n_jobs=12, verbose=0)(delayed(run_BLS)(f) for f in allfiles))
order  = np.argsort(output[:,5])[::-1]

names = ['#TIC', 'Period', 't0', 'Duration', 'Depth', 'SNR', 'Depth_even', 'Depth_odd', 'Depth_half']
opt   = Table(output[order], names=names)
opt['#TIC'] = opt['#TIC'].astype(int)

opt.write(args.output, format='ascii.basic', overwrite=True)
