import glob
import argparse
import numpy as np
from tqdm import tqdm
from lightkurve.lightcurve import TessLightCurve
from astropy.stats import BoxLeastSquares as BLS

parser = argparse.ArgumentParser(description='BLS for a folder with many LC files...')
parser.add_argument('Folder', help='Folder containing FITS files')
parser.add_argument('--target', type=int, default=None, help='Run on single target')
#parser.add_argument('--mags', type=float, nargs=2, help='Magnitude limits')
parser.add_argument('--max-period', type=float, default=30.)
parser.add_argument('--ncpu', type=int, default=10, help='Number of CPUs to use')
parser.add_argument('--output', default='BLS_result.dat')

args = parser.parse_args()

folder = args.Folder
if folder[-1] != '/':
    folder += '/'

def run_BLS(fl):
    t, f = np.genfromtxt(fl, usecols=(0,1), unpack=True)
    mask = (t > 2458492.3) + ((t>4913338.3)*(t<4913337))
    t = t[mask]
    f = f[mask]
    #mask = (t > 2458492.) & ((t < 2458504.5) | (t > 2458505.))
    lc   = TessLightCurve(time=t, flux=f).flatten()

    durations = np.linspace(0.05, 0.2, 50)# * u.day
    model     = BLS(lc.time,lc.flux)
    try:
        result    = model.autopower(durations, frequency_factor=5.0, maximum_period=args.max_period)
    except:
        print(fl)
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

    return fl, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half

if args.target is not None:
    targetfile = folder + 'TIC%d.dat' % args.target
    t,f        = np.genfromtxt(targetfile, usecols=(0,1), unpack=True)
    lc         = TessLightCurve(time=t, flux=f).flatten()


    result = run_BLS(targetfile)
    period = result[1]
    t0     = result[2]
    dur    = result[3]
    depth  = result[4]
    snr    = result[5]

    ph = (t-t0 + 0.5*period) % period - 0.5*period

    import matplotlib.pyplot as plt
    fig2, ax2 = plt.subplots(figsize=[20,3])
    ax2.plot(lc.time, lc.flux, 'k', lw=.8, zorder=-5)
    ax2.scatter(lc.time, lc.flux, s=10, color='tomato', edgecolor='black', lw=.5, zorder=-4)

    fig, ax = plt.subplots(figsize=[10,4])

    ax.plot(ph*24, lc.flux, '.k', ms=5)
    print(ph.max(), ph.min())

    ax.set_xlim(-48*dur, 48*dur)
    ax.set_ylim(1-1.5*depth, 1+5e-3)

    ax.set_title(r''+ targetfile + r'   $P=%.5f$' % period + r'   SNR=%f' % snr)
    ax.set_xlabel('Hours from mid-transit')
    ax.set_ylabel('Normalized flux')
    print(result)

    plt.show()


else:
    from joblib import Parallel, delayed, Memory

    memory  = Memory('./cachedir', verbose=0)
    costoso = memory.cache(run_BLS)

    allfiles = glob.glob(folder + 'TIC*.dat')
    results  = np.memmap('temp.npz', dtype='float32', mode='w+', shape=(len(allfiles),9))

    results  = np.array(Parallel(n_jobs=args.ncpu, verbose=0)(delayed(costoso)(f) for f in tqdm(allfiles)))
    order    = np.argsort(results[:,5])[::-1]
    results  = results[order]
    print(results)

    np.savetxt(args.output, results, fmt='%s')
