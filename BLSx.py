import glob
import argparse
import numpy as np
from astropy.stats import BoxLeastSquares as BLS

parser = argparse.ArgumentParser(description='BLS for a folder with many LC files...')
parser.add_argument('Folder', help='Folder containing FITS files')
parser.add_argument('--target', type=int, default=None, help='Run on single target')
parser.add_argument('--mags', type=float, nargs=2, help='Magnitude limits')
parser.add_argument('--output', default=None)

args = parser.parse_args()

folder = args.Folder
if folder[-1] != '/':
    folder += '/'

def run_BLS(fl):
    t, f = np.genfromtxt(fl, usecols=(0,1), unpack=True)

    durations = np.linspace(0.05, 0.2, 50)# * u.day
    model     = BLS(t,f)
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

    return fl, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half

if args.target is not None:
    targetfile = folder + 'TIC%d.dat' % args.target
    t,f        = np.genfromtxt(targetfile, usecols=(0,1), unpack=True)

    result = run_BLS(targetfile)
    period = result[1]
    t0     = result[2]
    dur    = result[3]
    depth  = result[4]
    snr    = result[5]

    ph = (t-t0 + 0.5*period) % period - 0.5*period

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=[10,4])

    ax.plot(ph*24, f, '.k', ms=5)
    print(ph.max(), ph.min())

    ax.set_xlim(-48*dur, 48*dur)
    ax.set_ylim(1-1.5*depth, 1+5e-3)

    ax.set_title(r''+ targetfile + r'   $P=%.5f$' % period + r'   SNR=%f' % snr)
    ax.set_xlabel('Hours from mid-transit')
    ax.set_ylabel('Normalized flux')
    print(result)

    plt.show()


else:
    from joblib import Parallel, delayed

    allfiles = glob.glob(folder + 'TIC*.dat')
    results  = np.array(Parallel(n_jobs=12, verbose=10)(delayed(run_BLS)(f) for f in allfiles))
    order    = np.argsort(results[:,5])[::-1]
    results  = results[order]
    print(results)

    np.savetxt('BLS_result.dat', results, fmt='%s')
