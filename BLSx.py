import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cleaner import cleaner
from scipy.ndimage import median_filter
from tqdm import tqdm
from lightkurve.lightcurve import TessLightCurve
from astropy.stats import mad_std
from astropy.timeseries import BoxLeastSquares as BLS
from scipy.ndimage import median_filter
from transitleastsquares import transitleastsquares, period_grid

parser = argparse.ArgumentParser(description='BLS for a folder with many LC files...')
parser.add_argument('Folder', help='Folder containing FITS files')
parser.add_argument('--target', type=int, default=None, help='Run on single target')
#parser.add_argument('--mags', type=float, nargs=2, help='Magnitude limits')
parser.add_argument('--max-period', type=float, default=25)
parser.add_argument('--min-period', type=float, default=0.2)
parser.add_argument('--ncpu', type=int, default=10, help='Number of CPUs to use')
parser.add_argument('--TLS', action='store_true')
parser.add_argument('--output', default='BLS_result.dat')

args = parser.parse_args()

folder = args.Folder
if folder[-1] != '/':
    folder += '/'

def run_BLS(fl):
    t, f, e = np.genfromtxt(fl, usecols=(0,1,2), unpack=True)
    mask    = cleaner(t,f)
    
    t = t[~mask]
    f = f[~mask]
    e = e[~mask]

    lc   = TessLightCurve(time=t, flux=f, flux_err=e).flatten(window_length=51, polyorder=2, niters=5)

    #Test Fill
    '''
    diffs = np.diff(lc.time)
    stdd  = np.nanstd(diffs)
    medd  = np.nanmedian(diffs)

    maskgaps = diffs > 0.2#np.abs(diffs-medd) > stdd
    maskgaps = np.concatenate((maskgaps,[False]))
    '''

    '''
    for mg in np.where(maskgaps)[0]:
        addtime = np.arange(lc.time[mg]+0.05, lc.time[mg+1], 0.05)
        addflux = np.random.normal(1, 8e-4, len(addtime))

        lc.time = np.concatenate((lc.time, addtime))
        lc.flux = np.concatenate((lc.flux, addflux))

    addorder = np.argsort(lc.time)
    lc.time = lc.time[addorder]
    lc.flux = lc.flux[addorder]
    '''

    #fmed = np.nanmedian(lc.flux)
    #fstd = np.nanstd(lc.flux)
    #stdm = lc.flux < 0.97#np.abs(lc.flux-fmed) > 3*fstd

    periods   = np.exp(np.linspace(np.log(args.min_period), np.log(args.max_period), 5000))
    durations = np.linspace(0.05, 0.15, 20)# * u.day
    model     = BLS(lc.time,lc.flux) if not args.TLS else transitleastsquares(lc.time.value, lc.flux, lc.flux_err)

    #result    = model.power(periods, durations, oversample=20)#, objective='snr')
    result    = model.power(period_min=args.min_period, oversampling_factor=2, n_transits_min=1, use_threads=1, show_progress_bar=False)
    #try:
    #result    = model.autopower(durations, frequency_factor=2.0, maximum_period=args.max_period)
    #except:
    #    print(fl)
    idx       = np.argmax(result.power)


    period = result.period[idx]
    t0     = result.transit_time[idx]
    dur    = result.duration[idx]
    depth  = result.depth[idx]
    snr    = result.depth_snr[idx]
    '''
    period = result.period
    t0     = result.T0
    dur    = result.duration
    depth  = 1 - result.depth
    snr    = result.snr
    '''


    try:
        stats  = model.compute_stats(period, dur, t0)
        depth_even = stats['depth_even'][0]
        depth_odd  = stats['depth_odd'][0]
        depth_half = stats['depth_half'][0]
        t0, t1     = stats['transit_times'][:2]
        ntra       = len(stats['transit_times'])
    except:
        depth_even = 0
        depth_odd  = 0
        depth_half = 0
        t1         = 0
        ntra       = 0

    if args.target is not None:
        return fl, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half, t1, ntra, result.period, result.power, lc.time, lc.flux, diffs
    else:
        return fl, period, t0, dur, depth, snr, depth_even, depth_odd, depth_half, t1, ntra

if args.target is not None:
    from matplotlib.gridspec import GridSpec
    targetfile = folder + 'TIC%d.dat' % args.target
    #t,f        = np.genfromtxt(targetfile, usecols=(0,1), unpack=True)
    #mask       = ((t > 4913400) & (t < 4913403.5)) + ((t > 4913414.2) & (t < 4913417)) #s10
    #lc         = TessLightCurve(time=t, flux=f).flatten(window_length=31, polyorder=2, niters=3)

    result = run_BLS(targetfile)
    period = result[1]
    t0     = result[2]
    dur    = result[3]
    depth  = result[4]
    snr    = result[5]
    t      = result[13]
    f      = result[14]
    diffs  = result[15]
    pers   = result[11]
    powers = result[12]

    print(period,t0,dur)

    maskgaps = diffs > 0.2#np.abs(diffs-medd) > stdd
    maskgaps = np.concatenate((maskgaps,[False]))

    fig = plt.figure(figsize=[16,6], constrained_layout=True)
    gs  = GridSpec(2,6, figure=fig)

    pax = fig.add_subplot(gs[1,:2])
    pax.plot(pers, powers, '-k')
    pax.set_xlabel(r'Period  (days)', fontweight='bold')
    pax.set_ylabel(r'Power', fontweight='bold')


    ph = (t-t0 + 0.5*period) % period - 0.5*period

    ax2 = fig.add_subplot(gs[0,:])
    ax2.plot(t, f, 'k', lw=.8, zorder=-5)
    ax2.scatter(t, f, s=10, color='tomato', edgecolor='black', lw=.5, zorder=-4)
    if t0 > 1e6:
        t0s  = np.arange(t0, t[-1], period)
        for ti0 in t0s:
            ax2.axvline(ti0, alpha=.5)

    ax = fig.add_subplot(gs[1,2:4])
    ax.plot(ph*24, f, '.k', ms=5)
    print(ph.max(), ph.min())

    ax.set_xlim(-48*dur, 48*dur)
    ax.set_ylim(1-1.5*depth, 1+5e-3)

    ax.set_title(r''+ targetfile + r'   $P=%.5f$' % period + r'   SNR=%f' % snr)
    ax.set_xlabel('Hours from mid-transit')
    ax.set_ylabel('Normalized flux')

    plt.show()


else:
    from joblib import Parallel, delayed, Memory

    #memory  = Memory('./cachedir', verbose=0)
    #costoso = memory.cache(run_BLS)

    allfiles = glob.glob(folder + 'TIC*.dat')
    #results  = np.memmap('temp.npz', dtype='float32', mode='w+', shape=(len(allfiles),9))

    #results  = np.array(Parallel(n_jobs=args.ncpu, verbose=0)(delayed(costoso)(f) for f in tqdm(allfiles)))
    results  = np.array(Parallel(n_jobs=args.ncpu, verbose=0)(delayed(run_BLS)(f) for f in tqdm(allfiles)))
    order    = np.argsort(results[:,5])[::-1]
    results  = results[order]
    print(results)

    np.savetxt(args.output, results, fmt='%s')
