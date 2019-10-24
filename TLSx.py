import argparse
import numpy as np
from lightkurve.lightcurve import TessLightCurve
from transitleastsquares import transitleastsquares as TLS

parser = argparse.ArgumentParser(description='TLS for a folder with many LC files...')
parser.add_argument('Folder', help='Folder containing .dat files')
parser.add_argument('--target', type=int, default=None, help='Run on single target')
parser.add_argument('--max-period', type=float, default=25)
parser.add_argument('--min-period', type=float, default=0.2)
parser.add_argument('--ncpu', type=int, default=10, help='Number of CPUs to use')
parser.add_argument('--output', default='TLS_result.dat')

args = parser.parse_args()

def run_TLS(fn):
    #fn    = '%sTIC%d.dat' % (args.Folder, TIC)
    t,f,e = np.genfromtxt(fn, usecols=(0,1,2), unpack=True)
    lc    = TessLightCurve(time=t, flux=f, flux_err=e).flatten(window_length=51, polyorder=2, niters=5)

    fmed = np.nanmedian(lc.flux)
    fstd = np.nanstd(lc.flux)
    stdm = lc.flux < 0.97#np.abs(lc.flux-fmed) > 3*fstd

    mask1  = ((lc.time > 2458325) & (lc.time < 2458326)) + ((lc.time > 2458347) & (lc.time < 2458350)) + ((lc.time > 2458352.5) & (lc.time < 2458353.2))
    mask3  = ((lc.time > 2458382) & (lc.time < 2458384)) + ((lc.time > 2458407) & (lc.time < 2458410)) + ((lc.time > 2458393.5) & (lc.time < 2458397))
    mask4  = ((lc.time > 2458419) & (lc.time < 2458422)) + ((lc.time > 2458422) & (lc.time < 2458424)) + ((lc.time > 2458436) & (lc.time < 2458437))
    mask5  = ((lc.time > 2458437.8) & (lc.time < 2458438.7)) + ((lc.time > 2458450) & (lc.time < 2458452)) + ((lc.time > 2458463.4) & (lc.time < 2458464.2))
    mask6  = ((lc.time > 2458476.7) & (lc.time < 2458478.7))
    mask7  = ((lc.time > 2458491.6) & (lc.time < 2458492)) + ((lc.time > 2458504.6) & (lc.time < 2458505.2))
    mask8  = ((lc.time > 2458517.4) & (lc.time < 2458518)) + ((lc.time > 2458530) & (lc.time < 2458532))
    mask10 = ((lc.time > 4913400) & (lc.time < 4913403.5)) + ((lc.time > 4913414.2) & (lc.time < 4913417)) #s10
    mask11 = ((lc.time > 2458610.6) & (lc.time < 2458611.6)) + ((lc.time > 2458610.6) & (lc.time < 2458611.6))
    mask12 = ((lc.time > 2458624.5) & (lc.time < 2458626))
    mask13 = ((lc.time > 2458653.5) & (lc.time < 2458655.75)) + ((lc.time > 2458668.5) & (lc.time < 2458670))

    mask   = mask1 + mask3 + mask4 + mask5 + mask6 + mask7 + mask8 + mask10 + mask11 + mask12 + mask13 + stdm

    lc.time = lc.time[~mask]
    lc.flux = lc.flux[~mask]

    try:
        model   = TLS(lc.time, lc.flux, lc.flux_err)
        results = model.power(n_transits_min=1, period_min=args.min_period, use_threads=1, show_progress_bar=False)
    except:
        return

    if args.target is not None:
        return results

    else:
        try:
            return fn, results.period, results.T0, results.duration, 1-results.depth, results.SDE, np.nanmedian(results.depth_mean_even), np.nanmedian(results.depth_mean_odd), results.odd_even_mismatch, results.transit_times[1], results.transit_count
        except:
            return

if args.target is not None:
    fn      = args.Folder + 'TIC%d.dat' % args.target
    results = run_TLS(fn)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=[10,3])
    ax.plot(results.periods, results.power, '-k', lw=0.5)
    print(results.keys())
    fig, ax = plt.subplots()
    phase = (results.model_lightcurve_time - results.T0 + 0.5*results.period) % results.period - 0.5*results.period
    ax.plot((results.folded_phase-0.5)*results.period, results.folded_y, '.k')

    ax.set_xlim(-results.duration*1.5, results.duration*1.5)
    ax.set_ylim(results.depth*0.995, 1.005)

    print(1-results.depth)

    plt.show()

else:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import glob

    allfiles = glob.glob(args.Folder + 'TIC*.dat')

    results  = np.array(Parallel(n_jobs=args.ncpu, verbose=0)(delayed(run_TLS)(f) for f in tqdm(allfiles[:1000])))
    print(results)
    #results  = np.array([run_TLS(f) for f in tqdm(allfiles)])
    order    = np.argsort(results[:,5])[::-1]
    results  = results[order]
    print(results)

    np.savetxt(args.output, results, fmt='%s')
