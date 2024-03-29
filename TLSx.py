import argparse
import numpy as np
from cleaner import cleaner
from astropy.table import Table
from lightkurve.lightcurve import TessLightCurve
from transitleastsquares import transitleastsquares as TLS


def run_TLS(fn, min_period=0.2, target=None):
    t,f,e = np.genfromtxt(fn, usecols=(0,1,2), unpack=True)
    mask  = cleaner(t, f)

    t = t[~mask]
    f = f[~mask]
    e = e[~mask]

    lc = TessLightCurve(time=t, flux=f, flux_err=e).flatten(window_length=51, polyorder=2, niters=5)

    return the_TLS(fn, lc.time.value, lc.flux, lc.flux_err, min_period, target)

def the_TLS(fn, t, f, e, min_period=0.2, target=None, use_threads=1, show_progress_bar=False):
    #try:
    model   = TLS(t, f, e) #Algunos targets pueden tener flujo negativo (borde FFI) y se cae
    results = model.power(n_transits_min=1, period_min=min_period, use_threads=use_threads, show_progress_bar=show_progress_bar)
    #except:
    #    return fn, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99


    if target is not None:
        return results

    else:
        try:
            depth_mean_even = np.nanmedian(results.depth_mean_even)
            depth_mean_odd  = np.nanmedian(results.depth_mean_odd)
            return fn, results.period, results.T0, results.duration, 1-results.depth, results.SDE, depth_mean_even, depth_mean_odd, results.odd_even_mismatch, results.transit_times[1], results.transit_count
        except:
            return fn, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLS for a folder with many LC files...')
    parser.add_argument('Folder', help='Folder containing .dat files')
    parser.add_argument('--target', type=int, default=None, help='Run on single target')
    parser.add_argument('--max-period', type=float, default=25)
    parser.add_argument('--min-period', type=float, default=0.2)
    parser.add_argument('--ncpu', type=int, default=10, help='Number of CPUs to use')
    parser.add_argument('--output', default='TLS_result.dat')

    args = parser.parse_args()

    if args.target is not None:
        fn      = args.Folder + 'TIC%d.dat' % args.target
        results = run_TLS(fn)

        import matplotlib.pyplot as plt
        #fig, ax = plt.subplots(figsize=[10,3])
        #ax.plot(results[1], results.power, '-k', lw=0.5)
        #print(results.keys())

        t,f,e = np.genfromtxt(fn, usecols=(0,1,2), unpack=True)

        fig, ax = plt.subplots()
        phase = (t - results[2] + 0.5*results[1]) % results[1] - 0.5*results[1]
        ax.plot(phase, f, '.k')

        #ax.set_xlim(-results[3]*1.5, results[3]*1.5)
        #ax.set_ylim(results[4]*0.995, 1.005)

        print(results)

        plt.show()

    else:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        import glob

        allfiles = glob.glob(args.Folder + 'TIC*.dat')

        results  = Table(rows=Parallel(n_jobs=args.ncpu, verbose=0)(delayed(run_TLS)(f, min_period=args.min_period) for f in tqdm(allfiles)))
        print(results)
        rmask    = results['col5'] > 0
        results  = results[rmask]
        #results  = np.array([run_TLS(f) for f in tqdm(allfiles)])
        order    = np.argsort(results['col5'])[::-1]
        results  = results[order]
        print(results)

        np.savetxt(args.output, results, fmt='%s')
