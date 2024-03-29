import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lightkurve.lightcurve import TessLightCurve
from cleaner import cleaner
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import transitleastsquares as TLS

parser = argparse.ArgumentParser(description='Searches for n planets in a light curve')
parser.add_argument('File', help='Light curve file')
parser.add_argument('--nplanets', type=int, default=5, help='Number of planets to search')
parser.add_argument('--ncpu', type=int, default=4, help='Number of CPUs to use')
parser.add_argument('--min-period', type=float, default=0.2)
parser.add_argument('--max-period', type=float, default=20)
parser.add_argument('--method', type=str, default='TLS')

args = parser.parse_args()


t,f,e = np.genfromtxt(args.File, usecols=(0,1,2), unpack=True)
ma    = cleaner(t, f)

t,f,e = t[~ma], f[~ma], e[~ma]
wl    = 2*(len(t)//42)+1
lc    = TessLightCurve(time=t, flux=f, flux_err=e).flatten(window_length=wl, polyorder=3, niters=5)

fig = plt.figure(constrained_layout=True, figsize=[15, 6])
gs  = GridSpec(ncols=5, nrows=3, figure=fig, height_ratios=[3,2,2])

#Light curve
axlc = fig.add_subplot(gs[0,:])
axlc.plot(lc.time.value, lc.flux, '.k', ms=1)

color = plt.cm.rainbow(np.linspace(0,1,args.nplanets))
#Iterate TLS
for i in range(args.nplanets):
    if args.method == 'TLS':
        model  = TLS(lc.time.value, lc.flux, lc.flux_err)
        result = model.power(n_transits_min=1, period_min=args.min_period, period_max=args.max_period, use_threads=args.ncpu, show_progress_bar=True)

        period = result.period
        t0     = result.T0
        dur    = result.duration
        depth  = result.depth

        periods, power = result.periods, result.power

    elif args.method == 'BLS':
        model  = BoxLeastSquares(lc.time.value, lc.flux, dy=lc.flux_err)
        result = model.autopower(0.15)

        periods, power = result.period, result.power

        idx    = np.argmax(power)
        period = periods[idx]
        t0     = result.transit_time[idx]
        dur    = result.duration[idx]
        depth  = result.depth[idx]

    #if i==1:
    #    period *= 2 2458339.018159

    phase = (lc.time.value - t0 + 0.5*period) % period - 0.5*period
    fph   = (lc.time.value - t0 + 0.5*period) % period - 0.5*period

    ax = fig.add_subplot(gs[1,i])
    ax.plot(fph, lc.flux, '.', color=color[i])
    axlc.plot(result.transit_times, depth*np.ones(len(result.transit_times)) - 2500e-6, '^', color=color[i])

    ax.set_xlim(-2.5*dur, 2.5*dur)
    ax.set_title(r'$P=%f$ / $T_0=%f$' % (period,t0))

    ax2 = fig.add_subplot(gs[2,i])
    ax2.plot(periods, power, '-', lw=1)

    tma = phase < dur
    lc  = TessLightCurve(time=lc.time[~tma], flux=lc.flux[~tma], flux_err=lc.flux_err[~tma])
    '''
    aux = lc.time[~tma]
    lc.time.value     = aux
    lc.flux     = lc.flux[~tma]
    lc.flux_err = lc.flux_err[~tma]
    '''

plt.show()
