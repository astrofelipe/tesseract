import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from astropy.io import ascii
from matplotlib.widgets import Button
from lightkurve.lightcurve import TessLightCurve
from matplotlib.gridspec import GridSpec


parser = argparse.ArgumentParser(description='BLS reviewer')
parser.add_argument('File', help='BLS output')
parser.add_argument('--max-depth', type=float, default=0.04, help='Maximum depth allowed')
parser.add_argument('--min-period', type=float, default=0, help='Minimum period')
parser.add_argument('--max-period', type=float, default=999, help='Maximum period')
parser.add_argument('--start', type=int, default=0, help='Iteration to start from')
parser.add_argument('--ntra', type=int, default=2, help='Minimum number of transits')
parser.add_argument('--nogaia', action='store_true')

args = parser.parse_args()

names   = ['Files', 'P', 't0', 'duration', 'depth', 'snr', 'depth_even', 'depth_odd', 'depth_half', 't1', 'ntra']
#BLSdata = pd.read_csv(args.File, delimiter=' ', names=names)
BLSdata = ascii.read(args.File, names=names)
mask    = ((BLSdata['ntra'] >= args.ntra) & (BLSdata['depth'] < args.max_depth) & (BLSdata['P'] > args.min_period) & (BLSdata['P'] < args.max_period))# + ((np.abs(BLSdata['P']) - 13.4) > 0.55)
BLSdata = BLSdata[mask]

plt.style.use('bmh')

if not args.nogaia:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    from astroquery.gaia import Gaia

nlc = 7

for i in range(args.start, len(BLSdata)):
    print('\nIteration: ',i)

    fig = plt.figure(constrained_layout=True, figsize=[22, 1.5*nlc], dpi=80)
    gs  = GridSpec(ncols=9, nrows=nlc, figure=fig, width_ratios=[9,2,2,1,0.5,9,2,2,1])

    lcs = np.ravel([fig.add_subplot(gs[k%nlc, 5*(k//nlc)]) for k in range(2*nlc)])
    lcf = np.ravel([fig.add_subplot(gs[k%nlc, 1+5*(k//nlc)]) for k in range(2*nlc)])
    lc2 = np.ravel([fig.add_subplot(gs[k%nlc, 2+5*(k//nlc)]) for k in range(2*nlc)])
    opr = np.ravel([fig.add_subplot(gs[k%nlc, 3+5*(k//nlc)]) for k in range(2*nlc)])
    pbu = [Button(opr[k], 'Print') for k in range(2*nlc)]

    chunk = BLSdata[2*nlc*i:2*nlc*(i+1)]

    funcs = []

    for j in range(2*nlc):
        fn     = chunk['Files'][j]#chunk.iloc[j,0]
        period = chunk['P'][j]#chunk.iloc[j,1]
        t0     = chunk['t0'][j]#chunk.iloc[j,2]
        depth  = chunk['depth'][j]#chunk.iloc[j,4]

        t, y = np.genfromtxt(fn, unpack=True, usecols=(0,1))
        lc   = TessLightCurve(time=t, flux=y).flatten()
        obj  = (fn.split('/')[-1])[3:-4]
        chunk['Files'][j] = obj

        p    = (t - t0 + 0.5*period) % period - 0.5*period
        p2   = (t - t0 + period) % period - 0.5*period

        if not args.nogaia:
            cdata = Catalogs.query_object('TIC' + obj, radius=0.018, catalog='Gaia')
            rval  = cdata[0]['radius_val']
            #chunk[j]['rval'] = rval

        else:
            rval = np.nan

        lcs[j].plot(lc.time, lc.flux, '.', ms=1)
        if t0 > 50:
            transits = np.arange(t0, t[-1], period)
            for t0s in transits:
                lcs[j].axvline(t0s, lw=6, alpha=.3, color='tomato')
        lcs[j].set_xlim(np.nanmin(t), np.nanmax(t))
        lcs[j].set_ylabel('Norm Flux (ppm)', fontsize=10)
        lcs[j].set_ylim(1-2*depth, 1+depth)

        lcf[j].plot(p, lc.flux, '.', ms=1.5)
        lcf[j].set_xlim(-0.2, 0.2)
        lcf[j].set_ylim(1-1.5*depth, 1.005)

        lc2[j].plot(p2, lc.flux, '.', ms=1.5)
        lc2[j].set_xlim(-0.2, 0.2)
        lc2[j].set_ylim(1-0.5*depth, 1.005)



        chunk['depth'][j] = chunk['depth'][j]*1e6
        lcs[j].set_title(r'$%s$  /  $P=%f$  /  Depth$=%f$  /  $R_{\star}=%f$  /  $R_p = %f$' % (obj, period, depth, rval, rval*np.sqrt(depth)*9.95), fontsize=10)

        def on_press(event):
            #chunk['Files', 'P', 't0', 'duration', 'depth'][j].pprint(show_name=False, align='<')
            hey = chunk[j]['Files', 'P', 't0', 'duration', 'depth']
            print(hey)

        funcs.append(on_press)

        pbu[j].on_clicked(funcs[j])

    chunk['duration'] = chunk['duration']*24
    chunk['duration'].format = '%.2f'
    chunk['depth'].format    = '%d'

    print(chunk)
    #gs.tight_layout(fig)
    plt.show()
    plt.close(fig)
