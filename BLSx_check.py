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

if not args.nogaia:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    from astroquery.gaia import Gaia

for i in range(args.start, len(BLSdata)):
    print('\nIteration: ',i)

    fig = plt.figure(constrained_layout=True, figsize=[8*2, 6*1.5])
    gs  = GridSpec(12, 8, figure=fig)
    lcs = [fig.add_subplot(gs[2*k:2*(k+1),:4]) for k in range(6)]
    lcf = [fig.add_subplot(gs[2*k:2*(k+1),4]) for k in range(6)]
    lc2 = [fig.add_subplot(gs[2*k:2*(k+1),5]) for k in range(6)]

    opc = [fig.add_subplot(gs[2*k,6]) for k in range(6)]
    oeb = [fig.add_subplot(gs[2*k+1,6]) for k in range(6)]
    orr = [fig.add_subplot(gs[2*k,7]) for k in range(6)]
    oot = [fig.add_subplot(gs[2*k+1,7]) for k in range(6)]

    bpc = [Button(opc[k], 'Planet') for k in range(6)]
    beb = [Button(oeb[k], 'Eclipsing Binary') for k in range(6)]
    brr = [Button(orr[k], 'RR Lyrae') for k in range(6)]
    bot = [Button(oot[k], 'Other Variable') for k in range(6)]

    chunk = BLSdata[6*i:6*(i+1)]

    for j in range(6):
        fn     = chunk['Files'][j]#chunk.iloc[j,0]
        period = chunk['P'][j]#chunk.iloc[j,1]
        t0     = chunk['t0'][j]#chunk.iloc[j,2]
        depth  = chunk['depth'][j]#chunk.iloc[j,4]

        t, y = np.genfromtxt(fn, unpack=True, usecols=(0,1))
        lc   = TessLightCurve(time=t, flux=y).flatten()

        p    = (t - t0 + 0.5*period) % period - 0.5*period
        p2   = (t - t0 + period) % period - 0.5*period

        if not args.nogaia:
            obj   = (fn.split('/')[-1])[3:-4]
            chunk.iloc[j,0] = obj

            cdata = Catalogs.query_object('TIC' + obj, radius=0.018, catalog='Gaia')
            rval  = cdata[0]['radius_val']
            chunk.iloc[j]['rval'] = rval

        else:
            rval = np.nan

        lcs[j].plot(lc.time, lc.flux, '.', ms=1)
        if t0 > 50:
            transits = np.arange(t0, t[-1], period)
            for t0s in transits:
                lcs[j].axvline(t0s, color='orange')
        lcs[j].set_xlim(np.nanmin(t), np.nanmax(t))
        lcs[j].set_ylabel('Norm Flux (ppm)')

        lcf[j].plot(p, lc.flux, '.', ms=1)
        lcf[j].set_xlim(-0.2, 0.2)
        lcf[j].set_ylim(1-1.5*depth, 1.005)

        lc2[j].plot(p2, lc.flux, '.', ms=1)
        lc2[j].set_xlim(-0.2, 0.2)

        #inf[j].text(0.5, 0.5, r'$P=%f$' % period, ha='center', va='center', transform=inf[j].transAxes)
        #inf[j].set_axis_off()

        lcs[j].set_title(r'%s  /  $P=%f$  /  Depth$=%f$  /  $R_{\star}=%f$  /  $R_p = %f$' % (fn, period, depth, rval, rval*np.sqrt(depth)*9.95))

        chunk['Files'][j] = chunk['Files'][j].split('TIC')[-1].split('.')[0]
    chunk['duration'] = chunk['duration']*24
    chunk['depth'] = int(chunk['depth']*1e6)
    print(chunk)
    plt.show()
    plt.close(fig)
