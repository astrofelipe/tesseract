import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from matplotlib.gridspec import GridSpec


parser = argparse.ArgumentParser(description='BLS reviewer')
parser.add_argument('File', help='BLS output')
parser.add_argument('--max-depth', type=float, default=0.04, help='Maximum depth allowed')
parser.add_argument('--min-period', type=float, default=0, help='Minimum period')
parser.add_argument('--max-period', type=float, default=999, help='Maximum period')
parser.add_argument('--start', type=int, default=0, help='Iteration to start from')
parser.add_argument('--nogaia', action='store_true')

args = parser.parse_args()

names   = ['Files', 'P', 't0', 'duration', 'depth', 'snr', 'depth_even', 'depth_odd', 'depth_half']
BLSdata = pd.read_csv(args.File, delimiter=' ', names=names)
mask    = ((BLSdata['depth'] < args.max_depth) & (BLSdata['P'] > args.min_period) & (BLSdata['P'] < args.max_period))# + ((np.abs(BLSdata['P']) - 13.4) > 0.55)
BLSdata = BLSdata[mask]

if not args.nogaia:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    from astroquery.gaia import Gaia

for i in range(args.start, len(BLSdata)):
    print('\nIteration: ',i)

    fig = plt.figure(constrained_layout=True, figsize=[7*4, 6*3])
    gs  = GridSpec(6, 7, figure=fig)
    lcs = [fig.add_subplot(gs[k,:5]) for k in range(6)]
    lcf = [fig.add_subplot(gs[k,5]) for k in range(6)]
    lc2 = [fig.add_subplot(gs[k,6]) for k in range(6)]

    chunk = BLSdata[6*i:6*(i+1)]

    for j in range(6):
        fn     = chunk.iloc[j,0]
        period = chunk.iloc[j,1]
        t0     = chunk.iloc[j,2]
        depth  = chunk.iloc[j,4]

        t, y = np.genfromtxt(fn, unpack=True, usecols=(0,1))
        p    = (t - t0 + 0.5*period) % period - 0.5*period
        p2   = (t - t0 + period) % period - 0.5*period

        if not args.nogaia:
            obj   = (fn.split('/')[-1])[3:-4]
            chunk.iloc[j,0] = obj

            cdata = Catalogs.query_object('TIC' + obj, radius=0.018, catalog='Gaia')
            rval  = cdata[0]['radius_val']
            chunk[j]['rval'] = rval

        else:
            rval = np.nan

        lcs[j].plot(t, y, '.', ms=1)
        lcs[j].set_xlim(np.nanmin(t), np.nanmax(t))
        lcs[j].set_ylabel('Norm Flux (ppm)')

        lcf[j].plot(p, y, '.', ms=1)
        lcf[j].set_xlim(-0.2, 0.2)

        lc2[j].plot(p2, y, '.', ms=1)
        lc2[j].set_xlim(-0.2, 0.2)

        #inf[j].text(0.5, 0.5, r'$P=%f$' % period, ha='center', va='center', transform=inf[j].transAxes)
        #inf[j].set_axis_off()

        lcs[j].set_title(r'%s  /  $P=%f$  /  Depth$=%f$  /  $R_{\star}=%f$' % (fn, period, depth, rval))

    print(chunk)
    plt.show()
    plt.close(fig)
