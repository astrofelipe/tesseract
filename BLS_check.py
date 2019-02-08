import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from matplotlib.gridspec import GridSpec
from astropy.table import Table
from astropy.io import fits

#pd.set_option('display.max_columns', 50)
pd.set_option('display.expand_frame_repr', False)

parser = argparse.ArgumentParser(description='BLS check')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Filename', type=str, help='BLS output')
parser.add_argument('--start', type=int, default=0, help='Iteration to start from')
parser.add_argument('--max-depth', type=float, default=1e6, help='Maximum depth to consider')
parser.add_argument('--min-period', type=float, default=0, help='Minimum period to consider')
parser.add_argument('--merged', action='store_true', help='Use this if the file contains results over merged lcs')

args = parser.parse_args()
sect = args.Sector

results = pd.read_csv(args.Filename, delimiter=' ')

themask = (results['Depth'] < args.max_depth) & (results['Period'] > args.min_period)
permask = (np.abs(results['Period'] - 27.5) > 0.5)
results = results[themask & permask]

print results.columns

for i in xrange(args.start, len(results)): #Nunca llegara alguien al final?
    print '\nIteration: ',i
    fig = plt.figure(constrained_layout=True, figsize=[7*4, 6*3])
    gs  = GridSpec(6, 7, figure=fig)
    lcs = [fig.add_subplot(gs[k,:5]) for k in range(6)]
    lcf = [fig.add_subplot(gs[k,5]) for k in range(6)]
    inf = [fig.add_subplot(gs[k,6]) for k in range(6)]

    chunk = results[6*i:6*(i+1)]
    amags = np.zeros(6)

    for j in range(6):
        datic = chunk.iloc[j,0]
        fname = glob.glob('/data/TESS/LC/official/s*/*%d*_lc.fits' % datic)[0]
        hdus  = fits.open(fname)
        data  = hdus[1].data
        tmag  = hdus[0].header['TESSMAG']
        amags[j] = tmag

        period = chunk.iloc[j,1]
        t0     = chunk.iloc[j,2]
        depth  = chunk.iloc[j,4]

        if args.merged:
            fmerged = glob.glob('/data/TESS/LC/official/merged/*%d*.dat' % datic)[0]
            t, y, e = np.genfromtxt(fmerged, unpack=True)
        else:
            y = data['PDCSAP_FLUX']
            t = data['TIME']
        y = (y / np.nanmedian(y) - 1)*1e6
        p = (t - t0 + 0.5*period) % period - 0.5*period

        trend = medfilt(y, 751)

        lcs[j].plot(t, y, '.', ms=.5, alpha=.66)
        lcs[j].set_xlim(np.nanmin(t), np.nanmax(t))
        lcs[j].set_ylabel('Norm Flux (ppm)')

        lcf[j].plot(p, y - trend, '.', ms=.5, alpha=.66)
        lcf[j].set_xlim(-0.2, 0.2)

        inf[j].text(0.5, 0.5, r'$P=%f$' % period, ha='center', va='center', transform=inf[j].transAxes)
        inf[j].set_axis_off()

        lcs[j].set_title(r'TIC $%d$  /  $P=%f$  /  Depth$=%f$ / TESSMAG$=%f$' % (datic, period, depth, tmag))
  
    chunk['TESSMAG'] = amags
    print chunk
    plt.show()
    plt.close(fig)
    
