import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from astropy.stats import BoxLeastSquares as BLS

parser = argparse.ArgumentParser(description='Plot a LC')
parser.add_argument('TIC', type=int, help='TIC ID')
args = parser.parse_args()

files = np.sort(glob.glob('/horus/TESS/LC/*/TIC%d.dat' % args.TIC))

t = []
f = []

for fn in files:
    ti, fi = np.genfromtxt(fn, usecols=(0,1), unpack=True)
    t.append(ti)
    f.append(fi / np.nanmedian(fi))

t = np.concatenate(t)
f = np.concatenate(f)

fig, ax = plt.subplots(figsize=[15,3])
ax.plot(t, f, '-k', lw=1, zorder=-2)
ax.scatter(t, f, c='gold', edgecolor='black', s=15, lw=.5, zorder=-1)

durations = np.linspace(0.05, 0.2, 50)# * u.day
model     = BLS(t,f)
result    = model.autopower(durations, frequency_factor=5.0, maximum_period=30.0)
idx       = np.argmax(result.power)

period = result.period[idx]
t0     = result.transit_time[idx]
dur    = result.duration[idx]
depth  = result.depth[idx]

ph = (t - t0 + 0.5*period) % period - 0.5*period

fig2, ax2 = plt.subplots(figsize=[8,2])
ax2.scatter(ph, f, c='gold', edgecolor='black', s=15, lw=.5)

plt.show()
