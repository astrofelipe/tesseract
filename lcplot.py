import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

parser = argparse.ArgumentParser(description='Plot a LC')
parser.add_argument('TIC', type=int, help='TIC ID')
args = parser.parse_args()

files = glob.glob('/horus/TESS/LC/*/TIC%d.dat' % args.TIC)

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

plt.show()
