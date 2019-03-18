import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot a LC')
parser.add_argument('File', type=str, help='Light curve filename')

args = parser.parse_args()

t, f = np.genfromtxt(args.File, usecols=(0,1), unpack=True)

fig, ax = plt.subplots(figsize=[10,3])
ax.plot(t, f, '-k')
ax.plot(t, f, 'ob', lw=3)

plt.show()
