import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot a LC')
parser.add_argument('File', type=str, help='Light curve filename')

args = parser.parse_args()

t, f = np.genfromtxt(args.File, usecols=(0,1), unpack=True)

fig, ax = plt.subplots(figsize=[15,3])
ax.plot(t, f, '-k', zorder=-2)
ax.scatter(t, f, c='gold', edgecolor='black', s=10, lw=.5, zorder=-1)

plt.show()
