import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

parser = argparse.ArgumentParser(description='Calculates the PSD for the out of transit light curve')
parser.add_argument('File', help='File containing the light curve')
parser.add_argument('t0', help='Ephemeris')
parser.add_argument('P', help='Period')
parser.add_argument('--dur', type=float, help='x1.5 Duration of the transit', default=0.3)

args = parser.parse_args()


t,f,e = np.genfromtxt(args.File, usecols=(0,1,2), unpack=True)
phase = (t - args.t0 + 0.5*args.P) % args.P - 0.5*args.P
mask  = np.abs(phase) > args.dur

to, fo, eo = t[mask], f[mask], e[mask]

fig, ax = plt.subplots()
ax.plot(to, fo, '.k', ms=1)

plt.show()
