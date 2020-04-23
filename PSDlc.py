import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from astropy.timeseries import LombScargle

parser = argparse.ArgumentParser(description='Calculates the PSD for the out of transit light curve')
parser.add_argument('File', help='File containing the light curve')
parser.add_argument('t0', type=float, help='Ephemeris')
parser.add_argument('P', type=float, help='Period')
parser.add_argument('--dur', type=float, help='x1.5 Duration of the transit', default=0.3)

args = parser.parse_args()


t,f,e = np.genfromtxt(args.File, usecols=(0,1,2), unpack=True)
phase = (t - args.t0 + 0.5*args.P) % args.P - 0.5*args.P
mask  = np.abs(phase) > args.dur

to, fo, eo = t[mask], f[mask], e[mask]

fig, ax = plt.subplots(figsize=[10,3])
ax.errorbar(to, fo, yerr=eo, fmt='.k', ms=1, alpha=.66)

tlim = np.max(to) - np.min(to)
fnyq = (60*24/2)
print(1/fnyq, tlim)
PSDe = np.mean(np.var(eo**2)) / fnyq
pers = np.linspace(1/fnyq, tlim)
freq = np.linspace(1/tlim, fnyq, 10000)
pow  = LombScargle(to, fo, eo, normalization='psd').power(freq)
#freq, pow = LombScargle(to, fo, eo, normalization='psd').autopower()

fig, ax = plt.subplots(figsize=[6,3])
ax.plot(pers, pow, '-k')
#ax.axhline(PSDe, ls='--')
#ax.axvline(1/args.P, c='r')


plt.show()
