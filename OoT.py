import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from astropy.timeseries import LombScargle
from astropy.table import Table

parser = argparse.ArgumentParser(description='Calculates the PSD for the out of transit light curve')
parser.add_argument('File', help='File containing the light curve')
parser.add_argument('t0', type=float, help='Ephemeris')
parser.add_argument('P', type=float, help='Period')
parser.add_argument('--dur', type=float, help='x1.5 Duration of the transit', default=0.3)

args = parser.parse_args()


#t,f,e = np.genfromtxt(args.File, usecols=(0,1,2), unpack=True)
#insts = np.genfromtxt(args.File, usecols=(3,), dtype=str)
data = Table.read(args.File, format='ascii.no_header')

t,f,e = data['col1'], data['col2'], data['col3']
phase = (t - args.t0 + 0.5*args.P) % args.P - 0.5*args.P
mask  = np.abs(phase) > 3*args.dur

data2 = data[mask]

'''
fig, ax = plt.subplots(figsize=[10,3])
#ax.errorbar(to, fo, yerr=eo, fmt='.k', ms=1, alpha=.66)
ax.plot(to, fo, '.k', ms=1, alpha=.66)

tlim = int(np.max(to) - np.min(to))
fnyq = (60*24/4)
print(1/fnyq, tlim)
print(1/tlim, fnyq)
PSDe = np.mean(np.var(eo**2)) / 0.004

ls = LombScargle(to, fo, eo, normalization='psd')

pers = np.linspace(1/fnyq, tlim, 10000)
freq = np.linspace(1/tlim, fnyq, 10000)
pow  = ls.power(1/pers)
fap  = ls.false_alarm_probability(pow.max())
#freq, pow = LombScargle(to, fo, eo, normalization='psd').autopower()

fig, ax = plt.subplots(figsize=[6,3])
ax.plot(pers, pow, '-k')
#ax.axhline(fap, ls='--')
#ax.axvline(1/args.P, c='r')


plt.show()
'''

data2.write(args.File.replace('.dat', '_OoT.dat'), format='ascii.no_header')
