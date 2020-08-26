import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='LC and transit plot (useful for TTV priors!)')
parser.add_argument('LCFiles', type=str, nargs='+', help='Light curve file(s)')
parser.add_argument('--transit', type=float, nargs=2, help='P and t0 values')

args = parser.parse_args()

fig, ax = plt.subplots(figsize=[15,3])


for f in args.LCFiles:
    #t,f,e = np.genfromtxt(f, usecols=(0,1,2), unpack=True)
    data = pd.read_csv(f, sep=' ', names=['time', 'flux', 'error', 'instrument'])
    inst = np.unique(data['instrument'])

    for i,it in enumerate(inst):
        ma = data['instrument'] == it
        t  = data['time'][ma]
        f  = data['flux'][ma]

        ax.plot(t,f,'.', ms=2, alpha=.6, label=it)

        if args.transit is not None:
            P, t0 = args.transit

            tmin = np.nanmin(t)
            tmax = np.nanmax(t)

            ttimes = t0 + np.arange(-100,100)*P
            tmask  = (ttimes > tmin) & (ttimes < tmax)
            ttimes = ttimes[tmask]

            print('Transit times\n')
            print(ttimes)

            for j,tt in enumerate(ttimes):
                ax.axvline(tt)
                ax.text(tt, 1-i*0.002, j, ha='right', va='center')

    ax.legend()

plt.show()
