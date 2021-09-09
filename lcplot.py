import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='LC and transit plot (useful for TTV priors!)')
parser.add_argument('LCFiles', type=str, nargs='+', help='Light curve file(s)')
parser.add_argument('--transit', type=float, nargs=2, help='P and t0 values')

args = parser.parse_args()

fig, ax = plt.subplots(figsize=[15,3])

pd.options.display.float_format = '{:.6f}'.format
for f in args.LCFiles:
    #t,f,e = np.genfromtxt(f, usecols=(0,1,2), unpack=True)
    data = pd.read_csv(f, sep=' ', names=['time', 'flux', 'error', 'instrument'])
    inst = np.unique(data['instrument'])

    if np.sum(np.isnan(inst))!=0:
        inst = ['inst']

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

            print(it)
            print(pd.DataFrame(ttimes, columns=['time']))
            print()

            for j,tt in enumerate(ttimes):
                ax.axvline(tt)
                ax.text(tt, 1-i*0.002, j, ha='right', va='center')

    ax.legend()

plt.show()

'''
#LC: todos menos 11 y 16 (0,16)
#SC: 1, 4, 5, 6, 7, 8, 9
#CHAT y ELSAUCE: 0

CHAT
           time
0  2.458571e+06

ELSAUCE
           time
0  2.458892e+06

TESSLC
            time
0   2.458326e+06
1   2.458339e+06
2   2.458352e+06
3   2.458365e+06
4   2.458378e+06
5   2.458391e+06
6   2.458404e+06
7   2.458416e+06
8   2.458429e+06
9   2.458442e+06
10  2.458455e+06
11  2.458468e+06
12  2.458481e+06
13  2.458493e+06
14  2.458506e+06
15  2.458519e+06
16  2.458532e+06

TESSSC
           time
0  2.458558e+06
1  2.458571e+06
2  2.458583e+06
3  2.458596e+06
4  2.458609e+06
5  2.458622e+06
6  2.458635e+06
7  2.458648e+06
8  2.458660e+06
9  2.458673e+06
'''
