import glob
import argparse
import numpy as np
from lightkurve.lightcurve import TessLightCurve
from TLSx import the_TLS
from cleaner import cleaner

parser = argparse.ArgumentParser(description='Run TLS over selected multisector LCs')
parser.add_argument('Folder', type=str, help='Folder with LCs (organized by sector and magnitude)')
parser.add_argument('File', type=str, help='Multisector list')
parser.add_argument('--target', type=int, default=None, help='Runs on single target')

args = parser.parse_args()

subfolder = args.File.split('_')[-1].split('.')[0]

if args.target:
    fns = glob.glob(args.Folder + '*/%s/TIC%d.dat' % (subfolder, args.target))

    t,f,e = [],[],[]

    for fn in fns:
        tt,ff,ee = np.genfromtxt(fn, unpack=True, usecols=(0,1,2))
        if np.median(tt) > 3000000:
            tt -= 2454833

        lc = TessLightCurve(time=tt, flux=ff, flux_err=ee).flatten(window_length=51, polyorder=2, niters=5)
        t.append(lc.time)
        f.append(lc.flux)
        e.append(lc.flux_err)

    t = np.concatenate(t)
    f = np.concatenate(f)
    e = np.concatenate(e)


    cm = cleaner(t, f)
    t  = t[~cm]
    f  = f[~cm]
    e  = e[~cm]
    print(the_TLS(fn,t,f,e))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(t, f, '.k')

    plt.show()
