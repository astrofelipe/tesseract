import __future__
import h5py
import argparse
import glob
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Targets', type=str)

args = parser.parse_args()

fs  = np.sort(glob.glob('/horus/TESS/FFI/s%04d/*.h5py' % args.Sector))
h5s = h5py.File(f for f in fs)

if args.Targets[-3:] == 'pkl':
    import pickle
    f = open(args.Targets, 'rb')
    d = pickle.load(f)
    tics = d.keys()

    sval = np.array([list(item.values()) for item in d.values()]).astype(bool)

    print(tics)
    print(sval)

else:
    tics = np.genfromtxt(args.Target, usecols=(0,), delim=',')
