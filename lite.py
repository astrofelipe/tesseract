import __future__
import h5py
import argparse
import glob
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', type=int, help='TESS Sector')
parser.add_argument('Targets', type=str)
parser.add_argument('--output', action='store_true')

args = parser.parse_args()

fs  = np.sort(glob.glob('/horus/TESS/FFI/s%04d/*.hdf5' % args.Sector))
print(fs)
h5s = [h5py.File(f, 'r') for f in fs]

if args.Targets[-3:] == 'pkl':
    import pickle
    f = open(args.Targets, 'rb')
    d = pickle.load(f)
    tics = np.array([int(item) for item in d.keys()])

    svals = np.array([list(item.values()) for item in d.values()]).astype(bool)
    smask = svals[:,args.Sector-1]

    if args.output:
        np.savetxt('targets_s%04d.txt' % args.Sector, tics[smask], fmt='str')


else:
    tics = np.genfromtxt(args.Target, usecols=(0,), delim=',')
