import __future__
import h5py
import argparse
import glob
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Generate lightcurves!')
parser.add_argument('Sector', help='TESS Sector')

args = parser.parse_args()

fs  = np.sort(glob.glob(args.Folder + '*.h5py'))
h5s = h5py.File(f for f in fs)
