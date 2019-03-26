import __future__
import h5py
import glob
import argparse
import numpy as np
from joblib import Parallel, delayed
from astropy.io import fits
from tqdm import tqdm

parser = argparse.ArgumentParser(description='FFIs to single h5 file')
parser.add_argument('Folder', type=str)
parser.add_argument('Sector', type=int)
parser.add_argument('Camera', type=int)
parser.add_argument('Chip', type=int)
parser.add_argument('--ncpu', type=int, default=10)
args = parser.parse_args()

files  = np.sort(glob.glob(args.Folder + '*%d-%d*.fits' % (args.Camera, args.Chip)))
nfiles = len(files)

for i,f in enumerate(tqdm(files)):
    dat = fits.getdata(f)

    if i==0:
        nx, ny = dat.shape
        output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w')
        dset   = output.create_dataset('FFIs', (nfiles, nx, ny), dtype='f', compression='gzip')

    dset[i] = dat
    del dat

'''
nx, ny = fits.getdata(files[0]).shape

output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w')
dset   = output.create_dataset('FFIs', (nfiles, nx, ny), dtype='f', compression='gzip')

dset[:] = Parallel(n_jobs=args.ncpu)(delayed(fits.getdata)(f) for f in tqdm(files))
'''


print(dset)
dset.flush()
output.close()
