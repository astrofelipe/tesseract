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
parser.add_argument('--nomemmap', action='store_false')
parser.add_argument('--nstart', type=int, default=0)
parser.add_argument('--nstop', type=int, default=None)
args = parser.parse_args()

allfiles  = np.sort(glob.glob(args.Folder + '*%d-%d*.fits' % (args.Camera, args.Chip)))
files     = allfiles[args.nstart:args.nstop]
nfiles    = len(allfiles)

'''
for i,f in enumerate(tqdm(files)):
    hdu = fits.open(f, memmap=False)

    flu = hdu[1].data
    err = hdu[2].data
    hdr = hdu[1].header

    if i==0:
        nx, ny = flu.shape
        output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w')
        dset   = output.create_dataset('FFIs', (nfiles, nx, ny), dtype='float64', compression='gzip')
        derr   = output.create_dataset('errs', (nfiles, nx, ny), dtype='float64', compression='gzip')
        table  = output.create_dataset('data', (3, nfiles), dtype='float64', compression='gzip')

    dset[i] = flu
    derr[i] = err

    output['data'][0,i] = 0.5*(hdr['TSTART'] + hdr['TSTOP']) + hdr['BJDREFI']
    output['data'][1,i] = hdr['BARYCORR']
    output['data'][2,i] = hdr['DQUALITY']

    del flu, err, hdr, hdu

'''

def make_table(f):
    hdr = fits.getheader(f, 1)

    #t = 0.5*(hdr['TSTART'] + hdr['TSTOP']) + hdr['BJDREFI']
    ti = hdr['TSTART'] + hdr['BJDREFI']
    tf = hdr['TSTOP'] + hdr['BJDREFI']
    #c  = hdr['FFIINDEX']
    b  = hdr['BARYCORR']
    q  = hdr['DQUALITY']
    #p1 = hdr['POS_CORR1']
    #p2 = hdr['POS_CORR2']

    return ti,tf,b,q


nx, ny = fits.getdata(files[0]).shape

output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w', libver='latest')
dset   = output.create_dataset('FFIs', (nfiles, nx, ny), dtype='float64', compression='lzf')
derr   = output.create_dataset('errs', (nfiles, nx, ny), dtype='float64', compression='lzf')
table  = output.create_dataset('data', (4, nfiles), dtype='float64', compression='lzf')

dset[args.nstart:args.nstop] = Parallel(n_jobs=args.ncpu)(delayed(fits.getdata)(f, memmap=args.nomemmap) for f in tqdm(files))
derr[args.nstart:args.nstop] = Parallel(n_jobs=args.ncpu)(delayed(fits.getdata)(f, 2, memmap=args.nomemmap) for f in tqdm(files))

table[args.nstart:args.nstop] = np.transpose(Parallel(n_jobs=args.ncpu)(delayed(make_table)(f) for f in tqdm(files)))


print(dset)
dset.flush()
output.close()
