import __future__
import h5py
import glob
import argparse
import numpy as np
from mpi4py import MPI
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

allfiles  = np.sort(glob.glob(args.Folder + '*-%d-%d-*.fits' % (args.Camera, args.Chip)))[args.nstart:args.nstop]
files     = allfiles[args.nstart:args.nstop]
nfiles    = len(allfiles)

nprocs = MPI.COMM_WORLD.size
rank   = MPI.COMM_WORLD.rank

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

def get_data(f, ext=1):
    hdu = fits.open(f, memmap=args.nomemmap)
    dat = hdu[ext].data
    hdu.close()
    del hdu
    return np.array(dat)


nx, ny = fits.getdata(files[0]).shape

#output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w', libver='latest')
output = h5py.File('TESS-FFIs_s%04d-%d-%d.hdf5' % (args.Sector, args.Camera, args.Chip), 'w', driver='mpio', comm=MPI.COMM_WORLD)

dset   = output.create_dataset('FFIs', (nfiles, nx, ny), dtype='float64', chunks=True)#, compression='gzip')
derr   = output.create_dataset('errs', (nfiles, nx, ny), dtype='float64', chunks=True)#, compression='gzip')
table  = output.create_dataset('data', (4, nfiles), dtype='float64', chunks=True)#, compression='gzip')

#dset[args.nstart:args.nstop] = Parallel(n_jobs=args.ncpu)(delayed(get_data)(f) for f in tqdm(files))
#derr[args.nstart:args.nstop] = Parallel(n_jobs=args.ncpu)(delayed(get_data)(f, 2) for f in tqdm(files))
#dset[args.nstart:args.nstop] = [get_data(f) for f in tqdm(files)]
#derr[args.nstart:args.nstop] = [get_data(f, 2) for f in tqdm(files)]

for i,f in enumerate(tqdm(files)):
    if i % nprocs == rank:
        hdu  = fits.open(f, memmap=True)
        dat1 = hdu[1].data
        dat2 = hdu[2].data

        dset[i] = dat1
        derr[i] = dat2
        table[:,i] = make_table(f)

        hdu.close()
        del hdu, dat1, dat2


#table[args.nstart:args.nstop] = np.transpose(Parallel(n_jobs=args.ncpu)(delayed(make_table)(f) for f in tqdm(files)))


print(dset)
dset.flush()
output.close()
