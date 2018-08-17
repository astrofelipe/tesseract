import glob
import argparse
import h5py
import os
from tqdm import trange
import numpy as np
from FITS_tools.hcongrid import hcongrid
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Clean and align images')
parser.add_argument('Folder', type=str, help='Folder containing all .fits FFI images')
parser.add_argument('--ncpu', type=int, default=8, help='Number of CPUs to use')
parser.add_argument('--nx', type=int, default=2048, help='X dimension length')
parser.add_argument('--ny', type=int, default=2048, help='Y dimension length')
parser.add_argument('--blocksize', type=int, default=50, help='Blocksize for masterframe calculation')
args = parser.parse_args()

nx  = args.nx
ny  = args.ny

folder = args.Folder
files  = np.sort(glob.glob('%s*ffic.fits' % folder))

ref, rhead = fits.getdata(files[0], header=True)
rhead['CRPIX1'] = 1001.
rhead['NAXIS1'] = nx
rhead['NAXIS2'] = ny

def align_one(f):
    img, hdr = fits.getdata(f, header=True)
    w    = WCS(hdr)
    cut  = Cutout2D(img, (1068, 1024), (nx, ny), wcs=w)
    bimg = cut.data

    #Update header
    hdr['CRPIX1'] = 1001.
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny

    #Background?

    #Align
    align = hcongrid(bimg, hdr, rhead)

    #Update header
    hdr['CTYPE1'] = rhead['CTYPE1']
    hdr['CTYPE2'] = rhead['CTYPE2']
    hdr['CRVAL1'] = rhead['CRVAL1']
    hdr['CRVAL2'] = rhead['CRVAL2']
    hdr['CRPIX1'] = rhead['CRPIX1']
    hdr['CRPIX2'] = rhead['CRPIX2']
    hdr['CD1_1'] = rhead['CD1_1']
    hdr['CD1_2'] = rhead['CD1_2']
    hdr['CD2_1'] = rhead['CD2_1']
    hdr['CD2_2'] = rhead['CD2_2']

    #Write
    #shd = fits.PrimaryHDU(align, header=hdr)
    #shd.writeto(f.replace('ffic', 'ffic_AL'), overwrite=True)

    #Return for hdf5
    return align

def chunk_median(i):
    r = np.min([len(dset), (i+1)*args.blocksize])
    return np.nanmedian(dset[i*args.blocksize:r], axis=0)

aligned = Parallel(n_jobs=args.ncpu, verbose=5)(delayed(align_one)(f) for f in files)
length  = len(aligned)

path = os.path.dirname(files[0])
oupt = h5py.File(path + '/aligned.hdf5', 'w')

dset = oupt.create_dataset('imgs', (length,nx,ny), 'f')
for i in trange(length):
    dset[i] = aligned[i]
del aligned

nchunks = np.ceil(length/float(args.blocksize)).astype(int)
medians = Parallel(n_jobs=args.ncpu, verbose=5)(delayed(chunk_median)(i) for i in range(nchunks))
mframe  = np.nanmedian(medians, axis=0)

mastah = oupt.create_dataset('mframe', (nx,ny), 'f')
mastah[:] = mframe
print mastah

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.matshow(np.log10(mastah), cmap='bone')
plt.show()

oupt.close()
