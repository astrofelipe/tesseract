import __future__
import glob
import argparse
import h5py
import os
import pandas as pd
from tqdm import trange
import numpy as np
from astropy import units as u
from FITS_tools.hcongrid import hcongrid
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils import Background2D, MMMBackground, SExtractorBackground, DAOStarFinder, CircularAperture, aperture_photometry
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
parser.add_argument('--notalign',  action='store_true', help='Do not align, just save original image on hdf5')
args = parser.parse_args()

nx  = args.nx
ny  = args.ny
notalign = args.notalign

folder = args.Folder
files  = np.sort(glob.glob('%s*ffic.fits' % folder))
print('Encontrados %d archivos' % len(files))

ref, rhead = fits.getdata(files[0], header=True)
rhead['CRPIX1'] = 1001.
rhead['NAXIS1'] = nx
rhead['NAXIS2'] = ny

def align_one(f):
    hdus = fits.open(f)
    img  = hdus[1].data
    imge = hdus[2].data
    hdr  = hdus[1].header

    w     = WCS(hdr)
    cut   = Cutout2D(img, (1068, 1024), (nx, ny), wcs=w)
    cute  = Cutout2D(imge, (1068, 1024), (nx, ny), wcs=w)
    bimg  = cut.data
    bimge = cute.data

    #Update header
    hdr['CRPIX1'] = 1001.
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny

    #Background? This may fail with bright stars
    mask       = ~np.isfinite(bimg)
    mask      |= (bimg > 8e4) #Flux cutoff
    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
    bkg_estim  = MMMBackground()

    #This should work...
    bkg = Background2D(bimg, (64, 64),
                filter_size=(15,15),
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estim,
                mask=mask, exclude_percentile=50)

    #Maybe apply a filter before?
    bimg    -= bkg.background

    #Heh...
    #bimg[bimg < 0] = 1

    #Align
    aligni = hcongrid(bimg, hdr, rhead) if not notalign else bimg
    aligne = hcongrid(bimge, hdr, rhead) if not notalign else bimge

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
    return np.array([aligni, aligne])

def chunk_median(i):
    r = np.min([length, (i+1)*args.blocksize])
    return np.nanmedian(imgs[i*args.blocksize:r], axis=0)

aligned = np.array(Parallel(n_jobs=args.ncpu, verbose=5)(delayed(align_one)(f) for f in files[:100]))
length  = len(aligned)

path = os.path.dirname(files[0])

opt  = h5py.File(path + '/aligned.hdf5', 'w', libver='latest')
imgs = opt.create_dataset('imgs', (length, nx, ny), 'f')
imge = opt.create_dataset('imgs_err', (length, nx, ny), 'f')

for i in trange(length):
    imgs[i] = aligned[i,0]
    imge[i] = aligned[i,1]
del aligned

nchunks = np.ceil(length/float(args.blocksize)).astype(int)
medians = Parallel(n_jobs=args.ncpu, verbose=7)(delayed(chunk_median)(i) for i in range(nchunks))
mframe  = np.nanmedian(medians, axis=0)

mastah = opt.create_dataset('mframe', (nx,ny), 'f')
mastah[:] = mframe


w = WCS(rhead)
'''
corners = np.array([[0,0], [nx,0], [0,ny], [nx,ny]])
cradec  = np.transpose(w.all_pix2world(corners, 1))
cminra  = np.min(cradec[0])
cmaxra  = np.max(cradec[0])
cminde  = np.min(cradec[1])
cmaxde  = np.max(cradec[1])


tics = pd.read_csv('GI_S001.csv')
ra   = tics['RA']*u.degree
dec  = tics['Dec']*u.degree

tx, ty = w.wcs_world2pix(ra, dec, 1)
'''

mean, median, std = sigma_clipped_stats(mframe, sigma=3, maxiters=5)
print('Std: ', std, 9*std)
daofind = DAOStarFinder(fwhm=1.5, threshold=9*std)
sources = daofind(mframe - median)
xx, yy  = (sources['xcentroid'], sources['ycentroid'])

rx, ry = w.all_pix2world(xx, yy, 1)
yii = 14500
print(rx[yii], xx[yii])
print(ry[yii], yy[yii])
print(rx.size)
print(mean,median,std)
print(mframe.max(), mframe.min())

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.matshow(np.log10(mframe), cmap='bone')
ax.scatter(xx[yii], yy[yii], s=20, facecolor='none', edgecolor='cyan', zorder=999)
#ax.scatter(tx, ty, s=20, facecolor='none', edgecolor='k')
#ax.plot(xx, yy, '.', ms=.5, color='r')

ax.set_xlim(0,2048)
ax.set_ylim(0,2048)
plt.show()

opt.close()
