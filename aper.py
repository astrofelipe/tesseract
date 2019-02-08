import __future__
import glob
import argparse
import h5py
import os
import subprocess
import pandas as pd
from tqdm import tqdm
import numpy as np
from astropy import units as u
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, RectangularAperture, aperture_photometry, SExtractorBackground
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from astropy.table import Table, Column
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Clean and align images')
parser.add_argument('Folder', type=str, help='Folder containing all .fits FFI images for same camera and chip')
parser.add_argument('--ncpu', type=int, default=8, help='Number of CPUs to use')
parser.add_argument('--aperture', type=float, default=3.5, help='Optimal aperture')

args = parser.parse_args()
rad  = args.aperture

#Buscar archivos en la carpeta
folder = args.Folder
files  = np.sort(glob.glob('%s*ffic.fits' % folder))[200:600]
print('Encontrados %d archivos' % len(files))

#Imagen de referencia para buscar fuentes, se toman 50 al azar y la mediana
print('Generando Masterframe...')
refsel = np.random.choice(files, size=50, replace=False)
refmed = np.nanmedian([fits.getdata(f) for f in refsel], axis=0)
mean, median, std = sigma_clipped_stats(refmed, sigma=3, maxiters=5)

#Encuentra fuentes (asumiendo no se usara una star list)
print('Buscando fuentes...')
print(std)
daofind = DAOStarFinder(fwhm=3.5, threshold=200)#threshold=8000)
sources = daofind(refmed)
x, y    = sources['xcentroid'], sources['ycentroid']
pom     = (x < 2090) & (x > 42) & (y < 1998) & (y > 50)
pos     = np.array([x[pom], y[pom]])

print('\tEncontradas %d fuentes' % len(pos[0]))

#Aperutras y mascaras para bkg
aps     = CircularAperture(pos, r=rad)

bkg_ap = RectangularAperture(pos, h=13, w=13)
bkg_ma = bkg_ap.to_mask(method='center')

#Centroides xy a radec
print('\tCalculando RADEC...')
rhead = fits.getheader(files[0], 1)
w     = WCS(rhead)
radec = w.all_pix2world(pos[0], pos[1], 1)
ra    = Column(radec[0], name='RA')
dec   = Column(radec[1], name='DEC')

#Supertabla!
postable = Table((sources['id'][pom], sources['xcentroid'][pom], sources['ycentroid'][pom], ra, dec))
postable.write('positions.dat', format='ascii.commented_header', overwrite=True)
del postable

#Super Numpy Array!
#supertable = np.memmap('supertable.dat', dtype='float64', mode='w+', shape=(len(pos[0])+2, len(files)))

superfile  = h5py.File('supertable.hdf5', 'w')
supertable = superfile.create_dataset('data', (len(pos[0])+2, len(files)), dtype='float64')
refimg     = superfile.create_dataset('ref', refmed.shape, dtype='float64')
refimg[:]  = refmed

bfile = h5py.File('backgrounds.hdf5', 'r')

#Tiempos y Quality
print('Comenzando fotometria...')
for i,f in enumerate(tqdm(files)):
    hdus = fits.open(f)
    data = hdus[1].data
    hdr  = hdus[1].header

    tstart = hdr['TSTART']
    tstop  = hdr['TSTOP']
    bjd    = np.mean([tstart, tstop])
    dqual  = hdr['DQUALITY']

    supertable[0,i] = bjd
    supertable[1,i] = dqual

    bkg = bfile['bkgs'][i]
    data = data# - bkg
    
    rawflux = aperture_photometry(data, aps, method='center')

    '''
    bkg_median = []
    for bm in bkg_ma:
        bk_dat = bm.multiply(data)
        bk_dat = bk_dat[bm.data > 0]
        ssc    = SigmaClip(sigma=3.)
        msc    = SExtractorBackground(ssc).calc_background(bk_dat)
        #_, msc, _ = sigma_clipped_stats(bk_dat)
        bkg_median.append(msc)
    bkg_median = np.array(bkg_median)
    bkg_final  = bkg_median*aps.area()
    '''
    
    final_flux = np.array(rawflux['aperture_sum'])# - bkg_final)
    supertable[2:,i] = final_flux

    del final_flux, rawflux#, bkg_final

print(supertable)
print(supertable.shape)
print(supertable[2:].max(), supertable[2:].min())

fig, ax = plt.subplots()
ax.matshow(np.log10(refmed))
ax.plot(pos[0], pos[1], '.r', ms=1)
plt.show()
superfile.close()
