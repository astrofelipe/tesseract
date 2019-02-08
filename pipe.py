import __future__
import glob
import argparse
import h5py
import os
import subprocess
import pandas as pd
from tqdm import trange
import numpy as np
from astropy import units as u
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils import Background2D, MMMBackground, SExtractorBackground, DAOStarFinder, CircularAperture, aperture_photometry
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
parser.add_argument('--krnl', type=int, default=2, help='Size of the kernel')
parser.add_argument('--stmp', type=int, default=3, help='Size of the stamp')
parser.add_argument('--ordr', type=int, default=0, help='Order')
parser.add_argument('--nrstars', type=int, default=500)

args = parser.parse_args()
rad  = args.aperture
nrstars = args.nrstars
stmp = args.stmp
krnl = args.krnl
ordr = args.ordr

#Compilar el programa que hace la diferenciacion (ojo con los paths de cfitiso)
compdiff = os.system('gcc -g oisdifference.c -L/usr/local/lib -I/usr/local/include -lcfitsio -lm -lcurl')

#Buscar archivos en la carpeta
folder = args.Folder
files  = np.sort(glob.glob('%s*ffic.fits' % folder))
print('Encontrados %d archivos' % len(files))

#Primera imagen se usa como referencia (works ok y menos atado que hacer un masterframe)
ref, rhead = fits.getdata(files[0], header=True)
mean, median, std = sigma_clipped_stats(ref, sigma=3, maxiters=5)
exptime = rhead['EXPOSURE']*3600.*24.

#Quitar el background (!)
masta = ref# - median

#Copia fits como reference frame (sin background)
mhd = fits.PrimaryHDU(masta, header=rhead)
mhd.writeto(folder + 'ref.fits', overwrite=True)

#Encuentra fuentes (asumiendo no se usara una star list)
daofind = DAOStarFinder(fwhm=1.5, threshold=9*std)
sources = daofind(masta)
pos     = np.array([sources['xcentroid'], sources['ycentroid']])
aps     = CircularAperture(pos, r=rad)

#Centroides xy a radec
w = WCS(rhead)
radec = w.all_pix2world(pos[0], pos[1], 1)
ra    = Column(radec[0], name='RA')
dec   = Column(radec[1], name='DEC')

#Supertabla!
supertable = Table((sources['id'], sources['xcentroid'], sources['ycentroid'],ra, dec))
supertable.write('supertable.hdf5', path='data', compression=True, overwrite=True)

del supertable
supertable = Table.read('supertable.hdf5', path='data')

#Itera sobre todas las imagenes (podria paralelizarse a coste de mas espacio en disco)
for f in files[1:]:
    hdus = fits.open(f)
    img  = hdus[1].data
    hdr  = hdus[1].header

    bjd = np.mean([hdr['TSTART'], hdr['TSTOP']])

    #Cut overscan
    #w     = WCS(hdr)
    #img   = Cutout2D(img, (1068, 1024), (2048, 2048), wcs=w).data

    mean, median, std = sigma_clipped_stats(img, sigma=3, maxiters=5)
    nimg = img# - median

    #Write img.fits
    ihd = fits.PrimaryHDU(nimg, header=hdr)
    ihd.writeto(folder+'img.fits', overwrite=True)

    rawflux = aperture_photometry(nimg, aps)

    flx  = rawflux['aperture_sum']
    flxe = np.sqrt(np.abs(flx))
    gdfl = np.where(flx > 0)

    if len(gdfl[0] > 0):
        flx  = flx[gdfl[0]]
        flxe = flxe[gdfl[0]]
        x, y = pos[:,gdfl[0]]

    mags = 25.0 - 2.5*np.log10(flx)
    mage = np.array((2.5/np.log(10))*(flxe/flx))

    #Seleccionar estrellas de referencia (aisladas)
    cnt = 0
    itr = 0

    XX   = np.transpose([x,y])
    tree = KDTree(XX, leaf_size=2)
    nn   = tree.query_radius(XX, r=3, count_only=True)

    #Mascara de vecinos cercanos (1) y de magnitud+posicion
    nnm = nn == 1
    pom = (mage > 0) & (mage < 0.02) & (x < 2040) & (x > 92) & (y < 1998) & (y > 50)

    xp, yp = x[nnm & pom], y[nnm & pom]

    #Seleccionamos nrstars al azar (o las que hayan sino)
    sel    = np.random.randint(0,len(xp),nrstars) if len(xp) > nrstars else True
    spos   = XX[sel]
    fns    = len(sel)

    #Guardar estrellas de referencia
    np.savetxt(folder + 'refstars.txt', spos, fmt='%4d %4d')

    #Param file
    output = open(folder + 'parms.txt', 'w')
    output.write('%1d %1d %1d %d\n' % (stmp, krnl, ordr, fns))
    output.close()

    #Reference image
    output = open(folder + 'ref.txt', 'w')
    output.write(folder + 'ref.fits\n')
    output.close()

    #Imagen creada ahora (sin background)
    output = open(folder + 'img.txt', 'w')
    output.write(folder + 'img.fits\n')
    output.close()

    #Diferenciacion!
    print('Diff!')
    #dodiff = os.system('./a.out')
    dodiff = subprocess.Popen(('./a.out'))
    dodiff.wait()
    mvdiff = subprocess.Popen(('mv', 'dimg.fits', str(bjd)+'.fits'))
    mvdiff.wait()

    #Se abre la imagen diferenciada...
    dif, dhd = fits.getdata(str(bjd)+'.fits', header=True)

    #Fotometria!
    rawflux = aperture_photometry(dif, aps)

    mean, median, std = sigma_clipped_stats(dif, sigma=3, maxiters=5)
    print(mean,median,std)

    #Agregamos a la tabla con el BJD como nombre
    supertable[str(bjd)] = rawflux['aperture_sum']
    del rawflux
    print(supertable)

    #deldiff = subprocess.Popen(('rm',folder+'dimg.fits'))
    #deldiff.wait()


print(supertable)
os.exit(1)

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
