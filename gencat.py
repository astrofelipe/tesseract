from astroquery.mast import Catalogs, Tesscut
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import vstack, unique
from tqdm import tqdm
from tess_stars2px import tess_stars2px_function_entry as ts2p
from joblib import Parallel, delayed
import numpy as np
import argparse

#img = '/Volumes/Felipe/TESS/FFIs-5/tess2018345062939-s0005-3-3-0125-s_ffic.fits'
#img = '/Volumes/Felipe/TESS/FFIs-5/tess2018345092939-s0005-3-4-0125-s_ffic.fits'
#sec = img.split('-')[-5]
parser = argparse.ArgumentParser(description='Catalog generator')
parser.add_argument('Sector', type=int)
parser.add_argument('--min-mag', type=float, default=-999, help='Minimum magnitude')
parser.add_argument('--max-mag', type=float, default=14, help='Maximum magnitude')
parser.add_argument('--ncpu', type=int, default=5, help='Number of CPUs to use')

args = parser.parse_args()

sec = 's%04d' % args.Sector
print(sec)

img = '/horus/TESS/FFI/s0009/tess2019059192935-s0009-1-4-0139-s_ffic.fits'

hdr = fits.getheader(img, 0)
dat = fits.getdata(img)
w   = WCS(hdr)

'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.matshow(np.log10(dat), cmap='YlGnBu_r')
ax.plot([44], [0], '.r')
plt.show()
'''

ra1, dec1 = w.all_pix2world(44, 0, 0)       #TOP LEFT
ra2, dec2 = w.all_pix2world(44, 2047, 0)    #BOTTOM LEFT
ra3, dec3 = w.all_pix2world(2091, 2047, 0)  #BOTTOM RIGHT
ra4, dec4 = w.all_pix2world(2091, 0, 0)     #TOP RIGHT

print(ra1,ra2,ra3,ra4)
print(dec1,dec2,dec3,dec4)


#Ecliptic limits:
#eclat ~-72 a -inf (-90)
#eclon ver en cada sector (ancho ~90)

c1 = SkyCoord(ra1, dec1, unit='deg').transform_to('geocentrictrueecliptic')
c2 = SkyCoord(ra2, dec2, unit='deg').transform_to('geocentrictrueecliptic')
c3 = SkyCoord(ra3, dec3, unit='deg').transform_to('geocentrictrueecliptic')
c4 = SkyCoord(ra4, dec4, unit='deg').transform_to('geocentrictrueecliptic')
cx = SkyCoord(24.604344, -55.772082, unit='deg').transform_to('geocentrictrueecliptic')

print(cx)
print(c1, c2, c3, c4)


left   = np.min([ra1, ra2])
right  = np.max([ra3, ra4])
top    = np.max([dec1, dec4])
bottom = np.min([dec2, dec3])

print(left, right, top, bottom)

eclim = {'s0002': [[298, 388], [-90, 0]],
         's0005': [[19, 111], [-90, 0]],
         's0006': [[46, 140], [-90, 0]],
         's0007': [[74, 166], [-90, 0]],
         's0008': [[101, 193], [-90, 0]],
         's0009': [[], []]}

elo, ela = eclim[sec]

#Not pole
eclos = np.arange(elo[0], elo[1]+1.1, 4) % 360
eclas = np.arange(ela[0], ela[1]+1.1, 4)

wrapcheck = np.any(np.diff(eclos) < 0)
if wrapcheck:
    idx   = np.where(np.diff(eclos) < 0)[0]
    eclos = np.insert(eclos, idx+1, [360, 0])

print('Scanning... (1/2)')
def gocat(i, j):
    eloi1 = int(eclos[i])
    eloi2 = int(eclos[i+1])

    if eloi1==360 and eloi2==0:
        return

    elai1 = int(eclas[j])
    elai2 = int(eclas[j+1])

    catalogdata = Catalogs.query_criteria(catalog='Tic',
                                          eclong=[eloi1, eloi2],
                                          eclat=[elai1, elai2],
                                          Tmag=[args.min_mag, args.max_mag],
                                          objType='STAR')

    tics = np.array(catalogdata['ID'])
    ras  = np.array(catalogdata['ra'])
    dec  = np.array(catalogdata['dec'])

    res = ts2p(tics, ras, dec)

    sma = res[3] == args.Sector
    sid = res[0][sma]

    _, mask, _ = np.intersect1d(tics, sid, return_indices=True)

    catalogdata = catalogdata[mask]
    return catalogdata

supercata1 = vstack(Parallel(n_jobs=args.ncpu)(delayed(gocat)(i,j) for i in tqdm(range(len(eclos) - 1)) for j in tqdm(range(len(eclas) - 1))))

print(supercata1)
print('\nScanning... (2/2)')
#Pole
eclos = np.arange(0, 360+1.1, 5)
eclas = np.arange(-92, -72-1.1, 5)

supercata2 = vstack(Parallel(n_jobs=args.ncpu)(delayed(gocat)(i,j) for i in tqdm(range(len(eclos) - 1)) for j in tqdm(range(len(eclas) - 1))))
print(supercata2)
'''
for i in tqdm(range(len(eclos) - 1)):
    eloi1 = int(eclos[i])
    eloi2 = int(eclos[i+1])

    for j in tqdm(range(len(eclas) - 1)):
        elai1 = int(eclas[j])
        elai2 = int(eclas[j+1])

        catalogdata = Catalogs.query_criteria(catalog='Tic',
                                              eclong=[eloi1, eloi2],
                                              eclat=[elai1, elai2],
                                              Tmag=[args.min_mag, args.max_mag],
                                              objType='STAR')

        tics = np.array(catalogdata['ID'])
        ras  = np.array(catalogdata['ra'])
        dec  = np.array(catalogdata['dec'])

        res = ts2p(tics, ras, dec, scInfo=res[-1])

        sma = res[3] == 6
        sid = res[0][sma]

        _, mask, _ = np.intersect1d(tics, sid, return_indices=True)

        catalogdata = catalogdata[mask]

        supercata = vstack([supercata, catalogdata])
        supercata = unique(supercata, keys=['ID'])
'''

supercata = vstack([supercata1, supercata2])
print(supercata)
supercata = unique(supercata, keys=['ID'])

catalogfilt = supercata['ID', 'ra', 'dec', 'Tmag']
magord      = np.argsort(catalogfilt['Tmag'])
catalogfilt = catalogfilt[magord]
print(catalogfilt)
catalogfilt.write('%s_%f-%f.csv' % (sec, args.min_mag, args.max_mag), format='csv', overwrite=True)
