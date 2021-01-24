from astroquery.mast import Catalogs, Tesscut
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import vstack, unique
from tqdm import tqdm
from tess_stars2px import tess_stars2px_function_entry as ts2p
from joblib import Parallel, delayed
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Catalog generator')
parser.add_argument('Sector', type=int)
parser.add_argument('--min-mag', type=float, default=-999, help='Minimum magnitude')
parser.add_argument('--max-mag', type=float, default=14, help='Maximum magnitude')
parser.add_argument('--ncpu', type=int, default=5, help='Number of CPUs to use')

args = parser.parse_args()

sec = 's%04d' % args.Sector
print(sec)

tots = 1 if args.Sector < 14 else 2

if (args.Sector < 14) or (args.Sector > 26):
    TOP_LEFT     = glob.glob('/horus/TESS/FFI/%s/*%d-3-3*' % (sec, args.Sector))[0]
    BOTTOM_RIGHT = glob.glob('/horus/TESS/FFI/%s/*%d-3-4*' % (sec, args.Sector))[0]
else:
    TOP_LEFT     = glob.glob('/horus/TESS/FFI/%s/*%d-1-1*' % (sec, args.Sector))[0]
    BOTTOM_RIGHT = glob.glob('/horus/TESS/FFI/%s/*%d-1-2*' % (sec, args.Sector))[0]

for img in [TOP_LEFT, BOTTOM_RIGHT]:
    print(img)
    hdr = fits.getheader(img, 1)
    dat = fits.getdata(img)
    w   = WCS(hdr)

    ra1, dec1 = w.all_pix2world(44, 0, 0)       #TOP LEFT
    ra2, dec2 = w.all_pix2world(44, 2047, 0)    #BOTTOM LEFT
    ra3, dec3 = w.all_pix2world(2091, 2047, 0)  #BOTTOM RIGHT
    ra4, dec4 = w.all_pix2world(2091, 0, 0)     #TOP RIGHT

    c1 = SkyCoord(ra1, dec1, unit='deg').transform_to('geocentrictrueecliptic')
    c2 = SkyCoord(ra2, dec2, unit='deg').transform_to('geocentrictrueecliptic')
    c3 = SkyCoord(ra3, dec3, unit='deg').transform_to('geocentrictrueecliptic')
    c4 = SkyCoord(ra4, dec4, unit='deg').transform_to('geocentrictrueecliptic')
    print(c1, c2, c3, c4)


    left   = np.min([ra1, ra2])
    right  = np.max([ra3, ra4])
    top    = np.max([dec1, dec4])
    bottom = np.min([dec2, dec3])

eclim = {'s0001': [[271, 361], [-90, 0]],
         's0002': [[298, 388], [-90, 0]],
         's0003': [[326, 415], [-90, 0]],
         's0004': [[345, 450], [-90, 0]],
         's0005': [[19, 111], [-90, 0]],
         's0006': [[46, 140], [-90, 0]],
         's0007': [[74, 166], [-90, 0]],
         's0008': [[101, 193], [-90, 0]],
         's0009': [[125, 218], [-90, 0]],
         's0010': [[154, 244], [-90, 0]],
         's0011': [[181, 271], [-90, 0]],
         's0012': [[209, 299], [-90, 0]],
         's0013': [[235, 330], [-90, 0]],
         's0014': [[275, 340], [0, 90]],
         's0015': [[324, 370], [0, 90]],#65,190
         's0016': [[330, 390], [0, 90]],
         's0017': [[350, 410], [0, 90]],
         's0018': [[30, 80], [0, 90]],
         's0019': [[60, 100], [0, 90]],
         's0020': [[80, 130], [0,90]],
         's0021': [[110, 160], [0, 90]],
         's0022': [[140, 190], [0, 90]],
         's0023': [[160, 210], [0, 90]],
         's0024': [[180, 260], [0, 90]],
         's0025': [[210, 280], [0, 90]],
         's0026': [[240, 300], [0, 90]],
         's0027': [[271, 320], [-90, 0]],
         's0028': [[270, 370], [-90, 0]],
         's0029': [[290, 400], [-90, 0]],
         's0030': [[325, 425], [-90, 0]],
         's0031': [[355, 450], [-90, 0]]}

elo, ela = eclim[sec]
print(elo,ela)

def gocat(i, j, im):
    eloi1 = int(eclos[i])
    eloi2 = int(eclos[i+1])

    elai1 = int(eclas[j])
    elai2 = int(eclas[j+1])

    dec_max = 30 if args.Sector > 13 else 90 #This is redundant...
    catalogdata = Catalogs.query_criteria(catalog='Tic',
                                          eclong=[eloi1, eloi2],
                                          eclat=[elai1, elai2],
                                          dec=[-90, dec_max],
                                          Tmag=[magbin[im], magbin[im+1]],
                                          objType='STAR')

    return catalogdata

def stacker(catalogs):
    catalogs = vstack(catalogs)

    tics = np.array(catalogs['ID'])
    ras  = np.array(catalogs['ra'])
    dec  = np.array(catalogs['dec'])

    res = ts2p(tics, ras, dec, trySector=args.Sector)
    sid = res[0]

    _, mask, _ = np.intersect1d(tics, sid, return_indices=True)

    return catalogs[mask]


print('Scanning... (1/%d)' % tots)
#Not pole
eclos = np.linspace(elo[0], elo[1]+1.1, 10) % 360
eclas = np.linspace(ela[0], ela[1]+1.1, 10)

wrapcheck = np.any(np.diff(eclos) < 0)
if wrapcheck:
    idx   = np.where(np.diff(eclos) < 0)[0]
    eclos = np.insert(eclos, idx+1, [360, 0])

#Magnitudes (safer?)
magbin = np.linspace(args.min_mag, args.max_mag, 4)

supercata1 = Parallel(n_jobs=args.ncpu)(delayed(gocat)(i,j,im) for im in tqdm(range(len(magbin) - 1))
                                                               for i in range(len(eclos) - 1)
                                                               for j in range(len(eclas) - 1))

#Search pole only if southern hemisphere
if args.Sector < 14:
    print('\nScanning... (2/%d)' % tots)
    #Pole
    eclos = np.arange(0, 361, 5)
    eclas = np.arange(-90, -71, 5)

    supercata2 = Parallel(n_jobs=args.ncpu)(delayed(gocat)(i,j,im) for im in tqdm(range(len(magbin) - 1))
                                                                   for i in range(len(eclos) - 1)
                                                                   for j in range(len(eclas) - 1))
    supercata2 = stacker(supercata2)



    supercata = vstack([supercata1, supercata2], silent=True)

else:
    supercata = stacker(supercata1)
    print(supercata1)


if len(supercata) > 0:
    supercata = unique(supercata, keys=['ID'])

    catalogfilt = supercata['ID', 'ra', 'dec', 'Tmag']
    magord      = np.argsort(catalogfilt['Tmag'])
    catalogfilt = catalogfilt[magord]
    print(catalogfilt)

    catalogfilt.write('%s_%f-%f.csv' % (sec, args.min_mag, args.max_mag), format='csv', overwrite=True)
