import glob
import argparse
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.utils.console import color_print
from astropy.io import fits

parser = argparse.ArgumentParser(description='Get CCD corner coordinates')
parser.add_argument('Folder', type=str, help='Folder with subfolders (sectors) containing FFIs')

args = parser.parse_args()

sector_folders = np.sort(glob.glob(args.Folder + 's00*/'))

for sector in sector_folders:
    sn = int(sector[-3:-1])
    color_print('SECTOR %d' % sn, 'cyan')

    for cam in range(1,5):
        for ccd in range(1,5):
            print(sector + 'tess*-%d-%d-*_ffic.fits' % (cam, ccd))
            img = glob.glob(sector + 'tess*-%d-%d-*_ffic.fits' % (cam, ccd))[5]
            hdr = fits.getheader(img, 1)
            w   = WCS(hdr)

            ra1, dec1 = w.all_pix2world(44, 0, 0)       #TOP LEFT
            ra2, dec2 = w.all_pix2world(44, 2047, 0)    #BOTTOM LEFT
            ra3, dec3 = w.all_pix2world(2091, 2047, 0)  #BOTTOM RIGHT
            ra4, dec4 = w.all_pix2world(2091, 0, 0)     #TOP RIGHT

            #Ecliptic limits:
            #eclat ~-72 a -inf (-90)

            TL = SkyCoord(ra1, dec1, unit='deg').transform_to('barycentrictrueecliptic')
            BL = SkyCoord(ra2, dec2, unit='deg').transform_to('barycentrictrueecliptic')
            BR = SkyCoord(ra3, dec3, unit='deg').transform_to('barycentrictrueecliptic')
            TR = SkyCoord(ra4, dec4, unit='deg').transform_to('barycentrictrueecliptic')

            color_print('\tCAM %d, CCD %d' % (cam, ccd), 'lightgreen')
            print('\t\t', TL.lat, TL.lon)
            print('\t\t', BL)
            print('\t\t', BR)
            print('\t\t', TR)
