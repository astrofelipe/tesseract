import glob
import argparse
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.console import color_print
from astropy.io import fits, ascii

parser = argparse.ArgumentParser(description='Get CCD corner coordinates')
parser.add_argument('Folder', type=str, help='Folder with subfolders (sectors) containing FFIs')

args = parser.parse_args()

t = Table()

sector_folders = np.sort(glob.glob(args.Folder + 's00*/'))

for sector in sector_folders:
    sn = int(sector[-3:-1])
    color_print('\nSECTOR %d' % sn, 'cyan')

    for cam in range(1,5):
        for ccd in range(1,5):
            img = glob.glob(sector + 'tess*-%d-%d-*_ffic.fits' % (cam, ccd))[7]
            hdr = fits.getheader(img, 1)
            w   = WCS(hdr)

            T = np.transpose([w.all_pix2world(i,0,0) for i in range(44,2092,10)])
            R = np.transpose([w.all_pix2world(2091,i,0) for i in range(1,2048,10)])
            B = np.transpose([w.all_pix2world(i,2047,0) for i in range(2091,43,-10)])
            L = np.transpose([w.all_pix2world(44,i,0) for i in range(2046,0,-10)])

            TOP    = SkyCoord(T[0], T[1], unit='deg').transform_to('barycentrictrueecliptic')
            RIGHT  = SkyCoord(R[0], R[1], unit='deg').transform_to('barycentrictrueecliptic')
            BOTTOM = SkyCoord(B[0], B[1], unit='deg').transform_to('barycentrictrueecliptic')
            LEFT   = SkyCoord(L[0], L[1], unit='deg').transform_to('barycentrictrueecliptic')


            #ra1, dec1 = w.all_pix2world(44, 0, 0)       #TOP LEFT
            #ra2, dec2 = w.all_pix2world(44, 2047, 0)    #BOTTOM LEFT
            #ra3, dec3 = w.all_pix2world(2091, 2047, 0)  #BOTTOM RIGHT
            #ra4, dec4 = w.all_pix2world(2091, 0, 0)     #TOP RIGHT

            #Ecliptic limits:
            #eclat ~-72 a -inf (-90)

            color_print('CAM %d, CCD %d' % (cam, ccd), 'lightgreen')
            print('\t', TOP[0].lon.degree, TOP[0].lat.degree)
            print('\t', RIGHT[0].lon.degree, RIGHT[0].lat.degree)
            print('\t', BOTTOM[0].lon.degree, BOTTOM[0].lat.degree)
            print('\t', LEFT[0].lon.degree, LEFT[0].lat.degree)

            label    = 'S%02d_%d_%d' % (sn, cam, ccd)
            t[label+'_lon'] = [TOP.lon.degree, RIGHT.lon.degree, BOTTOM.lon.degree, LEFT.lon.degree]
            t[label+'_lat'] = [TOP.lat.degree, RIGHT.lat.degree, BOTTOM.lat.degree, LEFT.lat.degree]

ascii.write(t, 'corners.dat', format='commented_header')
