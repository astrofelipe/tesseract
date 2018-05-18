import glob
import argparse
import numpy as np
from FITS_tools.hcongrid import hcongrid
from astropy.wcs import WCS

parser = argparse.ArgumentParser(description='Clean and align images')
parser.add_argument('Folder', type=str, help='Folder containing all .fist FFI images')
args = parser.parse_args()

folder = args.folder
files  = np.sort(glob.glob('%s*.fits' % folder))

ref, rhead = fits.getdata(files[0], header=True)
rhead['CRPIX1'] = 1001.
rhead['NAXIS1'] = 2048
rhead['NAXIS2'] = 2048

for i,f in enumerate(files):
    img, hdr = fits.getdata(f, header=True)
    #w = WCS(hdr)

    #Update header
    hdr['CRPIX1'] = 1001.
    hdr['NAXIS1'] = 2048
    hdr['NAXIS2'] = 2048

    #Align
    align = hcongrid(img, header, rhead)

    #Update header
    header['CTYPE1'] = rhead['CTYPE1']
	header['CTYPE2'] = rhead['CTYPE2']
	header['CRVAL1'] = rhead['CRVAL1']
	header['CRVAL2'] = rhead['CRVAL2']
	header['CRPIX1'] = rhead['CRPIX1']
	header['CRPIX2'] = rhead['CRPIX2']
	header['CD1_1'] = rhead['CD1_1']
	header['CD1_2'] = rhead['CD1_2']
    header['CD2_1'] = rhead['CD2_1']
    header['CD2_2'] = rhead['CD2_2']

    #Write
    shd = fits.PrimaryHDU(align, header=header)
    shd.writeto('test%d.fits' % i, overwrite=True)
