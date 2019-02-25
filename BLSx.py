import glob
import argparse
from astropy.stats import BoxLeastSquares

parser = argparse.ArgumentParser(description='BLS for a folder with many LC files...')
parser.add_argument('Folder', help='Folder containing FITS files')
parser.add_argument('--mags', type=float, nargs=2, help='Magnitude limits')
parser.add_argument('--output', default=None)

args = parser.parse_args()

folder = args.Folder
if folder[-1] != '/':
    folder += '/'

allfiles = glob.glob(folder)
