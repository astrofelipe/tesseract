import glob
import argparse
import numpy as np
from astropy.utils.console import color_print
from astropy.io import fits

parser = argparse.ArgumentParser(description='Get CCD corner coordinates')
parser.add_argument('Folder', type=str, help='Folder with subfolders (sectors) containing FFIs')

args = parser.parse_args()

sector_folders = np.sort(glob.glob(args.Folder + 's00*'))

for sector in sector_folders:
    color_print(sector, 'cyan')
