import glob
import argparse
from astropy.io import fits

parser = argparse.ArgumentParser(description='Get CCD corner coordinates')
parser.add_argument('Folder', type=str, help='Folder with subfolders (sectors) containing FFIs')

args = parser.parse_args()

sector_folders = glob.glob(args.Folder + 's00*')
print(sector_folders)

#for sector in range(1,14):
