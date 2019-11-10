import glob
import argparse
from tess_stars2px import tess_stars2px_function_entry as ts2p

parser = argparse.ArgumentParser(description='Aperture preview (JPG)')
parser.add_argument('Folder', type=str, help='Folder with LCs')

args = parser.parse_args()

ffipath = args.Folder.replace('LC','FFI').split('/')
print(ffipath)

lcs = glob.glob(args.Folder + 'TIC*.dat')


def generate_jpg(fn):
    TIC = int(fn.split('TIC')[-1].split('.')[0])
