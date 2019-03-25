import __future__
import h5py
import glob
import argparse

parser = argparse.ArgumentParser(description='FFIs to single h5 file')
parser.add_argument('Folder', type=str)
parser.add_argument('Camera', type=int)
parser.add_argument('Chip', type=int)

files = glob.glob(args.Folder + '*%d-%d*.fits' % (args.Camera, args.Chip))

print(files)
