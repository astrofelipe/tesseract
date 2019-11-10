import glob
import argparse
import numpy as np
from TLSx import the_TLS

parser = argparse.ArgumentParser(description='Run TLS over selected multisector LCs')
parser.add_argument('Folder', type=str, help='Folder with LCs (organized by sector and magnitude)')
parser.add_argument('File', type=str, help='Multisector list')
parser.add_argument('--target', type=int, default=None, help='Runs on single target')

args = parser.parse_args()

if args.target:
    fns = glob.glob(args.Folder + '*/*/TIC%d.dat' % args.target)
    print(fns)
