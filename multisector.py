import os
import glob
import argparse

parser = argparse.ArgumentParser(description='Find multiple sector targets and run TLS')
parser.add_argument('Folder', type=str, help='Folder with LCs (organized by sector and magnitude bin)')
parser.add_argument('MaxMag', type=int, help='Maximum magnitude. Minimum will be max-1 or -inf if it is 9')

args = parser.parse_args()

maxmag = str(args.MaxMag)
minmag = str(args.MaxMag - 1) if maxmag != 9 else 'inf'

all_files = glob.glob(args.Folder + '*/' + minmag + '-' + maxmag + '/')
print(all_files)
