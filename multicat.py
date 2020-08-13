import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='Search for multisector targets based on catalogs')
parser.add_argument('Folder', type=str, help='Folder with LCs (organized by sector and magnitude)')
parser.add_argument('MinMag', type=int, help='Minimum magnitude')
parser.add_argument('MaxMag', type=int, help='Maximum magnitude')
parser.add_argument('Hemisphere', type=str, help='North or South')

args = parser.parse_args()

minmag = args.MinMag
maxmag = args.MaxMag

if args.Hemisphere is 'North':
    pass
else:
    minsec = 1
    maxsec = 13

catfiles = [args.Folder + 's%04d/s%04d_%d.000000-%d.000000.csv' % (s,s,minmag,maxmag) for s in range(minsec,maxsec+1)]
catalogs = [pd.read_csv(f, names=['ID', 'ra', 'dec', 'Tmag'], skiprows=1) for f in catfiles]
concat   = pd.concat(catalogs)

print(concat.ID.values_counts())
