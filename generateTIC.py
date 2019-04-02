import dask.dataframe as dd
import glob
import argparse


parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('Folder', type=str)

args = parser.parse_args()

files = glob.glob(args.Folder + 'tic_*.csv')

df = dd.read_csv(files[0])

print(df)
