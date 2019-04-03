import dask.dataframe as dd
import pandas as pd
import glob
import argparse


parser = argparse.ArgumentParser(description='Extract Lightcurves from FFIs')
parser.add_argument('Folder', type=str)

args = parser.parse_args()

names = ['ID', 'version', 'HIP', 'TYC', 'UCAC', '2MASS', 'SDSS', 'ALLWISE', 'GAIA', 'APASS', 'KIC',
         'objType', 'typeSrc', 'ra', 'dec', 'POSflag', 'pmRA', 'e_pmRA', 'pmDEC', 'e_pmDEC', 'PMflag',
         'plx', 'e_plx', 'PARflag', 'gallong', 'gallat', 'eclong', 'eclat', 'Bmag', 'e_Bmag',
         'Vmag', 'e_Vmag', 'umag', 'e_umag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag',
         'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag', 'TWOMflag', 'prox', 'w1mag', 'e_w1mag',
         'w2mag', 'e_w2mag', 'w3mag', 'e_w3mag', 'w4mag', 'e_w4mag', 'GAIAmag', 'e_GAIAmag',
         'Tmag', 'e_Tmag', 'TESSflag', 'SPFlag', 'Teff', 'e_Teff', 'logg', 'e_logg', 'MH', 'e_MH',
         'rad', 'e_rad', 'mass', 'e_mass', 'rho', 'e_rho', 'lumclass', 'lum', 'e_lum', 'd', 'e_d',
         'ebv', 'e_ebv', 'numcont', 'contratio', 'disposition', 'duplicate_i', 'priority', 'objID']

files = glob.glob(args.Folder + 'tic_*.csv')

df = pd.read_csv(files[0])
print(len(names), len(df.columns))
print(df.columns)
print(df['eclong'], df['eclat'])

print(df.head())
