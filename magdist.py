import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Magnitude distribution from catalog')
parser.add_argument('File', help='Folder containing FITS files')

args = parser.parse_args()

data = pd.read_csv(args.File, comment='#')

fig, ax = plt.subplots()
bins    = np.arange(0,21)
print(bins)

ax.hist(data['Tmag'], bins=bins)
#ax.set_yscale('log')

plt.show()
