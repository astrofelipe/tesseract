import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Separates in transit and out of transit parts of a light curve')
parser.add_argument('File', help='File containing the light curve')
parser.add_argument('t0', type=float, help='Ephemeris')
parser.add_argument('P', type=float, help='Period')
parser.add_argument('--dur', type=float, help='In transit region (same units of time)', default=0.25)

args = parser.parse_args()

t,f,e = np.genfromtxt(File, usecols=(0,1,2), unpack=True)
p = (t - args.t0 + 0.5*args.P) % args.P - 0.5*args.P
m = np.abs(p < args.dur)

data_in = np.transpose([t,f,e])[m]
data_out = np.transpose([t,f,e])[~m]

np.savetxt(File.replace('.','_in.'), data_in)
np.savetxt(File.replace('.','_out.'), data_out)