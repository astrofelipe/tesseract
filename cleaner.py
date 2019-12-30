import numpy as np
import argparse

def cleaner(time, flux):

    stdm = (flux < 0.98) & (time > 2458380) & (time < 2458415)

    mask1  = ((time > 2458325) & (time < 2458326)) + ((time > 2458347) & (time < 2458350)) + ((time > 2458352.5) & (time < 2458353.2))
    mask3  = ((time > 2458382) & (time < 2458384)) + ((time > 2458407) & (time < 2458410)) + ((time > 2458393.5) & (time < 2458397))
    mask4  = ((time > 2458419) & (time < 2458422)) + ((time > 2458422) & (time < 2458424)) + ((time > 2458436) & (time < 2458437))
    mask5  = ((time > 2458437.8) & (time < 2458438.7)) + ((time > 2458450) & (time < 2458452)) + ((time > 2458463.4) & (time < 2458464.2))
    mask6  = ((time > 2458476.7) & (time < 2458478.7)) #check 2do
    mask7  = ((time > 2458491.6) & (time < 2458492)) + ((time > 2458504.6) & (time < 2458505.2))
    mask8  = ((time > 2458517.4) & (time < 2458518)) + ((time > 2458530) & (time < 2458536))
    mask9  = ((time > 2458543) & (time < 2458545)) + ((time > 2458556) & (time < 2458559))
    mask10 = ((time > 4913400) & (time < 4913403.5)) + ((time > 4913414.2) & (time < 4913417))
    mask11 = ((time > 2458610.6) & (time < 2458611.6)) + ((time > 2458610.6) & (time < 2458611.6)) #check 2do
    mask12 = ((time > 2458624.5) & (time < 2458626)) + ((time > 2458639) & (time < 2458642))
    mask13 = ((time > 2458657.5) & (time < 2458659)) + ((time > 2458667) & (time < 2458669)) + ((time > 2458681) & (time < 2458683)) #checked

    mask   = mask1 + mask3 + mask4 + mask5 + mask6 + mask7 + mask8 + mask9 + mask10 + mask11 + mask12 + mask13 + stdm

    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cuts systematics and other fluxes')
    parser.add_argument('File', help='Light curve file')

    args = parser.parse_args()

    data = np.genfromtxt(args.File, usecols=(0,1,2))
    strd = np.genfromtxt(args.File, usecols=(3), dtype='str')
    print(strd.shape, data.shape)
    time = data[:,0]
    flux = data[:,1]

    mask = cleaner(time, flux)

    strd2 = strd[~mask]
    data2 = data[~mask]

    finaldata = np.c_[data2, strd2]
    np.savetxt(args.File.replace('.dat','clean.dat'), finaldata, fmt='%s')
