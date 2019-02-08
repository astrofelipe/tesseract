import glob
import h5py
import numpy as np
import os
#from OJH_estimate import fit_background
#from RH_estimate import fit_background
from tqdm import tqdm
from joblib import Parallel, delayed
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils import make_source_mask, Background2D, SExtractorBackground, MMMBackground, RectangularAperture

files = np.sort(glob.glob('*ffic.fits'))[:100]
print('Encontrados %d archivos' % len(files))

test = fits.getdata(files[0])
size = test.shape

f = h5py.File('backgrounds.hdf5', 'w')
#bkgs = f.create_dataset('bkgs', (len(files), size[0], size[1]), dtype='float64')

pos = np.genfromtxt('positions.dat', unpack=True, usecols=(1,2))
aps = RectangularAperture(pos, 13, 13)
bkg_mask = aps.to_mask(method='center')

bkgs = f.create_dataset('bkgs', (len(files), len(pos[0])), dtype='float64')
#all_bkg = np.zeros((len(files), len(pos[0])))

def calc_bkg(fl):
    data = fits.getdata(fl)
    #mask = make_source_mask(data, snr=2, npixels=3, dilate_size=4)
    #mean, median, std= sigma_clipped_stats(
    '''
    mask = np.zeros(data.shape)
    mask[:45] = 1
    mask[2093:] = 1
    mask[:,2049:] = 1

    sigma_clip = SigmaClip(sigma=2.5)
    bkg_estimator = MMMBackground()#SExtractorBackground()
    bkg = Background2D(data, (104,148), filter_size=(1,1), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask).background
    '''
    bkg = fit_background(data)

    np.save(fl.replace('.fits', '_bkg'), bkg)
    del bkg, data

def calc_single_bkg(data, stamp):
    aper_data = stamp.multiply(data)
    aper_data = aper_data[stamp.data > 0]

    sigma_clip = SigmaClip(sigma=2.5)
    bkg = MMMBackground(sigma_clip=sigma_clip)
    return bkg.calc_background(aper_data)


#dummy = Parallel(n_jobs=10, backend='multiprocessing')(delayed(calc_bkg)(fl) for fl in tqdm(files))

#files = np.sort(glob.glob('*bkg.npy'))
#print('Encontrados %d backgrounds' % len(files))

for i,fl in enumerate(tqdm(files)):
    #data = np.load(fl)
    data = fits.getdata(fl)
    #print(data)
    bk = np.array(Parallel(n_jobs=10, backend='threading')(delayed(calc_single_bkg)(data, stamp) for stamp in bkg_mask))
    bkgs[i] = bk
    del data, bk

f.close()
#os.system('rm *bkg.npy')
    
    
