#import celerite
import numpy as np
from tqdm import tqdm
#from celerite import terms
from scipy.optimize import minimize
from scipy.signal import medfilt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from lightkurve.targetpixelfile import KeplerTargetPixelFileFactory

def mask_planet(t, t0, period, dur=0.25):
    phase  = (t - t0 + 0.5*period) % period - 0.5*period
    mask   = np.abs(phase) < dur

    return ~mask

'''
def BLSer(t, y, yerr, mw=351, maximum_period=30.):
    #Input needs to be normalized

    yr = (y / np.nanmedian(y) -1)*1e6
    yt = medfilt(yr, mw)
    y  = yr - yt

    durations = np.linspace(0.05, 0.2, 10)
    model     = BLS(t, y)
    results   = model.autopower(durations, maximum_period=maximum_period, frequency_factor=5.0)

    #TIC Period t0 Duration Depth SNR Depth_even Depth_odd Depth_half
    idx    = np.argmax(results.power)
    period = results.period[idx]
    t0     = results.transit_time[idx]
    depth  = results.depth[idx]
    dur    = results.duration[idx]
    SNR    = results.depth_snr[idx]

    return period, t0, dur, depth, SNR

def detrender(t, y, yerr):
    kernel = terms.Matern32Term(log_sigma=np.log(np.nanvar(y)), log_rho=-np.log(10.0)) + terms.JitterTerm(log_sigma=np.log(np.nanvar(y)))
    gp     = celerite.GP(kernel)#, mean=mean_model, fit_mean=True)
    gp.compute(t, yerr)

    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]

    initial_params = gp.get_parameter_vector()
    bounds         = gp.get_parameter_bounds()
    soln           = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method='L-BFGS-B', bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(soln.x)

    mu, var = gp.predict(y, t, return_var=True)
    std     = np.sqrt(var)

    return mu, std
'''

def FFICut(fnames, ra, dec, size):
    boxing = KeplerTargetPixelFileFactory(n_cadences=len(fnames), n_rows=size, n_cols=size)
    quals  = np.ones(len(fnames)) * np.nan

    for i,f in tqdm(enumerate(fnames), total=len(fnames)):
        hdu = fits.open(f)

        dflux   = hdu[1].data
        derr    = hdu[2].data
        hdr     = hdu[1].header

        if i==0:
            w      = WCS(hdr)
            x,y = w.all_world2pix(ra, dec, 0)
        #x  += 1
        #x = int(x) + 0.5
        #y = int(y) + 0.5

        cutflux = Cutout2D(dflux, (x-1,y), size, mode='trim', wcs=w)
        cuterrs = Cutout2D(derr, (x-1,y), size, mode='trim')

        quals[i] = hdr['DQUALITY']
        boxing.add_cadence(frameno=i, flux=cutflux.data, flux_err=cuterrs.data, header=hdr, wcs=cutflux.wcs)

    TPF = boxing.get_tpf()
    TPF.hdu[1].data['QUALITY']   = quals
    TPF.hdu[1].header['BJDREFI'] = hdr['BJDREFI']
    TPF.hdu[1].data.columns['TIME'].unit = 'BJD - %d' % hdr['BJDREFI']

    return TPF, cutflux.wcs

def pixel_border(mask):
    ydim, xdim = mask.shape

    x = []
    y = []

    for i in range(1,ydim-1):
        for j in range(1,xdim-1):
            if mask[i,j]:
                if not mask[i-1,j]:
                    x.append(np.array([j-0.5,j+0.5]))
                    y.append(np.array([i-0.5,i-0.5]))
                if not mask[i+1,j]:
                    x.append(np.array([j-0.5,j+0.5]))
                    y.append(np.array([i+0.5,i+0.5]))
                if not mask[i,j-1]:
                    x.append(np.array([j-0.5,j-0.5]))
                    y.append(np.array([i-0.5,i+0.5]))
                if not mask[i,j+1]:
                    x.append(np.array([j+0.5,j+0.5]))
                    y.append(np.array([i-0.5,i+0.5]))
    return x,y
