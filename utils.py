#import celerite
import numpy as np
import h5py #Maybe separate this, as a lot of utils can work without ever using h5py or local files
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
#from celerite import terms
from scipy.optimize import minimize
from scipy.signal import medfilt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from lightkurve.targetpixelfile import TargetPixelFileFactory


def dilution_factor(m_primary, m_comp, sep, pixscale=21):
    fac  = 10**((m_primary - m_comp)/2.5) * np.exp(-1.68*sep/pixscale)
    dfac = 1/(1+np.sum(fac))
    return dfac


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

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def FFICut(ffis, x, y, size):
    with h5py.File(ffis, 'r', libver='latest') as ffis:

        ncads  = len(ffis['FFIs'])
        x      = int(x)
        y      = int(y)

        #aflux  = np.transpose(ffis['FFIs'], axes=[2,0,1])[:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]
        #aerrs  = np.transpose(ffis['errs'], axes=[2,0,1])[:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]
        aflux  = ffis['FFIs'][:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]
        aerrs  = ffis['errs'][:, x-size//2:x+size//2+1, y-size//2:y+size//2+1]

        boxing = TargetPixelFileFactory(n_cadences=ncads, n_rows=size, n_cols=size)

        for i,f in enumerate(tqdm(aflux)):
            ti = ffis['data'][0,i]
            tf = ffis['data'][1,i]
            b  = ffis['data'][2,i]
            q  = ffis['data'][3,i]

            header = {'TSTART': ti, 'TSTOP': tf,
                      'QUALITY': q}

            boxing.add_cadence(frameno=i, flux=f, flux_err=aerrs[i], header=header)

    TPF = boxing.get_tpf()
    #TPF.hdu[1].data['QUALITY']   = ffis['data'][2]
    #TPF.hdu[1].data['TIME']      = ffis['data'][0]
    #TPF.hdu[1].header['BJDREFI'] = hdr['BJDREFI']
    #TPF.hdu[1].data.columns['TIME'].unit = 'BJD - %d' % hdr['BJDREFI']

    return TPF

def pixel_border(mask):
    ydim, xdim = mask.shape

    x = []
    y = []

    for i in range(1,ydim-1):
        for j in range(1,xdim-1):
            if mask[i,j]:
                if not mask[i-1,j]:
                    x.append(np.array([j,j+1]))
                    y.append(np.array([i,i]))
                if not mask[i+1,j]:
                    x.append(np.array([j,j+1]))
                    y.append(np.array([i+1,i+1]))
                if not mask[i,j-1]:
                    x.append(np.array([j,j]))
                    y.append(np.array([i,i+1]))
                if not mask[i,j+1]:
                    x.append(np.array([j+1,j+1]))
                    y.append(np.array([i,i+1]))
    return x,y
