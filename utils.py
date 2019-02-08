import celerite
import numpy as np
from celerite import terms
from scipy.optimize import minimize
from bls import BLS
from scipy.signal import medfilt

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
