#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function for estimation of sky background in TESS Full Frame Images.

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0
.. versionchanged:: 1.1

Notes: Copied over from TASOC/photometry/backgrounds.py [15/01/18]

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>

"""
#TODO: Use the known locations of bright stars
#TODO: Add testing call function

import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.io.fits as pyfits
from photutils import Background2D, SExtractorBackground
from astropy.stats import SigmaClip

from Functions import *

def fit_background(ffi):
    """
    Estimate the background of a Full Frame Image (FFI) using the photutils package.
    This method uses the photoutils Background 2D background estimation using a
    SExtracktorBackground estimator, a 3-sigma clip to the data, and a masking
    of all pixels above 3e5 in flux.

    Parameters:
        ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

    Returns:
        ndarray: Estimated background with the same size as the input image.

    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
    .. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
    """

    mask = ~np.isfinite(ffi)
    mask |= (ffi > 3e5)

    # Estimate the background:
    sigma_clip = SigmaClip(sigma=3.0, iters=5) #Sigma clip the data
    bkg_estimator = SExtractorBackground()     #Call background estimator
    bkg = Background2D(ffi, (64, 64),          #Estimate background on sigma clipped data
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=mask,
            exclude_percentile=50)

    bkg_est_unfilt = bkg.background

    #Smoothing the background using a percentile filter
    bkg_est = circular_filter(bkg_est_unfilt, diam=15, percentile=50)

    return bkg_est


if __name__ == '__main__':
    plt.close('all')

    # Read in data
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[0]

    #ffi, bkg = load_files(ffi_type)
    ffi, bkg = get_sim()

    #Run background estimation
    est_bkg = fit_background(ffi)

    #Plot background difference
    '''Plotting: all'''
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')

    fdiff, adiff = plt.subplots()
    diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
    fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
    adiff.set_title('Estimated bkg - True bkg')

    fest, aest = plt.subplots()
    est = aest.imshow(np.log10(est_bkg), cmap='Blues_r', origin='lower')
    fest.colorbar(est, label=r'$log_{10}$(Flux)')
    aest.set_title('Background estimated using the RH_estimate method')

    cc, button = close_plots()
    button.on_clicked(close)

    resRH = est_bkg - bkg
    medRH = np.median(100*resRH/bkg)
    stdRH = np.std(100*resRH/bkg)
    print('RH offset: '+str(np.round(medRH,3))+r"% $\pm$ "+str(np.round(stdRH,3))+'%')

    plt.show('all')
    plt.close('all')
