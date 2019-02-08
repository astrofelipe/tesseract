#!/bin/env python
# -*- coding: utf-8 -*-

"""
A code containing some multi-use functions.

.. versionadded:: 1.3

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.mlab as mlab
from matplotlib.widgets import Button
import astropy.io.fits as pyfits
import scipy.ndimage as nd
from tqdm import tqdm
import os
import sys


'''
def get_gaussian(X, Y, A, (mux, muy), (sigma_x, sigma_y)):
    
    A simple function that returns a 2D gaussian.

    Parameters:
        X (ndarray): A numpy meshgrid
        Y (ndarray): A numpy meshgrid
        (mux, muy) (float64, float64): The position of the Gaussian in x and y
            respecitvely.
        (sigma_x, sigma_y) (float64, float64): The standard deviation of the Gaussian
            in x and y respectively.

    Returns:
        ndarray: A 2D Gaussian on a meshgrid in the shape of the input X and Y.
    
    return  A   * np.exp(-0.5 * (mux - X)**2 / sigma_x**2) \
                * np.exp(-0.5 * (muy- Y)**2 / sigma_y**2)
'''

def circular_filter(data, diam=15, percentile=10, filter_type='percentile'):
    '''
    A function that runs a filter of choice using a circular footprint of a
    diameter determined by the user.

    Parameters:
        data (ndarray): An array containing the unsmoothed data.

        diam (int): Default: 15. The desired diameter in pixels of the circular
            footprint.

        percentile (int): Default: 10. The desired percentile to use on the percentile
            filter. If percentile is set to 50, it effectively functions as a median
            filter.

        filter_type (str): Default 'percentile'. Call 'minimum' for a minimum filter,
            'percentile' for a percentile filter or 'maximum' for a maximum filter.

    Returns:
        ndarray: An array of the same shape as the input data containing the data
            smoothed using the filter of choice.
    '''

    if diam%2 == 0: diam+=1 #Make sure the diameter is uneven for symmetry
    core = int(diam/2)      #Finding the centre of the circle
    X, Y = np.meshgrid(np.arange(diam), np.arange(diam))    #Creating a meshgrid
    circle = (X - core)**2 + (Y-core)**2        #Building the circle in the meshgrid
    lim = circle[np.where(circle==0)[0]][:,0]   #Finding the value at the diameter edge
    circle[circle <= lim] = 1  #Setting all values inside of the circle to 1
    circle[circle > lim] = 0    #Setting all values outside of the circle to 0

    if filter_type == 'percentile':
        filt = nd.filters.percentile_filter(data, percentile=percentile,\
                footprint=circle)

    if filter_type == 'minimum':
        filt = nd.filters.minimum_filter(data, footprint=circle)

    if filter_type == 'maximum':
        filt = nd.filters.maximum_filter(data, footprint=circle)

    return filt

def get_sim(style='flat'):
    '''
    A function that creates a simple testing backround.

    Note: The 'complex' background has a random element to it. If you re-run the
    code generation __do__not__commit__the__new__file!

    Parameters:
        style (str): Default 'flat'. Either 'flat' for a flat gaussian noise background,
            'complex' for a background that includes slopes and crowding, and 'full'
            for a full FFI simulation.

    Returns:
        ndarray: Simulated FFI of given shape.

        ndarray: The background of the simulated FFI of given shape.
    '''
    shape = (2048,2048)
    X, Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))

    if style == 'complex':
        #If the background has already been generated on this local repository, read in
        if os.path.isfile('../Tests/complex_sim.fits'):
            sim = pyfits.open('../Tests/complex_sim.fits')[0].data
            bkg = pyfits.open('../Tests/complex_sim_bkg.fits')[0].data

        else:
            a = -1.05    #Slope on spiffy bkg
            b = -3.55    #Slope on spiffy bkg

            z = 90000    #Median height of spiffy background
            sigma = 2000 #Sigma is std on spiffy bkg (2% spread)

            nstars = 2000   #Number of contaminant stars
            orig_stars = np.zeros(X.shape)
            locx = np.random.rand(nstars) * (shape[1]-1)    #Getting a random list of star positions
            locy = np.random.rand(nstars) * (shape[0]-1)
            print('Building simulated background...')
            for s in tqdm(range(nstars)):           #Calculating the gaussian for each position
                height = np.random.exponential()*1
                w = 20
                orig_stars +=  get_gaussian(X, Y, 1, (locx[s], locy[s]),(w, w))

            #Normalising the bkg_stars component to be a 10% fraction
            bkg_stars = orig_stars/orig_stars.max()
            bkg_stars *= 0.05
            bkg_stars += 1

            #Defining the other components
            bkg_slope = a*X + b*Y + z
            bkg_gauss = get_gaussian(X, Y, 0.2, (650,1650), (600,400)) + 1

            #Combining the backgrounds and adding noise
            bkg = bkg_slope * bkg_gauss * bkg_stars
            sim = np.random.normal(bkg, sigma, shape)

            #Saving the sim and background to fits formats to save space
            hdusim = pyfits.PrimaryHDU(sim)
            hdubkg = pyfits.PrimaryHDU(bkg)
            hdusim.writeto('complex_sim.fits')
            hdubkg.writeto('complex_sim_bkg.fits')
        return sim, bkg

    if style == 'flat':
        z = 1000        #height of the background
        sigma = 10     #Error on the background

        sim = np.random.normal(z, sigma, shape)
        return sim, np.ones(shape)*z

def load_files(ffi_type):
    '''
    A function that reads in the FFI testing data from inside the git repo.

    Parameters:
        ffi_type (str): The name of the type of ffi. ffi_north, ffi_south, or
            ffi_cluster.

    Returns:
        ndarray: The read-in simulated FFI

        ndarray: The read-in simulated background for the FFI

    '''

    sfile = glob.glob('../data/FFI/'+ffi_type+'.fits')[0]
    bgfile = glob.glob('../data/FFI/backgrounds_'+ffi_type+'.fits')[0]

    try:
        hdulist = pyfits.open(sfile)
        bkglist = pyfits.open(bgfile)

    except IOError:
        print('File not located correctly.')
        sys.exit()

    ffi = hdulist[0].data
    bkg = bkglist[0].data

    return ffi, bkg

def close_plots():
    '''
    A function that plots a button to instantly close all subplots. Useful when
    plotting a large number of comparisons.

    Returns:
        matplotlib.figure.Figure: 1 by 1 plot containing a 'close all' button.

        matplotlib.widgets.Button: A button widget required for button function.

    Note: Must be called with the line:
        button.on_clicked(close)

    Note: The close function must also be imported. Best to use
        from Functions import *

    '''
    fig, ax = plt.subplots(figsize=(1,1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    closeax =plt.axes([0.1,0.1,0.8,0.8])

    button = Button(closeax, 'Close Plots', color='white', hovercolor='r')
    return fig, button

def close(event):
    ''' A simple plt.close('all') function for use with close_plots().'''
    plt.close('all')
