#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function fo restimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0
.. versionchanged:: 1.2

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""
#TODO: Include a unity test

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import astropy.io.fits as pyfits
import sys
import glob
import corner
from tqdm import tqdm

import numpy as np
from scipy import interpolate
from scipy import stats

from MNL_estimate import cPlaneModel
from MNL_estimate import fRANSAC
from Functions import *


def fit_background(ffi, ribsize=8, nside=10, itt_ransac=500, order=1, plots_on=False):
	"""
	Estimate the background of a Full Frame Image (FFI).
	This method employs basic principles from two previous works:
	-	It uses the Kepler background estimation approach by measuring the
	background in evenly spaced squares around the image.
	-	It employs the same background estimation as Buzasi et al. 2015, by taking
	the median of the lowest 20% of the selected square.

	The background values across the FFI are then fit to with a 2D polynomial
	using the cPlaneModels class from the MNL_estimate code.


	Parameters:
		ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

		ribsize (int): Default: 8. A single integer value that determines the length
			of the sides of the boxes the backgrounds are measured in.

		nside (int): Default: 100. The number of points a side to evaluate the
			background for, not consider additional points for corners and edges.

		itt_ransac (int): Default 500. The number of RANSAC fits to make to the
			calculated modes across the full FFI.

		order (int): Default: 1. The desired order of the polynomial to be fit
			to the estimated background points.

		plots_on (bool): Default False. A boolean parameter. When True, it will show
			a plot indicating the location of the background squares across the image.

	Returns:
		ndarray: Estimated background with the same size as the input image.

	.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
	"""

	#Setting up the values required for the measurement locations
	if ffi.shape[0] < 2048:	#If FFI file is a superstamp, reduce ribsize
		ribsize = 4

	#The width of the image
	xlen = ffi.shape[1]
	ylen = ffi.shape[0]

	#The percentage of pixels either side to be sampled in higher density
	perc = 0.1

	#The pixel step difference between sampling points
	lx = xlen/(nside+2)
	ly = ylen/(nside+2)

	#The pixel step difference between sampling points in areas of higher density
	superx = lx/2
	supery = ly/2

	#The number of higher sampling points in each area of higher density
	nsuper = perc*nside*2
	#The number of regular sampling points in the image
	nreg = (1-2*perc)*nside

	#The ending point in x and y of the higher density ares
	xend = perc*xlen
	yend = perc*xlen

	#Define sampling locations in areas of higher density at low x and y
	xlocs_left = np.linspace(0., xend-superx, int(nsuper))
	ylocs_left = np.linspace(0., yend-supery, int(nsuper))

	#Define sampling locations in areas of higher density at high x and y
	xlocs_right = np.linspace(xlen-xend+superx, xlen, int(nsuper))
	ylocs_right = np.linspace(ylen-yend+supery, ylen, int(nsuper))

	#Define sampling locations in the rest of the image
	xlocs_mid = np.linspace(xend,xlen-xend,int(nreg))
	ylocs_mid = np.linspace(yend,ylen-yend,int(nreg))

	#Combine all three location arrays into a single array, create a corersponding meshgrid
	xx = np.append(xlocs_left, np.append(xlocs_mid, xlocs_right))
	yy = np.append(ylocs_left, np.append(ylocs_mid,ylocs_right))
	X, Y = np.meshgrid(xx, yy)

	#Setting up a mask with points considered for background estimation
	mask = np.zeros_like(ffi)
	bkg_field = np.zeros_like(X.ravel())
	hr = int(ribsize/2)

	#Calculating the KDE and consequent mode inside masked ares
	for idx, (xx, yy) in tqdm(enumerate(list(zip(X.ravel(), Y.ravel())))):
		y = int(yy)
		x = int(xx)
		#Checking for edges and change treatment accordingly
		xleft = x-hr
		xright = x+hr+1
		yleft = y-hr
		yright = y+hr+1
		if x == 0:
			xleft = x
		if x == xlen:
			xright = xlen
		if y == 0:
			yleft = y
		if y == ylen:
			yright = ylen

		ffi_eval = ffi[yleft:yright, xleft:xright] #Adding the +1 to make the selection even

		#Building a KDE on the data
		kernel = stats.gaussian_kde(ffi_eval.flatten(),bw_method='scott')
		alpha = np.linspace(ffi_eval.min(), ffi_eval.max(), 10000)

		#Calculate the optimal value of the mode from the KDE
		bkg_field[idx] = alpha[np.argmax(kernel(alpha))]
		mask[yleft:yright, xleft:xright] = 1			#Saving the evaluated location in a mask

	#Plotting the ffi with measurement locations shown
	if plots_on:
		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
		fig.colorbar(im,label=r'$log_{10}$(Flux)')
		ax.contour(mask, c='r', N=1)
		plt.show()

	#Interpolating to draw the background
	Xf, Yf = np.meshgrid(np.arange(xlen), np.arange(ylen))

	points = np.array([X.ravel(),Y.ravel()]).T
	bkg_est = interpolate.griddata(points, bkg_field, (Xf, Yf), method='cubic')

	return bkg_est


if __name__ == '__main__':
	plt.close('all')

	#Define parameters
	plots_on = True
	nside = 25
	npts = nside**2
	ribsize = 8
	itt_ransac = 500
	order = 1

	# Load file:
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[0]

	# ffi, bkg = load_files(ffi_type)
	ffi, bkg = get_sim(style='complex')

	#Get background
	est_bkg = fit_background(ffi, ribsize, nside, itt_ransac, order, plots_on)

	'''Plotting: all'''
	print('The plots are up!')
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
	aest.set_title('Background estimated with '+str(npts)+' squares of '+str(ribsize)+'x'+str(ribsize))

	cc, button = close_plots()
	button.on_clicked(close)

	resOJH = est_bkg - bkg
	medOH = np.median(100*resOJH/bkg)
	stdOH = np.std(100*resOJH/bkg)
	print('OJH offset: '+str(np.round(medOH,3))+r"% $\pm$ "+str(np.round(stdOH,3))+'%')

	plt.show('all')
	plt.close('all')
