#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function for estimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0
.. versionchanged:: 1.1

.. codeauthor:: Mikkel Nørup Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.io.fits as pyfits
from tqdm import tqdm
import corner

from scipy import optimize
from scipy import stats
from scipy import interpolate
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from sklearn import linear_model
from pyqt_fit import kde

from Functions import *

class cPlaneModel:
	def __init__(self,order=2,weights=None):
		"""
		Class that performs a linear least squares fit to data using a polynoimal
		of a chosen order.

		Parameters:
			order (int): Default 2. The polynomial order of the model.

			weights (ndarray): Default None. The sum of the RANSAC binary inlier
				masks across all iterations to be evaluated.

		Attributes:
			coeff (ndarray): Least squares solution to the fit.

			r (ndarray): Sums of residuals of the fit.

			rank (int): Rank of the coefficient matrix.

			s (ndarray): Singular values of the coefficient matrix.

		.. codeauthor:: Mikkel Nørup Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
		"""

		self.order = order
		self.weights = weights
		self.coeff = None
		self.r = None
		self.rank = None
		self.s = None

	def Amet(self, X2, Y2):
		"""
		Returns coefficients for the chosen polynomial order.

		Parameters:
			X2 (ndarray): Meshgrid of the X positions of the data points

			Y2 (ndarray): Meshgrid of the Y positions of the data points

		Returns:
			ndarray: Coefficients of the polynomial of chosen order.
		"""

		#Create the model to the chosen order
		if self.order==0:
			A = np.array([X2*0+1,]).T
		if self.order==1:
			A = np.array([X2*0+1, X2, Y2]).T
		if self.order==2:
			A = np.array([X2*0+1, X2, X2**2,Y2, Y2**2, X2*Y2]).T
		if self.order==3:
			A = np.array([X2*0+1, X2, X2**2, X2**3, Y2, Y2**2, Y2**3, X2*Y2, X2**2*Y2, X2*Y2**2]).T
		return A

	def evaluate(self, X, Y, m):
		"""
		Returns the evaluated model of chosen order for a meshgrid of X, Y and
		polynomial coefficients m.

		Parameters:
			X (ndarray): Meshgrid of the X positions of the data points

			Y (ndarray): Meshgrid of the Y positions of the data points

			m (ndarray): Coefficients for the polynomial of chosen order

		Returns:
			ndarray: 2D model of the polynomial of chosen order for the given
				meshgrid and coefficients.
		"""
		#Construct the model using the fit coefficients
		A = cPlaneModel.Amet(self,X, Y)
		B = np.dot(A, m)
		return B

	def fit(self, data):
		"""
		Fits a polynomial of chosen order to the data using inlier masks as weights.

		Parameters:
			data (ndarray): A  3 column array containing X and Y pixel values
				and the values to be fit to (Z).

		Returns:
			classobj: An instance of the cPlaneModel class following the assignment
				of values to all of its empty attributes.
		"""
		X2 = data[:,0]
		Y2 = data[:,1]
		A = cPlaneModel.Amet(self, X2, Y2)	#Building the model
		B = data[:,2]	#Calling the modes

		if not self.weights is None:
			W = self.weights/np.sum(self.weights)	#Normalize the inlier mask
			W = np.diag(np.sqrt(W))		#Create diagonal stucture for dotting
			Aw = np.dot(W,A)			#Multiply the model with weights
			Bw = np.dot(B,W)			#Multiply the mdoes with the weights

		else:
			Aw = A
			Bw = B

		#Fit the weighted model to the weighted data using a least squares fit
		coeff, r, rank, s = np.linalg.lstsq(Aw, Bw)
		self.coeff = coeff
		self.r = r
		self.rank = rank
		self.s = s

		return self

def fRANSAC(F, neighborhood, iterations):
	"""
	Fit a linear regression to some data using RANSAC an 'iterations' number of
	times. Returns a list of the coefficients and a sum of the inlier masks for
	all the fits.

	Parameters:
		F (ndarray): A 2D array of the data to be fit to.

		neighborhood (ndarray): A F.size x 3 array containing X and Y pixel values
			and the values to be fit to (Z).

		iterations (int): The number of iterations of RANSAC to be run.

	Returns:
		ndarray: Sum of binary RANSAC inlier masks for all iterations with the
			same size as F.size.

		ndarray: RANSAC best fit coefficients for all iterations with the shape
			(iterations x 2).

	.. codeauthor:: Mikkel Nørup Lund <mikkelnl@phys.au.dk>
    .. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
	"""

	inlier_masks = np.zeros(neighborhood.shape[0])
	coeffs = np.zeros([iterations, 2])
	for k in range(iterations):
		#Preparing the data for RANSAC
		XY = neighborhood[:,:2]
		Z = neighborhood[:,2]

		#Setting the RANSAC threshold as 90% of the M.A.D
		mad = 1.4826 * np.nanmedian(np.abs(F - np.nanmedian(F)))
		thresh = 0.9*mad

		ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=thresh)
		ransac.fit(XY, Z)

		#These inlier masks effectively remove stars from the continuum
		inlier_mask = ransac.inlier_mask_
		coeffs[k] = ransac.estimator_.coef_
		inlier_masks += inlier_mask

	return inlier_masks, coeffs

def fit_background(ffi, size=128, itt_field=1, itt_ransac=500, order=1, plots_on=False):
	"""
	Estimate the background of a Full Frame Image (FFI) using a number of steps:
		-Split the FFI into sub-sections of width 'size'
		-Fit the continuum of each subsection using RANSAC
		-Find the mode of the RANSAC qualified inliers
		-Fit to the continuum of the composite image of modes using RANSAC
		-Fit a 2D polynomial to the modes using the inlier masks as weights

	Parameters:
	    ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

		size (int): Default 128. The width of each sub-section of the ffi. Must
			multiply up to the width of the full FFI.

		itt_field (int): Default 1. The number of RANSAC fits to make to each
			sub-section.

		itt_ransac (int): Default 500. The number of RANSAC fits to make to the
			calculated modes across the full FFI.

		order (int): Default: 1. The desired order of the polynomial to be fit
			to the estimated background points.

	    plots_on (bool): Default False. When True, it will plot an example of
	        the method fitting to the first line of data on the first iteration
	        of the fitting loop.

	Returns:
	    ndarray: Estimated background with the same size as the input image.

	.. codeauthor:: Mikkel Nørup Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
	"""

	#Cutting the ffi up into 'size' smaller blocks
	block_ffi = (ffi.reshape(ffi.shape[0]//size, size, -1, size)
	            .swapaxes(1,2)
	            .reshape(-1, size, size))

	#Creating the storage for the mode positions in each block
	modes = np.zeros([ffi.shape[0]/size, ffi.shape[1]/size])

	#Creating an array with all pixel locations in a box in the 0 and 1 positions
	X0, Y0 = np.meshgrid(np.arange(size), np.arange(size))
	neighborhood0 = np.zeros([len(X0.flatten()), 3])
	neighborhood0[:, 0] = X0.flatten()
	neighborhood0[:, 1] = Y0.flatten()

	#Fitting RANSAC to each box and calculating the mode of the inlier pixels
	i = 0
	for j in tqdm(range(modes.shape[0])):
		for jj in range(modes.shape[1]):
			F = block_ffi[i, ::]    #Calling the first block

			neighborhood0[:, 2] = F.flatten()

			#Running RANSAC on itt_field iterations and saving inlier masks
			inlier_masks, _ = fRANSAC(F, neighborhood0, itt_field)

			#Putting the inlier masks back into 2D shape
			inlier_masks_arr = inlier_masks.reshape((F.shape[0], F.shape[1]))
			#Evaluating the inlier masks
			FFF2 = F[(inlier_masks_arr>itt_field/2)].flatten() #Inliers of more than 50%
			FFF2 = FFF2[(FFF2<np.percentile(FFF2,90))]       #Inlying below the 25th percentile

			#Building a KDE on the background inlier data
			kernel = stats.gaussian_kde(FFF2,bw_method='scott')
			alpha = np.linspace(FFF2.min(),FFF2.max(),10000)

			#Calculate an optimal value for the mode from the KDE
			modes[j, jj] = alpha[np.argmax(kernel(alpha))]
			i += 1 #Adding to the index to call the next block

	X, Y = np.meshgrid(np.arange(modes.shape[1]), np.arange(modes.shape[0]))
	pixel_factor = (size-1)			#The pixel position of the first box
	X = (X+0.5) * pixel_factor		#X and Y are now the pixel centroids of each box
	Y = (Y+0.5) * pixel_factor

	#Fitting a polynomial to the background
	Xfull, Yfull = np.meshgrid(np.arange(ffi.shape[1]), np.arange(ffi.shape[0]))

	# Creating a len(ffi)/size by 3 array of box positions and the mode of the box
	neighborhood = np.zeros([len(modes.flatten()),3])
	neighborhood[:, 0] = X.flatten()
	neighborhood[:, 1] = Y.flatten()
	neighborhood[:, 2] = modes.flatten()

	#Running RANSAC on the detected modes to expel outliers
	inlier_masks, coeffs = fRANSAC(modes, neighborhood, itt_ransac)

	if plots_on:
		fig = corner.corner(coeffs, labels=['m','c'])
		plt.show()

	#Calling a Plane Model class
	Model = cPlaneModel(order=order, weights=inlier_masks)
	Fit = Model.fit(neighborhood)	#Fitting the data with the model
	fit_coeffs = Fit.coeff

	#Construct the models for the sizexsize grid and the full ffi grid
	M = Model.evaluate(X, Y, fit_coeffs)
	bkg_est = Model.evaluate(Xfull, Yfull, fit_coeffs)

	return bkg_est


if __name__ == '__main__':
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[1]

	# ffi, bkg = load_files(ffi_type)
	ffi, bkg = get_sim()

	'''Program starts here:'''
	size = 128   		#Number of blocks to cut the ffi into
	itt_field = 1   	#Number of iterations of ransac fitting to each box
	itt_ransac = 500	#Number of iterations of ransac fitting to modes
	order = 0			#Order of polynomial to be fit
	plots_on = True		#Will display plots if True

	est_bkg = fit_background(ffi,\
	 				size, itt_field, itt_ransac, order, plots_on)

	'''Plotting: all'''
	print('The plots are up!')
	fig, ax = plt.subplots()
	im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
	fig.colorbar(im,label=r'$log_{10}$(Flux)')
	ax.set_title(ffi_type)

	fdiff, adiff = plt.subplots()
	diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
	fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
	adiff.set_title('Estimated bkg - True bkg')

	fest, aest = plt.subplots()
	est = aest.imshow(np.log10(est_bkg), cmap='Blues_r', origin='lower')
	fest.colorbar(est, label=r'$log_{10}$(Flux)')
	aest.set_title('Background estimate')

	cc, button = close_plots()
	button.on_clicked(close)

	resML = est_bkg - bkg
	medML = np.median(100*resML/bkg)
	stdML = np.std(100*resML/bkg)
	print('ML offset: '+str(np.round(medML,3))+r"% $\pm$ "+str(np.round(stdML,3))+'%')

	plt.show('all')
	plt.close('all')
