import juliet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from astropy.io import ascii

dataset = juliet.load(priors='ttvs_priors.dat', lcfilename='ttvs_lcs.dat', rvfilename='data/rvs.dat',
                      out_folder='ttvs_1pl', verbose=True,
                      lc_n_supersamp=[20], lc_exptime_supersamp=[0.020434], lc_instrument_supersamp='TESS')

results = dataset.fit(n_live_points=1000, pu=0.5, use_dynesty=True, dynesty_nthreads=30)
