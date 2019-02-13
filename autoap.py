from sklearn.cluster import DBSCAN
from skimage.morphology import watershed
from astropy.stats import mad_std
import scipy.ndimage as ndi
import numpy as np

#DBSCAN
rc = 1
nc = 4

def generate_aperture(fluxes, n=5):
    flsum = np.nansum(fluxes, axis=0)
    thr   = mad_std(flsum)
    thm   = flsum > n*thr

    pos = np.column_stack(np.where(thm))
    db  = DBSCAN(eps=rc, min_samples=nc).fit(pos)
    #ncl = np.unique(db.labels_).size - (1 if -1 in db.labels_ else 0)

    clpos = np.transpose(pos[db.labels_ != -1])
    cluster = np.zeros(flsum.shape).astype(bool)
    cluster[clpos[0], clpos[1]] = True

    gauss = ndi.gaussian_filter(flsum, 0.5)
    gauss[~cluster] = 0

    nbh      = np.ones((3,3))
    localmax = (ndi.filters.maximum_filter(gauss, footprint=nbh) == gauss) & cluster
    maxloc   = np.column_stack(np.where(localmax))
    markers  = ndi.label(localmax)[0]
    labels   = watershed(-flsum, markers, mask=cluster)
    '''
    nstars   = labels.max()

    starmask = np.zeros((nstars, flsum.shape[0], flsum.shape[1])).astype(bool)
    for i in range(nstars):
        starmask[i][labels == i+1] = True
    '''

    return labels

def select_aperture(labs, x, y):
    x = int(x)
    y = int(y)

    theidx   = labs[x,y]
    final_ap = labs == theidx

    return final_ap
