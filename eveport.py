#from everest.mathutils import Interpolate, SavGol
import numpy as np
#import everest
import celerite
from sklearn.decomposition import PCA
from itertools import combinations_with_replacement as multichoose

#class rPLDYay(everest.rPLD):#, mode='rpld'):
#    def load_tpf(self):
#        super(rPLDYay, self).load_tpf()

        #self.transitmask = transitmask
        #if args.inject[0] is not None: self.get_norm()

    #if mode='npld':
    #    def setup(self, **kwargs):
    #        self.X1N = X1N

def PLD(flux, aper, sap_flux):
    #1st order
    X_pld = np.reshape(flux[:, aper], (len(flux), -1))
    X_pld = X_pld / np.sum(flux[:, aper], axis=-1)[:, None]

    #2nd order + PCA
    X2_pld  = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld  = U[:, :X_pld.shape[1]]

    #Design matrix + fit
    X_pld = np.concatenate((np.ones((len(flux), 1)), X_pld, X2_pld), axis=-1)
    XTX   = np.dot(X_pld.T, X_pld)
    w_pld = np.linalg.solve(XTX, np.dot(X_pld.T, sap_flux))
    pld_flux = np.dot(X_pld, w_pld)

    return pld_flux


def PLD2(time, flux, ferr, lc, ap, n=None, mask=None, gp_timescale=30):
    if n is None:
        n = min(20, ap.sum())

    xmin, xmax = min(np.where(ap)[0]), max(np.where(ap)[0])
    ymin, ymax = min(np.where(ap)[1]), max(np.where(ap)[1])

    flux_crop = flux[:, xmin:xmax+1, ymin:ymax+1]
    ferr_crop = ferr[:, xmin:xmax+1, ymin:ymax+1]
    ap_crop   = ap[xmin:xmax+1, ymin:ymax+1]

    flux_err = np.nansum(ferr_crop[:, ap_crop]**2, axis=1)**0.5

    if mask is None:
        mask = np.where(time)

    #flsa = SavGol(lc)
    #med  = np.nanmedian(lc)
    #MAD  = 1.4826 * np.nanmedian(np.abs(lc - med))
    #print(np.where(~(lc > med + 10.*MAD) | (lc < med - 10.*MAD))[0])

    M = lambda x: x[mask]

    apval = np.copy(ap_crop).astype(int)

    ap_flux = np.array([f*apval for f in flux_crop]).reshape(len(flux_crop), -1)
    rawflux = np.sum(ap_flux.reshape(len(ap_flux), -1), axis=1)

    f1  = ap_flux / rawflux.reshape(-1,1)
    pca = PCA(n_components=n)
    X1  = pca.fit_transform(f1)

    f2  = np.product(list(multichoose(f1.T, 2)), axis=1).T
    pca = PCA(n_components=n)
    X2  = pca.fit_transform(f2)

    #f3  = np.product(list(multichoose(f1.T, 3)), axis=1).T
    #pca = PCA(n_components=n)
    #X3  = pca.fit_transform(f3)

    X  = np.hstack([np.ones(X1.shape[0]).reshape(-1,1), X1, X2])
    MX = M(X)


    y   = M(rawflux) - np.dot(MX, np.linalg.solve(np.dot(MX.T, MX), np.dot(MX.T, M(rawflux))))
    amp = np.nanstd(y)
    tau = gp_timescale
    ker = celerite.terms.Matern32Term(np.log(amp), np.log(tau))
    gp  = celerite.GP(ker)

    sigma = gp.get_matrix(M(time)) + np.diag(np.sum(M(ferr_crop).reshape(len(M(ferr_crop)), -1), axis=1)**2)

    A = np.dot(MX.T, np.linalg.solve(sigma, MX))
    B = np.dot(MX.T, np.linalg.solve(sigma, M(rawflux)))
    C = np.linalg.solve(A, B)

    model    = np.dot(X, C)
    det_flux = rawflux - (model - np.nanmean(model))

    return det_flux, flux_err

'''
def GetData(hdu, TICID, aperture, season=None):
    campaign = hdu[0].header['SECTOR']
    TICID    = hdu[1].header['TICID']

    qdata    = hdu[1].data
    cadn     = qdata['CADENCENO']
    time     = qdata['TIME']
    fpix     = qdata['FLUX']
    fpix_err = qdata['FLUX_ERR']
    pc1      = qdata['POS_CORR1']
    pc2      = qdata['POS_CORR2']
    qual     = qdata['QUALITY']

    #NaNs
    nani = np.where(np.isnan(time))
    time = Interpolate(np.arange(0, len(time)), nani, time)

    if not np.all(np.isnan(pc1)) and not np.all(np.isnan(pc2)):
        pc1 = Interpolate(time, np.where(np.isnan(pc1)), pc1)
        pc2 = Interpolate(time, np.where(np.isnan(pc2)), pc2)
    else:
        pc1 = None
        pc2 = None

    ap      = np.where(aperture & 1)
    fpix2D  = np.array([ff[ap] for ff in fpix], dtype='float64')
    fpixe2D = np.array([fe[ap] for fe in fpix_err], dtype='float64')

    binds = np.where(aperture ^ 1)
    if len(binds[0]) > 0:
        bkg = np.nanmedian(np.array([f[binds] for f in fpix], dtype='float64'), axis=1)
        bkg_err = 1.253 * np.nanmedian(np.array([e[binds] for e in fpix_err], dtype='float64'), axis=1) / np.sqrt(len(binds[0]))

        bkg = bkg.reshape(-1,1)
        bkg_err = bkg_err.reshape(-1,1)

    fpix = fpix2D - bkg
    fpix_err = np.sqrt(fpixe2D**2 + bkg_err**2)
    flux = np.sum(fpix, axis=1)
    ferr = np.sqrt(np.sum(fpix_err**2, axis=1))

    nanmask  = np.where(np.isnan(flux) | (flux==0))[0]
    fpix     = Interpolate(time, nanmask, fpix)
    fpix_err = Interpolate(time, nanmask, fpix_err)

    #Return
    data = everest.utils.DataContainer()
    data.name     = 'ee'
    data.ID       = TICID
    data.campaign = campaign
    data.cadn     = cadn
    data.time     = time
    data.fpix     = fpix
    data.fpix_err = fpix_err
    data.nanmask  = nanmask
    data.aperture = 1*aperture
    data.aperture_name = 'manual'
    data.apertures = {'manual': aperture}
    data.quality  = qual
    data.Xpos     = pc1
    data.Ypos     = pc2
    data.nearby   = []
    data.hires    = None
    data.saturated = False
    data.bkg      = bkg

    return data


    return campaign, TICID
'''
