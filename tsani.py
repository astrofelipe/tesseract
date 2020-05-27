import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (AsymmetricPercentileInterval, LogStretch, ImageNormalize)


#Based on Matplotlib example
class UpdateDist:
    def __init__(self, fig, ax, data, lc):
        self.data  = data
        self.stamp = ax.imshow(np.zeros(self.data[0].shape), origin='lower')
        self.count = ax.text(0.9, 0.9, '', transform=ax.transAxes, ha='right', va='top')
        self.ax    = ax
        self.fig   = fig

        #Params?
        #ax.set_xlim...

        #Lightcurve
        #ax.plot...

    def init(self):
        self.stamp.set_data(self.data[0])
        #self.count.set_text('0')
        return (self.stamp, self.count)

    def __call__(self, i):
        if i==0:
            return self.init()

        self.stamp.set_data(self.data[i])
        #self.count.set_text(i)
        return (self.stamp, self.count)

class Outreach(UpdateDist):
    def __init__(self, fig, ax, data, lc):
        nx, ny    = data[0].shape
        self.data = data[:,:7*nx//8,:]
        self.lc   = lc
        #nx, ny = data[0].shape

        UpdateDist.__init__(self, fig, ax, self.data, lc)


        norm = ImageNormalize(np.nanmedian(data, axis=0), interval=AsymmetricPercentileInterval(70,100))
        #lstd = np.nanstd(lkf.flux)

        self.stamp.set_norm(norm)
        self.stamp.set_interpolation('bicubic')
        #self.stamp.set_extent((0, ny, nx, nx//3))
        #self.ax.set_aspect('equal')
        self.stamp.set_cmap('bone')
        #self.stamp.set_clim(vmin=np.nanmin(self.data), vmax=np.nanmax(self.data))

        self.ax.grid(False)
        self.ax.axis('off')

        self.lcax = plt.axes([.25, .15, .5, .25], frameon=False)
        self.lcax.set_xticks([])
        self.lcax.set_yticks([])

        self.lcax.plot(lc.time, lc.flux, '-w', lw=.5, zorder=10)
        for n in np.arange(1,6,0.75):
            self.lcax.plot(lc.time, lc.flux, '-', c='#CCCCCC', lw=.5+(n/2)**2, zorder=10, alpha=.06)
            print(.5+(n/2)**2)

        self.point = [self.lcax.plot(lc.time[0], lc.flux[0], 'o', color='#AAEEFF' if n!=0 else 'w', ms=3+(n/3)**2, zorder=11 if n!=0 else 12, alpha=0.05 if n!=0 else 1, mew=0) for n in range(10)]

        self.xmin = self.lc.time[0] - 3
        self.xmax = self.lc.time[0] + 3
        self.lcax.set_xlim(self.xmin, self.xmax)

        self.fig.tight_layout()


    def __call__(self, i):
        UpdateDist.__call__(self, i)
        for p in self.point:
            p[0].set_data(self.lc.time[i], self.lc.flux[i])

        self.xmin = self.lc.time[i] - 3
        self.xmax = self.lc.time[i] + 3
        self.lcax.set_xlim(self.xmin, self.xmax)

        return (self.stamp, self.count, self.lcax)
