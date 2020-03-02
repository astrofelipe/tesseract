# tesseract
**tesseract** is a tool originally made for extract light curves from **TESS Full Frame Images** (FFIs). The philosophy is to make calls as simple as possible, this means that the basic usage is:

  python ticlc.py <TIC ID> <TESS Sector>
  python ticlc.py <RA> <DEC> <TESS Sector>

This will output an ascii file with time, flux (aperture photometry) and flux error columns. Additionaly there's an instrument column, so you can use this as input directly in juliet. A preview plot will be displayed, showing the postage stamp used, the light curve and the estimated background.

You can personalize some parameters using the current available flags:

  --size          Changes postage stamp height and width (Default: 21 pixels)
  --noplots       Doesn't display the preview plot
  --overwrite     Overwrites existing file (by default it doesn't)

  --circ          Uses circular apertures instead of default K2P2 method
  --manualap      Allows to input a file containing pixels to be in the aperture (To do: Interactive mode)
  --psf           Does PSF photometry (same routine as Eleanor). Experimental
  --pld           Pixel Level Decorrelation. I'm continuously changing this routine so probably won't work :)
  --mask-transit  Intended for pld (so it won't overfit/remove transits)

  --norm          Output will be divided by the median flux (conserves variability)
  --flatten       Output will be flattened (variability removal)

  --pixlcs        Shows an additional plot with light curves per pixel
  --pngstamp      Saves a high quality postage stamp. This can be a 'minimal' (image and aperture, no axes) or 'full' (axes, title, ticks)
  --gaia          Adds Gaia sources to postage stamps plots
  --maxgaiamag    Maximum Gaia RP magnitude to show in previous option (Default: 16)

  --folder        "Offline mode". Instead of making a request to the MAST, uses previously downloaded and stacked FFIs in hdf5 format (see FFI2h5.py)
  --cmap          Changes cmap used in plots (Default: 'YlGnBu')
  --cam           Overrides camera value. Can be useful for plots when the target is near an edge and queries tell that it doesn't belong to any Camera/CCD
  --ccd           Same for CCD
