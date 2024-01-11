from astropy.io import fits
import argparse
import glob

from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import os

det_list = ["ACS",'WFC']

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Make HST Image Compatible to DOLPHOT')
  parser.add_argument("--f", dest='filename', default=None, type = str, help='HST FITS Image')
  parser.add_argument("--c", dest='cut', default=False, type = bool , help='Cutout')
  parser.add_argument("--ra", dest='RA', default=24.1862604, type = float , help='RA in degrees')
  parser.add_argument("--dec", dest='Dec', default=15.7721020, type = float , help='Dec in degrees')
  parser.add_argument("--radius", dest='radius', default=50, type = float , help='Radius in arcsecs')
  
  options = parser.parse_args()

  filename = options.filename

  cutout = options.cut
  radius = options.radius
  name = filename.split('.fits')[0]

  ra = options.RA
  dec = options.Dec
  if os.path.exists(filename):
    print("Reading file " ,options.filename)

    hdul = fits.open(filename)
    data = hdul[1].data
    header = hdul[1].header
    wcs = WCS(header)

    if 'DOL_ACS' in hdul[1].header.keys():
      dol_tag = hdul[1].header['DOL_ACS']
      if dol_tag < 0:

        hdul[1].header['DOL_ACS'] = 1
        filename = name + '_conv.fits'

      if cutout:
        print(f"Generating Cutout for RA: {ra} and Dec: {dec}")
        pos = SkyCoord(ra=ra, dec=dec, unit='deg')
        cutout = Cutout2D(data,pos,size = radius*u.arcsec, wcs = wcs)

        header.update(cutout.wcs.to_header())
        hdu1 = fits.PrimaryHDU(hdul[0].data, header= hdul[0].header)
        hdu2 = fits.ImageHDU(cutout.data, header=header)
        hdul_new= fits.HDUList([hdu1,hdu2])
        hdul = hdul_new.copy()

      print(f"Saving {filename}")
      hdul.writeto(filename, output_verify='ignore', overwrite=True)

    else:
      print('Run acsmask!')

  
