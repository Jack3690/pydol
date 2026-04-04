from glob import glob
import astropy.io.fits as fits
import json
import os, sys
import astroquery
from astroquery.mast import Observations
import pandas as pd

from astropy.table import Table,unique, vstack
from astropy.wcs import WCS
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
from astropy.coordinates import angular_separation
import astropy.units as u
from pydol.pipeline.jwst import jpipe
import multiprocessing as mp
from pydol.photometry import nircam

if __name__ == "__main__":
        def run_phot(det):
                cal_files = glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F115W/stage2/*nrc{det}*_cal.fits')
                cal_files += glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F150W/stage2/*nrc{det}*_cal.fits')
                cal_files += glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F200W/stage2/*nrc{det}*_cal.fits')

                nircam.nircam_phot(cal_files,
                filter='f115w_f150w_f200w',
                output_dir= f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{det}/',
                ref_img_path='/zfs-home/202404072C/JWST/data/ngc4485/photometry/F200W_i2d',
                param_file=None,
                cat_name= f'_{det}')

        with mp.Pool(24) as p: p.map(run_phot,['a1','a2','a3','a4','b1','b2','b3','b4'])
