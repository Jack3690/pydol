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

from pydol.photometry.scripts import cmdtools
from pydol.photometry import nircam

with open('/zfs-home/202404072C/JWST/data/ngc4485/nircam_dets_ngc4485.reg') as f:
    dat = f.readlines()

regions_dict = {}
for n,i in enumerate(dat[3:-1]):
    ra = float(i.split(',')[0][4:])
    dec = float(i.split(',')[1])
    width = float(i.split(',')[2][:-1])
    height = float(i.split(',')[3][:-1])
    ang = float(i.split(',')[4].split(')')[0])
    regions_dict[i.split('{')[-1][:2]] = {'ra' : ra,
                                'dec': dec,
                                'width': width,
                                'height' : height,
                                 'angle' : ang}

def run_comp(p):
        ra = regions_dict[p]['ra']
        dec = regions_dict[p]['dec']
        for j in np.arange(16,26.1,0.5):
                nircam.nircam_phot_comp(m=[16,16,np.round(j,1)],
                                filter='f115w_f150w_f200w_',
                                region_name = 'fake',
                                output_dir=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}',
                                param_file=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/nircam_dolphot_f115w_f150w_f200w_{p}.param',
                                tab_path=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/f115w_f150w_f200w_{p}_photometry_filt.fits',
                                ref_img_path = f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/F200W_i2d',
                                cat_name=p, sharp_cut=0.01, crowd_cut=0.5,
                                ra=ra,dec=dec, width=regions_dict[p]['width']/3600,
                                height=regions_dict[p]['height']/3600,
                                ang=132.84632, nx=50, ny=30)


        for j in np.arange(16,26.1,0.5):
                nircam.nircam_phot_comp(m=[16,np.round(j,1),16],
                                filter='f115w_f150w_f200w_',
                                region_name = 'fake',
                                output_dir=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}',
                                param_file=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/nircam_dolphot_f115w_f150w_f200w_{p}.param',
                                tab_path=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/f115w_f150w_f200w_{p}_photometry_filt.fits',
                                ref_img_path = f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/F200W_i2d',
                                cat_name=p, sharp_cut=0.01, crowd_cut=0.5,
                                ra=ra,dec=dec, width=regions_dict[p]['width']/3600,
                                height=regions_dict[p]['height']/3600,
                                ang=132.84632, nx=50, ny=30)

        for j in np.arange(16,26.1,0.5):
                nircam.nircam_phot_comp(m=[np.round(j,1),16,16],
                                filter='f115w_f150w_f200w_',
                                region_name = 'fake',
                                output_dir=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}',
                                param_file=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/nircam_dolphot_f115w_f150w_f200w_{p}.param',
                                tab_path=f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/{p}/f115w_f150w_f200w_{p}_photometry_filt.fits',
                                ref_img_path = f'/zfs-home/202404072C/JWST/data/ngc4485/photometry/F200W_i2d',
                                cat_name=p, sharp_cut=0.01, crowd_cut=0.5,
                                ra=ra,dec=dec, width=regions_dict[p]['width']/3600,
                                height=regions_dict[p]['height']/3600,
                                ang=132.84632, nx=50, ny=30)

if __name__ == "__main__":
        with mp.Pool(8) as p:
                p.map(run_comp,['a1','a2','a3','a4','b1','b2','b3','b4'])
