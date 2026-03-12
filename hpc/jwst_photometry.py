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

filters = ['F115W', 'F150W', 'F200W']

# Query observations
obs_table = Observations.query_criteria(
    proposal_id="1783",
    objectname="NGC4485",
    filters=filters,
    dataRights="PUBLIC"
)

# Get product list once
products = Observations.get_product_list(obs_table)

# Remove duplicates
products = unique(products, keys="productFilename")

# Keep only UNCAL files
products = products[products['productSubGroupDescription'] == 'UNCAL']

for filt in filters:
    
    data_dir = f"/zfs-home/202404072C/JWST/data/ngc4485/data/stage0/{filt}/"
    os.makedirs(data_dir, exist_ok=True)

    # Select only files for this filter
    filt_products = products[products['filters'] == filt]

    Observations.download_products(
        filt_products,
        download_dir=data_dir,
        flat=True
    )

if __name__ == "__main__":

        input_files = glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F115W/stage0/*')
        out_dir     = '/zfs-home/202404072C/JWST/data/ngc4485/data/F115W/'
        jwst_data   = jpipe(input_files, out_dir,crds_dir='/zfs-home/202404072C/JWST/CRDS/',
                    crds_context='jwst_1466.pmap', filter='F115W',n_cores=24,
                        corr_1byf=False, corr_snowball=True)
        jwst_data()

        input_files = glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F150W/stage0/*')
        out_dir     = '/zfs-home/202404072C/JWST/data/ngc4485/data/F150W/'
        jwst_data   = jpipe(input_files, out_dir,crds_dir='/zfs-home/202404072C/JWST/CRDS/',
                    crds_context='jwst_1466.pmap', filter='F150W',n_cores=24,
                        corr_1byf=False, corr_snowball=True)
        jwst_data()

        input_files = glob(f'/zfs-home/202404072C/JWST/data/ngc4485/data/F200W/stage0/*')
        out_dir     = '/zfs-home/202404072C/JWST/data/ngc4485/data/F200W/'
        jwst_data   = jpipe(input_files, out_dir,crds_dir='/zfs-home/202404072C/JWST/CRDS/',
                    crds_context='jwst_1466.pmap',filter='F200W', n_cores=24,
                        corr_1byf=False, corr_snowball=True)
        jwst_data()
