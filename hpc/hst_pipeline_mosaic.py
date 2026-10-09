from glob import glob
import astropy.io.fits as fits
import shutil
import os
import astroquery
from astroquery.mast import Observations
import pandas as pd

from astropy.table import Table, unique, vstack
from astropy.wcs import WCS
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
from astropy.coordinates import angular_separation, SkyCoord
from astropy.visualization import simple_norm
import astropy.units as u
import multiprocessing as mp
from drizzlepac import astrodrizzle, tweakreg
from drizzlepac.processInput import getMdriztabPars
from stwcs import updatewcs
from acstools import calacs
from wfc3tools import calwf3
from pydol.photometry.scripts import cmdtools
from pydol.photometry.acs import acs_phot
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.nddata import Cutout2D


os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_PATH'] = '/zfs-home/202404072C/HST/CRDS/'
os.environ['iref'] = '/zfs-home/202404072C/HST/CRDS/references/hst/wfc3/'
os.environ['jref'] = '/zfs-home/202404072C/HST/CRDS/references/hst/acs/'
os.environ['PATH'] += ':/zfs-home/202404072C/softwares/pkgs/hstcal/bin'


obs_table = Observations.query_criteria(proposal_id=["10452"],objectname="m51",filters= ['F435W','F555W','F658N','F814W'])

t = [Observations.get_product_list(obs) for obs in obs_table]
files = unique(vstack(t), keys='productFilename')

mast_dir = 'mast:hst/product' # Download from MAST
for filt in ['F814W']:

        data_dir = f'/zfs-home/202404072C/HST/data/m51/data/stage3/{filt}'  # save downloaded data
        os.makedirs(data_dir, exist_ok=True)

        manifest = Observations.download_products(
                   files,
                   productSubGroupDescription='FLC',
                   filters=[filt],
                   download_dir = data_dir,
                   flat = True,
                   )

def get_vals_from_mdriztab(
    input_flcs,
    kw_list=[
        "driz_sep_bits",
        "combine_type",
        "driz_cr_snr",
        "driz_cr_scale",
        "final_bits",
    ],
):
    #Get only selected parameters from. the MDRIZTAB
    mdriz_dict = getMdriztabPars(input_flcs)

    requested_params = {}

    print("Outputting the following parameters:")
    for k in kw_list:
        requested_params[k] = mdriz_dict[k]
        print(k, mdriz_dict[k])

    return requested_params


pos='*'
flc_f814w = glob(f"../data/m51/data/stage3/F814W/hst_10452_{pos}*flc.fits")

tweakreg.TweakReg(
    flc_f814w,
    imagefindcfg={'threshold':10,'peakmax':70000,'conv_width':3.5, 'roundlo':-1,'roundhi':1},
    refimage = '/zfs-home/202404072C/HST/data/m51/data/stage3/F814W_mosaic_drc.fits',
    refimagefindcfg = {'threshold':10,'peakmax':70000,'conv_width':3.5, 'roundlo':-1,'roundhi':1},
    use_sharp_round =True,
    #refcat = 'gaia.cat',
    #refxcol = 1,
    #refycol = 2,
    #rfluxcol = 3,
    clean=True,
   # minobj=10,
    interactive=False,
    shiftfile=False,
    fitgeometry="general",
    updatehdr=False,
    expand_refcat=True,
    enforce_user_order=False,
    reusename=True,
    searchrad=0.5,
    separation =10,
)

selected_params = get_vals_from_mdriztab(flc_f814w)

astrodrizzle.AstroDrizzle(
    flc_f814w,
    output="../data/m51/data/stage3/F814W_mosaic",
    preserve=False,
    build=True,
    skymethod="match",
    driz_sep_bits=selected_params["driz_sep_bits"],
    driz_cr_corr=True,
    clean=True,
    final_bits=selected_params["final_bits"],
    configobj=None,
    num_cores=24,
    coeffs=False,
    final_kernel = 'gaussian'
)
