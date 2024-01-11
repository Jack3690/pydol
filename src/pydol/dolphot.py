import os
from .simulation import sim_stars, comp_fit, comp_base
from .catalog import filter_phot
import pandas as pd
import numpy as np

from pathlib import Path

scripts = Path(__file__).parent.joinpath('scripts')
params = Path(__file__).parent.joinpath('params]')

class Base():
  def __init__(self, dict_images={}, regions={}):
    self.regions = {}
    self.dict_images = {}

  def nircam_phot(self, out_dir, region='bubble', filt_n='F115W', d=50, comp=True, 
               skip_phot=False):
  
    ra, dec = self.regions[region]['ra'], self.regions[region]['dec']
  
    det_n = 'NRCB3'
    
    input_path = self.dict_images[det_n][filt_n]['images'][0]
    
    if not skip_phot:
        if not os.path.exists(f"../{out_dir}/{region}/"):
            os.mkdir(f"../{out_dir}/{region}")
  
        if not os.path.exists(f"../{out_dir}/{region}/{filt_n}/"):
            os.mkdir(f"../{out_dir}/{region}/{filt_n}")
  
        # Generating copy of data
        os.system(f"cp {input_path} ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Applying NIRCAMMASK
        os.system(f"nircammask ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Generating cutout
        os.system(f"python {scripts}/dolphot_convert_nircam.py --f ../{out_dir}/{region}/{filt_n}/data.fits --d 'NRCB3' --c True --ra {ra} --dec {dec} --radius {d}")
  
        # Removing copy
        os.remove(f"../{out_dir}/{region}/{filt_n}/data.fits")
        
        # Calculating Sky
        os.system(f"calcsky ../{out_dir}/{region}/{filt_n}/data_conv 10 25 2 2.25 2.00")
  
        # Editing DOLPHOT parameter file
        with open("{params}/nircam_dolphot.param") as f:
            dat = f.readlines()
  
        dat[4] = f'img1_file = ../{out_dir}/{region}/{filt_n}/data_conv            #image 1\n'
  
        with open("{params}/nircam_dolphot.param", 'w', encoding='utf-8') as f:
            f.writelines(dat)
  
        # Running DOLPHOT
        os.system(f"dolphot ../{out_dir}/{region}/{filt_n}/out -p{params}/nircam_dolphot.param")
  
        # Filtering Output photometric catalog
  
        filter_phot(out_dir, region, filt_n)
  
    if comp:
        n = 100
        counts = []
        mags = np.arange(20,30, 0.5)
        for mag in mags:
            # Completeness Analysis
            sim_stars(out_dir, region, filt_n, d, mag,n)
            # Calculating Sky
            os.system(f"calcsky ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv 10 25 2 2.25 2.00")
  
            # Editing DOLPHOT parameter file
            with open("{params}/nircam_dolphot.param") as f:
                dat = f.readlines()
  
            dat[4] = f'img1_file = ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv            #image 1\n'
  
            with open("{params}/nircam_dolphot.param", 'w', encoding='utf-8') as f:
                f.writelines(dat)
  
            # Running DOLPHOT
            os.system(f"dolphot ../PHOT_OUTPUT_m50/{region}/{filt_n}/out -p{params}/nircam_dolphot.param")
  
            # Filtering Output photometric catalog
  
            filter_phot('PHOT_OUTPUT_m50', region, filt_n)
  
            counts.append(comp_base(out_dir, region, filt_n, 0.06, mag))
        
        x = mags
        y = np.array(counts)/n
        df = pd.DataFrame(zip(x,y), columns=['mag','frac'])
        df.to_csv(f"../{out_dir}/{region}/{filt_n}/comp_{filt_n}.csv")
        comp_fit(out_dir, region, filt_n, x,y)
  
  def acs_phot(self, out_dir, region='bubble', filt_n='F435W', d=50, comp=True,
             skip_phot=False):
  
    ra, dec = self.regions[region]['ra'], self.regions[region]['dec']
  
    det_n = 'WFC'
    input_path = self.dict_images[det_n][filt_n]['images'][0]
    
    if not skip_phot:
        if not os.path.exists(f"../{out_dir}/{region}/"):
            os.mkdir(f"../{out_dir}/{region}")
  
        if not os.path.exists(f"../{out_dir}/{region}/{filt_n}/"):
            os.mkdir(f"../{out_dir}/{region}/{filt_n}")
  
        # Generating copy of data
        os.system(f"cp {input_path} ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Applying ACSMASK
        os.system(f"acsmask ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Generating cutout
        os.system(f"python {scripts}/dolphot_convert_acs.py --f ../{out_dir}/{region}/{filt_n}/data.fits --c True --ra {ra} --dec {dec} --radius {d}")
        
        # Removing copy
        os.remove(f"../{out_dir}/{region}/{filt_n}/data.fits")
        
        # Calculating Sky
        os.system(f"calcsky ../{out_dir}/{region}/{filt_n}/data_conv 15 35 4 2.25 2.00")
  
        # Editing DOLPHOT parameter file
        with open("{params}/acs_dolphot.param") as f:
            dat = f.readlines()
  
        dat[4] = f'img1_file = ../{out_dir}/{region}/{filt_n}/data_conv            #image 1\n'
  
        with open("{params}/acs_dolphot.param", 'w', encoding='utf-8') as f:
            f.writelines(dat)
  
        # Running DOLPHOT
        os.system(f"dolphot ../{out_dir}/{region}/{filt_n}/out -p{params}/acs_dolphot.param")
  
        # Filtering Output photometric catalog
  
        filter_phot(out_dir, region, filt_n)  
    
    if comp:
        n=100
        counts = []
        mags = np.arange(22,32, 0.5)
        for mag in mags:
            # Completeness Analysis
            sim_stars(out_dir, region, filt_n, d, mag,n)
  
            # Calculating Sky
            os.system(f"calcsky ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv 15 35 4 2.25 2.00")
  
            # Editing DOLPHOT parameter file
            with open("{params}/acs_dolphot.param") as f:
                dat = f.readlines()
  
            dat[4] = f'img1_file = ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv            #image 1\n'
  
            with open("{params}/acs_dolphot.param", 'w', encoding='utf-8') as f:
                f.writelines(dat)
  
            # Running DOLPHOT
            os.system(f"dolphot ../PHOT_OUTPUT_m50/{region}/{filt_n}/out -p{params}/acs_dolphot.param")
                # Filtering Output photometric catalog
  
            filter_phot('PHOT_OUTPUT_m50', region, filt_n)
            
            counts.append(comp_base(out_dir, region, filt_n, 0.10, mag))
  
        x = mags
        y = np.array(counts)/n
        df = pd.DataFrame(zip(x,y), columns=['mag','frac'])
        df.to_csv(f"../{out_dir}/{region}/{filt_n}/comp_{filt_n}.csv")
        comp_fit(out_dir, region, filt_n, x,y)
  
  def wfc3_phot(self, out_dir, region='bubble', filt_n='F275W', d=50, comp=True,
             skip_phot=False):
  
    ra, dec = self.regions[region]['ra'], self.regions[region]['dec']
    det_n = 'UVIS'
    input_path = self.dict_images[det_n][filt_n]['images'][0]
    
    if not skip_phot:
        if not os.path.exists(f"../{out_dir}/{region}/"):
            os.mkdir(f"../{out_dir}/{region}")
  
        if not os.path.exists(f"../{out_dir}/{region}/{filt_n}/"):
            os.mkdir(f"../{out_dir}/{region}/{filt_n}")
  
        # Generating copy of data
        os.system(f"cp {input_path} ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Applying WFC3MASK
        os.system(f"wfc3mask ../{out_dir}/{region}/{filt_n}/data.fits")
  
        # Generating cutout
        os.system(f"python {scripts}/dolphot_convert_wfc3.py --f ../{out_dir}/{region}/{filt_n}/data.fits --c True --ra {ra} --dec {dec} --radius {d}")
        
        # Removing copy
        os.remove(f"../{out_dir}/{region}/{filt_n}/data.fits")
        
        # Calculating Sky
        os.system(f"calcsky ../{out_dir}/{region}/{filt_n}/data_conv 15 35 4 2.25 2.00")
  
        # Editing DOLPHOT parameter file
        with open("{params}/wfc3_dolphot.param") as f:
            dat = f.readlines()
  
        dat[4] = f'img1_file = ../{out_dir}/{region}/{filt_n}/data_conv            #image 1\n'
  
        with open("{params}/wfc3_dolphot.param", 'w', encoding='utf-8') as f:
            f.writelines(dat)
  
        # Running DOLPHOT
        os.system(f"dolphot ../{out_dir}/{region}/{filt_n}/out -p{params}/wfc3_dolphot.param")
  
        # Filtering Output photometric catalog
  
        filter_phot(out_dir, region, filt_n)
    
    if comp:
        n = 100
        counts = []
        mags = np.arange(22, 32, 0.5)
        for mag in mags:
            # Completeness Analysis
            sim_stars(out_dir, region, filt_n, d, mag,n)
        
            # Calculating Sky
            os.system(f"calcsky ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv 15 35 4 2.25 2.00")
  
            # Editing DOLPHOT parameter file
            with open("{params}/wfc3_dolphot.param") as f:
                dat = f.readlines()
  
            dat[4] = f'img1_file = ../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv            #image 1\n'
  
            with open("{params}/wfc3_dolphot.param", 'w', encoding='utf-8') as f:
                f.writelines(dat)
  
            # Running DOLPHOT
            os.system(f"dolphot ../PHOT_OUTPUT_m50/{region}/{filt_n}/out -p{params}/wfc3_dolphot.param")
            # Filtering Output photometric catalog
  
            filter_phot('PHOT_OUTPUT_m50', region, filt_n)
            counts.append(comp_base(out_dir, region, filt_n, 0.12, mag))
            
        x = mags
        y = np.array(counts)/n
        
        df = pd.DataFrame(zip(x,y), columns=['mag','frac'])
        df.to_csv(f"../{out_dir}/{region}/{filt_n}/comp_{filt_n}.csv")
        
        comp_fit(out_dir, region, filt_n, x,y)
