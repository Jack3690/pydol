import os
"""
Module for JWST NIRCAM photometry and completeness analysis using DOLPHOT.

Provides functions to run photometry, filter catalogs, and generate completeness tests.
"""
from glob import glob
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import angular_separation
from astropy import units as u
import numpy as np
import multiprocessing as mp
from pathlib import Path
import subprocess
import pandas as pd
from .scripts.catalog_filter import box


param_dir_default = str(Path(__file__).parent.joinpath('params'))
script_dir = str(Path(__file__).parent.joinpath('scripts'))

sw_params = {'RAper': '2',
             'Rchi' : '1.5',
             'RSky0': '15',
             'RSky1': '35',
             'RSky2': '3 10',
             'RPSF' : '15',
             'apsky': '20 35',
             'aprad': '10',
             'RPSF ': '15'
             }

lw_params = {'RAper': '3',
             'Rchi' : '2',
             'RSky0': '15',
             'RSky1': '35',
             'RSky2': '4 10',
             'RPSF' : '15',
             'apsky': '20 35',
             'aprad': '10',
             'RPSF' : '15'
             }


def nircam_phot(input_files, filter='f200w',output_dir='.', ref_img_path='.',
                cat_name='', param_file=None,sharp_cut=0.01,
                crowd_cut=0.5, SNR_min=5, type=2):
  """
  Run DOLPHOT photometry for JWST NIRCAM images and filter the resulting catalog.

  Parameters
  ----------
  input_files : list of str
    Paths to JWST NIRCAM level 3 _cal.fits or _crf.fits files.
  filter : str, optional
    Name of the NIRCAM filter being processed (default: 'f200w').
  output_dir : str, optional
    Path to output directory (default: '.').
  ref_img_path : str, optional
    Path to level 3 drizzled image (_i2d.fits) (default: '.').
  cat_name : str, optional
    Output photometry catalogs will have prefix filter + cat_name (default: '').
  param_file : str, optional
    Path to DOLPHOT parameter file. If None, uses default params.
  sharp_cut : float, optional
    Maximum allowed squared sharpness for filtering (default: 0.01).
  crowd_cut : float, optional
    Maximum allowed crowding value for filtering (default: 0.5).
  SNR_min : float, optional
    Minimum allowed SNR for filtering (default: 5).
  type : int, optional
    Maximum allowed type value for filtering (default: 2).

  Returns
  -------
  None
    Writes filtered and unfiltered photometry tables to FITS files in output_dir.
  """
  if len(input_files)<1:
      raise Exception("input_files cannot be EMPTY")
  elif np.all([if '.fits' in i or '.FITS' in i for i in i input_files]:
      print(f"No of input images: {len(input_files)")
  else:
      raise Exception("Input file list contains data format other than FITS!!!")
    
  subprocess.run([f"nircammask {ref_img_path}.fits"], shell=True)
  if not os.path.exists(output_dir):
      os.mkdir(output_dir)

  if param_file is None or not os.path.exists(param_file) :
    print("Using Default params")
    edit_params = True
    param_file = param_dir_default + '/nircam_dolphot.param'
  else:
    edit_params = False

  out_id = filter + cat_name

  if edit_params:
    # Generating directories
    exps = []
    for i,f in enumerate(input_files):
        out_dir = f.split('/')[-1].split('.')[0]

        if not os.path.exists(f'{output_dir}/{out_dir}'):
            os.mkdir(f'{output_dir}/{out_dir}')
        if not os.path.exists(f"{output_dir}/{out_dir}/data.fits"):
            subprocess.run([f"cp {f} {output_dir}/{out_dir}/data.fits"],
                                shell=True)

        exps.append(f'{output_dir}/{out_dir}')

    # Applying NIRCAM Mask
    print("Running NIRCAMMASK and CALCSKY...")
    filter_wavelengths = []
    for f in exps:
        if not os.path.exists(f"{f}/data.sky.fits"):
            hdul = fits.open(f"{f}/data.fits")
            filter_wavelength = float(hdul[0].header['FILTER'][1:-1])
            filter_wavelengths.append(filter_wavelength)
            hdul.close()

            out = subprocess.run([f"nircammask {f}/data.fits"]
                                    ,shell=True)
            if filter_wavelength <= 200:
              out = subprocess.run([f"calcsky {f}/data 10 25 2 2.25 2.00"]
                                , shell=True, capture_output=True)
            else:
              # For Long Wavelength Filters (TBD)
              out = subprocess.run([f"calcsky {f}/data 10 25 2 2.25 2.00"]
                              , shell=True, capture_output=True)
    # Preparing Parameter file DOLPHOT NIRCAM
    with open(param_file) as f:
                dat = f.readlines()

    dat[0] = f'Nimg = {len(exps)}                #number of images (int)\n'
    dat[4] = f'img0_file = {ref_img_path}\n'
    dat[5] = ''

    for i,f in enumerate(exps):
        dat[5] += f'img{i+1}_file = {f}/data           #image {i+1}\n'
        if filter_wavelengths[i] <= 200:
            for key in sw_params.keys():
                dat[5] += f'img{i+1}_{key} = {sw_params[key]} \n'
        else:
            for key in lw_params.keys():
                dat[5] += f'img{i+1}_{key} = {lw_params[key]} \n'

    with open(f"{output_dir}/nircam_dolphot_{out_id}.param", 'w', encoding='utf-8') as f:
        f.writelines(dat)
    param_file = f"{output_dir}/nircam_dolphot_{out_id}.param"
  if not os.path.exists(f"{output_dir}/{out_id}_photometry.fits"):
      # Running DOLPHOT NIRCAM
      p = subprocess.Popen(["dolphot", f"{output_dir}/out", f"-p{param_file}"]
                          , stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
      while (line := p.stdout.readline()) != "":
        print(line)
  # Generating Astropy FITS Table

      out = subprocess.run([f"python {script_dir}/to_table.py --o {out_id}_photometry --f {output_dir}/out --d NIRCAM"],
                      shell=True)

  phot_table = Table.read(f"{output_dir}/{out_id}_photometry.fits")

  # Assingning RA-Dec using reference image
  hdu = fits.open(f"{ref_img_path}.fits")[1]

  wcs = WCS(hdu.header)
  positions = np.transpose([phot_table['x'] - 0.5, phot_table['y']-0.5])

  coords = np.array(wcs.pixel_to_world_values(positions))

  phot_table['ra']  = coords[:,0]
  phot_table['dec'] = coords[:,1]

  # Filtering stellar photometry catalog using Warfield et.al (2023) (Default)
  phot_table1 = phot_table[ (phot_table['obj_sharpness']**2<= sharp_cut) &
                                  (phot_table['obj_crowd']<= crowd_cut) &
                                  (phot_table['type'] <= type)]
  flag_keys = []
  for key in phot_table1.keys():
      if 'flag' in key:
          flag_keys.append(key)
  for i in flag_keys:
      phot_table1  = phot_table1[phot_table1[i]<=type]

  SNR_keys = []
  for key in phot_table1.keys():
      if 'SNR' in key:
          SNR_keys.append(key)
  for i in SNR_keys:
      phot_table1  = phot_table1[phot_table1[i]>=SNR_min]

  phot_table.write(f'{output_dir}/{out_id}_photometry.fits', overwrite=True)
  phot_table1.write(f'{output_dir}/{out_id}_photometry_filt.fits', overwrite=True)
  print('NIRCAM Stellar Photometry Completed!')

def nircam_phot_comp(param_file=None, m=[20], filter='f200w', region_name = '3',
                     output_dir='.', tab_path='.', ref_img_path=None,cat_name='', 
                     sharp_cut=0.01, crowd_cut=0.5, SNR_min=5, type=2,
                     ra_col='ra',dec_col='dec',ra=0,dec=0, shape='box',
                     width=24/3600,height=24/3600,ang=245, nx=10,ny=10):
  """
  Generate completeness tests for JWST NIRCAM photometry using fake star injection.

  Parameters
  ----------
  param_file : str
    Path to DOLPHOT parameter file (must exist).
  m : list of float, optional
    Magnitudes for fake stars (default: [20]).
  filter : str, optional
    Name of the NIRCAM filter being processed (default: 'f200w').
  region_name : str, optional
    Name/ID for the region (default: '3').
  output_dir : str, optional
    Path to output directory (default: '.').
  tab_path : str, optional
    Path to photometry table (default: '.').
  ref_img_path : str, optional
    Path to reference image for WCS assignment (default: None).
  cat_name : str, optional
    Output photometry catalogs will have prefix filter + cat_name (default: '').
  sharp_cut : float, optional
    Maximum allowed squared sharpness for filtering (default: 0.01).
  crowd_cut : float, optional
    Maximum allowed crowding value for filtering (default: 0.5).
  SNR_min : float, optional
    Minimum allowed SNR for filtering (default: 5).
  type : int, optional
    Maximum allowed type value for filtering (default: 2).
  ra_col, dec_col : str, optional
    Column names for RA and Dec (default: 'ra', 'dec').
  ra, dec : float, optional
    Center coordinates for region selection (default: 0).
  shape : str, optional
    Region shape ('box' or 'circle', default: 'box').
  width, height : float, optional
    Region width/height in degrees (default: 24/3600).
  ang : float, optional
    Region angle in degrees (default: 245).
  nx, ny : int, optional
    Number of grid points in x/y (default: 10).

  Returns
  -------
  None
    Writes fake star catalogs and filtered completeness tables to output_dir.
  """
  if param_file is None or not os.path.exists(param_file) :
    raise Exception("param_file cannot be EMPTY")
    
  if not os.path.exists(output_dir):
      os.mkdir(output_dir)

  out_id = filter + cat_name

    # Completeness
  tab = Table.read(tab_path)
  if shape=='box':
    tab_n = box(tab,ra_col,dec_col, ra, dec, 0,0, width, height, angle=ang)
  elif shape=='circle':
    tab_n = tab.copy()
    tab_n['r'] = angular_separation(tab[ra_col]*u.deg, tab[dec_col]*u.deg,
                                    ra*u.deg, dec*u.deg).to(u.deg).value
    tab_n = tab_n[tab_n['r']<=width]
    
    x_cen = 0.5*(tab_n['x'].min() + tab['x'].max())
    y_cen = 0.5*(tab_n['y'].min() + tab['y'].max())
    r_pix_max = np.sqrt( (tab_n['x'] - x_cen)**2 + (tab_n['y'] - y_cen)**2).max()
    
  x = tab_n['x']
  y = tab_n['y']
  xx, yy = np.meshgrid(np.linspace(x.min() + 10, x.max() - 10, nx),
                        np.linspace(y.min() + 10, y.max() - 10, ny))
  
  # Flatten and convert to integer
  x, y = xx.ravel().astype(int), yy.ravel().astype(int)
  if shape=='circle':
    r_pix = np.sqrt((x-x_cen)**2 + (y-y_cen)**2)
    ind = r_pix<=r_pix_max
    x = x[ind]
    y = y[ind]
  
  # Create the 'ext' and 'chip' columns directly
  ext = np.ones_like(x)
  chip = np.ones_like(x)
  
  # Create an array for all 'mag' columns at once
  mags = np.array([np.full(x.shape, m_) for m_ in m]).T
  
  # Build the DataFrame in a single step
  columns = {
      'ext': ext,
      'chip': chip,
      'x': x,
      'y': y,
  }
  for i in range(len(m)):
      columns[f'mag_{i}'] = mags[:, i]

  m = '_'.join(map(str,m))
  df = pd.DataFrame(columns)
  df.to_csv(f'{output_dir}/fake_{region_name}_{m}_{out_id}.txt', 
            sep=' ', index=False, header=False)

  with open(param_file) as f:
    dats = f.readlines()

  for n,dat in enumerate(dats):
    if 'FakeStars' in dat:
      break

  dats[n] = f'FakeStars =   {output_dir}/fake_{region_name}_{m}_{out_id}.txt\n'
  dats[n+1] = f'FakeOut =    {output_dir}/fake_{region_name}_{m}_{out_id}.fake\n'
  param_file_new = param_file.replace('.param',f'_{region_name}.param')
  with open(param_file_new,'w', encoding='utf-8') as f:
    f.writelines(dats)
  if os.path.exists(f"{output_dir}/{out_id}_photometry.fits"):
      # Running DOLPHOT NIRCAM
      p = subprocess.Popen(["dolphot", f"{output_dir}/out", f"-p{param_file_new}"]
                          , stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
      while (line := p.stdout.readline()) != "":
        print(line)
  # Generating Astropy FITS Table

      cmd = f"python {script_dir}/to_table_fake.py --f {output_dir}/fake_{region_name}_{m}_{out_id}.fake"
      cmd += f" --c {output_dir}/out.columns"
      cmd += f" --o fake_out_{region_name}_{m}_{out_id}"
      out = subprocess.run([cmd], shell=True)

      phot_table = Table.read(f"{output_dir}/fake_out_{region_name}_{m}_{out_id}.fits")

      if ref_img_path is not None:
      # Assingning RA-Dec using reference image
        hdu = fits.open(f"{ref_img_path}.fits")[1]
    
        wcs = WCS(hdu.header)
        positions = np.transpose([phot_table['x'] - 0.5, phot_table['y']-0.5])
    
        coords = np.array(wcs.pixel_to_world_values(positions))
    
        phot_table['ra']  = coords[:,0]
        phot_table['dec'] = coords[:,1]

      # Filtering stellar photometry catalog using Warfield et.al (2023)
      phot_table1 = phot_table[ (phot_table['obj_sharpness']**2<= sharp_cut) &
                                  (phot_table['obj_crowd']<= crowd_cut) &
                                  (phot_table['type'] <= type)]
      flag_keys = []
      for key in phot_table1.keys():
          if 'flag' in key:
              flag_keys.append(key)
      for i in flag_keys:
          phot_table1  = phot_table1[phot_table1[i]<=type]

      SNR_keys = []
      for key in phot_table1.keys():
          if 'SNR' in key:
              SNR_keys.append(key)
      for i in SNR_keys:
          phot_table1  = phot_table1[phot_table1[i]>=SNR_min]
      phot_table.write(f"{output_dir}/fake_out_{region_name}_{m}_{out_id}.fits", overwrite=True)
      phot_table1.write(f'{output_dir}/fake_out_{region_name}_{m}_{out_id}_filt.fits', overwrite=True)
      print('NIRCAM Completeness Completed!')
  else:
    print(f"{output_dir}/{out_id}_photometry.fits NOT FOUND!!")
   
