import os
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

def nircam_phot(input_files, filter='f200w',output_dir='.', ref_img_path='.',
                cat_name='', param_file=None,sharp_cut=0.01,
                crowd_cut=0.5):
    """
        Parameters
        ---------
        input_files: list,
                    list of paths to JWST NIRCAM level 3 _cal.fits or _crf.fits files
        filter: str,
                name of the NIRCAM filter being processed
        output_dir: str,
                    path to output directory.
                    Recommended: /photometry/
        ref_img_path: str,
                  path to level 3 drizzled image (_i2d.fits) image.
                  It is recommended to be inside /photometry/
        cat_name: str,
                  Output photometry catalogs will have prefix filter + cat_name

        Return
        ------
        None
    """
    if len(input_files)<1:
        raise Exception("input_files cannot be EMPTY")

    subprocess.run([f"nircammask {ref_img_path}.fits"], shell=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if param_file is None or not os.path.exists(param_file) :
      print("Using Default params")
      edit_params = True
      param_file = param_dir_default + '/nircam_sw_dolphot.param'
    else:
      edit_params = False

    out_id = filter + cat_name

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
    for f in exps:
        if not os.path.exists(f"{f}/data.sky.fits"):
            out = subprocess.run([f"nircammask {f}/data.fits"]
                                    ,shell=True)

            out = subprocess.run([f"calcsky {f}/data 10 25 2 2.25 2.00"]
                                , shell=True, capture_output=True)
    if edit_params:
      # Preparing Parameter file DOLPHOT NIRCAM
      with open(param_file) as f:
                  dat = f.readlines()

      dat[0] = f'Nimg = {len(exps)}                #number of images (int)\n'
      dat[4] = f'img0_file = {ref_img_path}\n'
      dat[5] = ''

      for i,f in enumerate(exps):
          dat[5] += f'img{i+1}_file = {f}/data           #image {i+1}\n'

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
                                    (phot_table['type'] <= 2)]
    flag_keys = []
    for key in phot_table1.keys():
        if 'flag' in key:
            flag_keys.append(key)
    for i in flag_keys:
        phot_table1  = phot_table1[phot_table1[i]<=2]

    SNR_keys = []
    for key in phot_table1.keys():
        if 'SNR' in key:
            SNR_keys.append(key)
    for i in SNR_keys:
        phot_table1  = phot_table1[phot_table1[i]>=5]

    phot_table.write(f'{output_dir}/{out_id}_photometry.fits', overwrite=True)
    phot_table1.write(f'{output_dir}/{out_id}_photometry_filt.fits', overwrite=True)
    print('NIRCAM Stellar Photometry Completed!')

def nircam_phot_comp(param_file=None, m=[20], filter='f200w', region_name = '3',
                     output_dir='.', tab_path='.', ref_img_path=None,cat_name='', 
                     sharp_cut=0.01, crowd_cut=0.5,
                     ra_col='ra',dec_col='dec',ra=0,dec=0, shape='box',
                     width=24/3600,height=24/3600,ang=245, nx=10,ny=10):
    """
        Parameters
        ---------
        filter: str,
                name of the NIRCAM filter being processed
        output_dir: str,
                    path to output directory.
                    Recommended: /photometry/
        tab_path: str,
                  path to photometry table.
                  It is recommended to be inside /photometry/
        cat_name: str,
                  Output photometry catalogs will have prefix filter + cat_name

        Return
        ------
        None
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
                                    (phot_table['type'] <= 2)]
        flag_keys = []
        for key in phot_table1.keys():
            if 'flag' in key:
                flag_keys.append(key)
        for i in flag_keys:
            phot_table1  = phot_table1[phot_table1[i]<=2]

        SNR_keys = []
        for key in phot_table1.keys():
            if 'SNR' in key:
                SNR_keys.append(key)
        for i in SNR_keys:
            phot_table1  = phot_table1[phot_table1[i]>=5]

        phot_table1.write(f'{output_dir}/fake_out_{region_name}_{m}_{out_id}_filt.fits', overwrite=True)
        print('NIRCAM Completeness Completed!')
    else:
      print(f"{output_dir}/{out_id}_photometry.fits NOT FOUND!!")
   
