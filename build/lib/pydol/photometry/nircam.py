import os
from glob import glob
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np
import multiprocessing as mp
from pathlib import Path
import subprocess
import pandas as pd
from .scripts.catalog_filter import box

param_dir_default = str(Path(__file__).parent.joinpath('params'))
script_dir = str(Path(__file__).parent.joinpath('scripts'))

def nircam_phot(crf_files, filter='f200w',output_dir='.', drz_path='.',
                cat_name='', param_file=None,sharp_cut=0.01,
                crowd_cut=0.5):
    """
        Parameters
        ---------
        crf_files: list,
                    list of paths to JWST NIRCAM level 3 _crf.fits files
        filter: str,
                name of the NIRCAM filter being processed
        output_dir: str,
                    path to output directory.
                    Recommended: /photometry/
        drz_path: str,
                  path to level 3 drizzled image (_i2d.fits) image.
                  It is recommended to be inside /photometry/
        cat_name: str,
                  Output photometry catalogs will have prefix filter + cat_name

        Return
        ------
        None
    """
    if len(crf_files)<1:
        raise Exception("crf_files cannot be EMPTY")

    subprocess.run([f"nircammask {drz_path}.fits"], shell=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if param_file is None or not os.path.exists(param_file) :
      print("Using Default params")
      edit_params = True
      param_file = param_dir_default + '/nircam_dolphot.param'
    else:
      edit_params = False

    out_id = filter + cat_name

    # Generating directories
    exps = []
    for i,f in enumerate(crf_files):
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
      dat[4] = f'img0_file = {drz_path}\n'
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
    hdu = fits.open(f"{drz_path}.fits")[1]

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

def nircam_phot_comp(crf_files,m=20, filter='f200w',output_dir='.', drz_path='.',
                cat_name='', param_file=None,sharp_cut=0.01, crowd_cut=0.5,
                ra=0,dec=0,width=24/3600,height=24/3600,ang=245):
    """
        Parameters
        ---------
        crf_files: list,
                    list of paths to JWST NIRCAM level 2 _crf.fits files
        filter: str,
                name of the NIRCAM filter being processed
        output_dir: str,
                    path to output directory.
                    Recommended: /photometry/
        drz_path: str,
                  path to level 3 drizzled image (_i2d.fits) image.
                  It is recommended to be inside /photometry/
        cat_name: str,
                  Output photometry catalogs will have prefix filter + cat_name

        Return
        ------
        None
    """
    if len(crf_files)<1:
        raise Exception("crf_files cannot be EMPTY")

    subprocess.run([f"nircammask {drz_path}.fits"], shell=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if param_file is None or not os.path.exists(param_file) :
      raise Exception("param_file cannot be EMPTY")

    out_id = filter + cat_name

     # Completeness
    tab = Table.read(drz_path)
    tab_n = box(tab,ra, dec, width, height, angle=ang)
    x = tab_n['x']
    y = tab_n['y']
    xx, yy = np.meshgrid(np.linspace(x.min()+10,x.max()-10,100),
                         np.linspace(y.min()+10,y.max()-10,100))
    x,y = xx.ravel().astype(int), yy.ravel().astype(int)
    mag = m + x*0
    ext = 1 + x*0
    chip = 1 + x*0

    df = pd.DataFrame(zip(ext,chip,x,y,mag),
                      columns=['ext', 'chip', 'x', 'y', 'mag'])
    df.to_csv(f'{output_dir}_fake_{m}_{out_id}.txt',sep=' ',
              index=None,header=None)

    with open([param_file]) as f:
      dats = f.readlines()

    for n,dat in enumerate(dats):
      if 'FakeStars' in dat:
        ind = n

    dats[n] = f'FakeStars =   {output_dir}_fake_{m}_{out_id}.txt\n'
    dat[n+1] = f'FakeOut =    {output_dir}_fake_{m}_{out_id}.fake\n'
    if os.path.exists(f"{output_dir}/{out_id}_photometry.fits"):
        # Running DOLPHOT NIRCAM
        p = subprocess.Popen(["dolphot", f"{output_dir}/out", f"-p{param_file}"]
                            , stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        while (line := p.stdout.readline()) != "":
          print(line)
    # Generating Astropy FITS Table

        cmd = f"python {script_dir}/to_table_fake.py --o fake_out_{m}_{out_id}"
        cmd += " --f {output_dir}/o{output_dir}_fake_{m}_{out_id}.fakeut"
        out = subprocess.run([cmd], shell=True)

        phot_table = Table.read(f"{output_dir}/fake_out_{m}_{out_id}.fits")

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

        phot_table1.write(f'{output_dir}/{out_id}_photometry_filt.fits', overwrite=True)
    print('NIRCAM Completeness Completed!')
