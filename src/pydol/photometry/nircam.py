import os
from glob import glob
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np
import multiprocessing as mp
from pathlib import Path
import subprocess

#sys_func = os.system
sys_func = os.subprocess.run

param_dir = str(Path(__file__).parent.joinpath('params'))
script_dir =str(Path(__file__).parent.joinpath('scripts'))

def nircam_phot(cal_files, name='f200w',output_dir='.', drz_path='.', ):
    sys_func(f"nircammask {drz_path}.fits") 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generating directories
    exps = []
    for i,f in enumerate(cal_files):
        out_dir = f.split('/')[-1].split('.')[0]

        if not os.path.exists(f'{output_dir}/{out_dir}'):
            os.mkdir(f'{output_dir}/{out_dir}')
        if not os.path.exists(f"{output_dir}/{out_dir}/data.fits"):
            sys_func(f"cp {f} {output_dir}/{out_dir}/data.fits")

        exps.append(f'{output_dir}/{out_dir}')

    # Applying NIRCAM Mask
    for f in exps:
        if not os.path.exists(f"{f}/data.sky.fits"):
            sys_func(f"nircammask {f}/data.fits")
            sys_func(f"calcsky {f}/data 10 25 2 2.25 2.00")

    # Preparing Parameter file DOLPHOT NIRCAM
    with open(f"{param_dir}/nircam_dolphot.param") as f:
                dat = f.readlines()

    dat[0] = f'Nimg = {len(exps)}                #number of images (int)\n'
    dat[4] = f'img0_file = {drz_path}\n'

    for i,f in enumerate(exps):
        dat[5+i] = f'img{i+1}_file = {f}/data           #image {i+1}\n'

    out_id = np.random.random()
    with open(f"{param_dir}/nircam_dolphot_{out_id}.param", 'w', encoding='utf-8') as f:
        f.writelines(dat)
        
    if not os.path.exists(f"{output_dir}/{name}_photometry.fits"):
        # Running DOLPHOT NIRCAM
        sys_func(f"dolphot {output_dir}/out -p{param_dir}/nircam_dolphot_{out_id}.param")

    # Generating Astropy FITS Table
   
    sys_func(f"python {script_dir}to_table.py --o {name}_photometry --n {len(exps)} --f {output_dir}/out")

    phot_table = Table.read(f"{output_dir}/{name}_photometry.fits")
    phot_table.rename_columns(['mag_vega'],[f'mag_vega_F200W'])

    # Assingning RA-Dec using reference image
    hdu = fits.open(f"{drz_path}.fits")[1]

    wcs = WCS(hdu.header)
    positions = np.transpose([phot_table['x'] - 0.5, phot_table['y']-0.5])

    coords = np.array(wcs.pixel_to_world_values(positions))

    phot_table['ra']  = coords[:,0]
    phot_table['dec'] = coords[:,1]

    # Filtering stellar photometry catalog using Warfield et.al (2023)
    phot_table1 = phot_table[ (phot_table['sharpness']**2   <= 0.01) &
                                (phot_table['obj_crowd']    <=  0.5) &
                                (phot_table['flags']        <=    2) &
                                (phot_table['type']         <=    2)]

    phot_table2 = phot_table[ ~((phot_table['sharpness']**2 <= 0.01) &
                                (phot_table['obj_crowd']    <=  0.5) &
                                (phot_table['flags']        <=    2) &
                                (phot_table['type']         <=    2))]
    print('NIRCAM SHORT')
    phot_table.write(f'{output_dir}/{name}_photometry.fits', overwrite=True)
    phot_table1.write(f'{output_dir}/{name}_photometry_filt.fits', overwrite=True)
    phot_table2.write(f'{output_dir}/{name}_photometry_rej.fits', overwrite=True)
    


