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
import astropy.units as u
from astropy.coordinates import angular_separation
from .scripts.catalog_filter import box

param_dir_default = str(Path(__file__).parent.joinpath('params'))
script_dir = str(Path(__file__).parent.joinpath('scripts'))

def acs_phot(
    flt_files,
    filter='f435w',
    output_dir='.',
    drz_path='.',
    cat_name='',
    param_file=None,
    sharp_cut=0.2,
    crowd_cut=2.25,
    SNR_min=5,
):
    """
    HST ACS photometry using DOLPHOT (HPC-friendly version)

    Parameters
    ----------
    flt_files : list
        List of ACS _flt.fits files
    filter : str
    output_dir : str
    drz_path : str
        Path (without .fits extension) to drz image
    cat_name : str
    param_file : str or None
    sharp_cut : float
    crowd_cut : float
    """

    if len(flt_files) < 1:
        raise ValueError("flt_files cannot be EMPTY")


    os.makedirs(output_dir, exist_ok=True)

    with fits.open(f"{drz_path}.fits") as hdul:
        header = hdul[0].header
        if 'DOL_ACS' not in header:
            subprocess.run(["acsmask", f"{drz_path}.fits"], check=True)

    if param_file is None or not os.path.exists(param_file):
        print("Using default DOLPHOT params")
        edit_params = True
        param_file = os.path.join(param_dir_default, "acs_dolphot.param")
    else:
        edit_params = False

    out_id = filter + cat_name

    # -------------------------------------------------------
    # Prepare exposures
    # -------------------------------------------------------
    if edit_params:

        exps = []

        for f in flt_files:
            out_dir = os.path.basename(f).split('.')[0]
            exp_dir = os.path.join(output_dir, out_dir)
            os.makedirs(exp_dir, exist_ok=True)

            data_fits = os.path.join(exp_dir, "data.fits")
            if not os.path.exists(data_fits):
                subprocess.run(["cp", f, data_fits], check=True)

            exps.append(exp_dir)

        print("Running ACSMASK, SPLITGROUPS, CALCSKY...")

        for exp_dir in exps:

            chip1 = os.path.join(exp_dir, "data.chip1.sky.fits")
            chip2 = os.path.join(exp_dir, "data.chip2.sky.fits")

            if not (os.path.exists(chip1) and os.path.exists(chip2)):

                subprocess.run(["acsmask", f"{exp_dir}/data.fits"], check=True)
                subprocess.run(["splitgroups", f"{exp_dir}/data.fits"], check=True)

                subprocess.run(
                    ["calcsky", f"{exp_dir}/data.chip1", "15", "35", "4", "2.25", "2.00"],
                    check=True
                )

                subprocess.run(
                    ["calcsky", f"{exp_dir}/data.chip2", "15", "35", "4", "2.25", "2.00"],
                    check=True
                )

        # -------------------------------------------------------
        # Rewrite param file
        # -------------------------------------------------------
        with open(param_file) as f:
            dat = f.readlines()

        dat[0] = f"Nimg = {2 * len(exps)}\n"
        dat[4] = f"img0_file = {drz_path}\n"
        dat[5] = ""

        for i, exp_dir in enumerate(exps):
            dat[5] += f"img{2*i+1}_file = {exp_dir}/data.chip1\n"
            dat[5] += f"img{2*i+2}_file = {exp_dir}/data.chip2\n"

        param_file = os.path.join(output_dir, f"acs_dolphot_{out_id}.param")
        with open(param_file, "w") as f:
            f.writelines(dat)

    # -------------------------------------------------------
    # Run DOLPHOT
    # -------------------------------------------------------
    out_fits = os.path.join(output_dir, f"{out_id}_photometry.fits")

    if not os.path.exists(out_fits):

        print("Running DOLPHOT...")

        process = subprocess.Popen(
            ["dolphot", f"{output_dir}/out", f"-p{param_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError("DOLPHOT failed.")

    else:
        print(f"{out_fits} already exists.")

    # -------------------------------------------------------
    # Convert to FITS table
    # -------------------------------------------------------
    subprocess.run(
        ["python", f"{script_dir}/to_table.py",
         "--o", f"{out_id}_photometry",
         "--f", f"{output_dir}/out",
         "--d", "ACS"],
        check=True
    )

    phot_table = Table.read(out_fits)

    # -------------------------------------------------------
    # Add RA/Dec
    # -------------------------------------------------------
    with fits.open(drz_path) as hdu:
        wcs = WCS(hdu[1].header)

    ra, dec = wcs.pixel_to_world_values(
        phot_table["x"] - 0.5,
        phot_table["y"] - 0.5
    )

    phot_table["ra"] = ra
    phot_table["dec"] = dec

    # -------------------------------------------------------
    # Quality filtering
    # -------------------------------------------------------
    mask = np.ones(len(phot_table), dtype=bool)

    for filt in filter.split("_"):
        filt = filt.upper()
        mask &= (
            (phot_table[f"sharpness_{filt}"]**2 <= sharp_cut) &
            (phot_table[f"crowd_{filt}"] <= crowd_cut) &
            (phot_table[f"SNR_{filt}"] >= SNR_min)
        )

    mask &= phot_table["type"] <= type

    phot_table_filt = phot_table[mask]

    phot_table.write(out_fits, overwrite=True)
    phot_table_filt.write(
        os.path.join(output_dir, f"{out_id}_photometry_filt.fits"),
        overwrite=True
    )

    print("ACS Stellar Photometry Completed!")

def acs_phot_comp(
    param_file=None,
    m=[20],
    filter='f814w',
    region_name='3',
    output_dir='.',
    tab_path='.',
    ref_img_path=None,
    cat_name='',
    sharp_cut=0.2,
    crowd_cut=2.25,
    SNR_min=5,
    type=2,
    ra_col='ra',
    dec_col='dec',
    ra=0,
    dec=0,
    shape='box',
    width=24/3600,
    height=24/3600,
    ang=245,
    nx=10,
    ny=10,
):
    """
    ACS completeness analysis using DOLPHOT (HPC-optimized version)
    """

    if param_file is None or not os.path.exists(param_file):
        raise ValueError("param_file cannot be EMPTY")

    os.makedirs(output_dir, exist_ok=True)

    out_id = filter + cat_name

    # -------------------------------------------------------
    # Region selection
    # -------------------------------------------------------
    tab = Table.read(tab_path)

    if shape == 'box':
        tab_n = box(tab, ra_col, dec_col, ra, dec, 0, 0, width, height, angle=ang)

    elif shape == 'circle':
        tab_n = tab.copy()
        tab_n['r'] = angular_separation(
            tab[ra_col]*u.deg,
            tab[dec_col]*u.deg,
            ra*u.deg,
            dec*u.deg
        ).to(u.deg).value

        tab_n = tab_n[tab_n['r'] <= width]

        x_cen = 0.5 * (tab_n['x'].min() + tab_n['x'].max())
        y_cen = 0.5 * (tab_n['y'].min() + tab_n['y'].max())
        r_pix_max = np.sqrt(
            (tab_n['x'] - x_cen)**2 +
            (tab_n['y'] - y_cen)**2
        ).max()

    # -------------------------------------------------------
    # Fake star grid
    # -------------------------------------------------------
    xvals = tab_n['x']
    yvals = tab_n['y']

    xx, yy = np.meshgrid(
        np.linspace(xvals.min() + 10, xvals.max() - 10, nx),
        np.linspace(yvals.min() + 10, yvals.max() - 10, ny)
    )

    x = xx.ravel().astype(int)
    y = yy.ravel().astype(int)

    if shape == 'circle':
        r_pix = np.sqrt((x - x_cen)**2 + (y - y_cen)**2)
        ind = r_pix <= r_pix_max
        x = x[ind]
        y = y[ind]

    ext = np.zeros_like(x)
    chip = np.ones_like(x)

    mags = np.array([np.full(x.shape, mag_) for mag_ in m]).T

    data = {
        'ext': ext,
        'chip': chip,
        'x': x,
        'y': y
    }

    for i in range(len(m)):
        data[f'mag_{i}'] = mags[:, i]

    mag_string = "_".join(map(str, m))

    fake_file = os.path.join(
        output_dir,
        f"fake_{region_name}_{mag_string}_{out_id}.txt"
    )

    pd.DataFrame(data).to_csv(
        fake_file,
        sep=' ',
        index=False,
        header=False
    )

    # -------------------------------------------------------
    # Modify param file
    # -------------------------------------------------------
    with open(param_file) as f:
        dats = f.readlines()

    for n, line in enumerate(dats):
        if 'FakeStars' in line:
            break

    dats[n] = f"FakeStars =   {fake_file}\n"
    dats[n+1] = (
        f"FakeOut =    "
        f"{output_dir}/fake_{region_name}_{mag_string}_{out_id}.fake\n"
    )

    param_file_new = param_file.replace(
        '.param',
        f'_{region_name}.param'
    )

    with open(param_file_new, 'w') as f:
        f.writelines(dats)

    # -------------------------------------------------------
    # Run DOLPHOT fake stars
    # -------------------------------------------------------
    phot_file = os.path.join(output_dir, f"{out_id}_photometry.fits")

    if not os.path.exists(phot_file):
        print(f"{phot_file} NOT FOUND!!")
        return

    print("Running DOLPHOT fake star test...")

    process = subprocess.Popen(
        ["dolphot", f"{output_dir}/out", f"-p{param_file_new}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("DOLPHOT fake star run failed.")

    # -------------------------------------------------------
    # Convert fake output
    # -------------------------------------------------------
    subprocess.run([
        "python",
        f"{script_dir}/to_table_fake.py",
        "--f", f"{output_dir}/fake_{region_name}_{mag_string}_{out_id}.fake",
        "--d", "ACS",
        "--c", f"{output_dir}/out.columns",
        "--o", f"fake_out_{region_name}_{mag_string}_{out_id}"
    ], check=True)

    fake_out_fits = os.path.join(
        output_dir,
        f"fake_out_{region_name}_{mag_string}_{out_id}.fits"
    )

    phot_table = Table.read(fake_out_fits)

    # -------------------------------------------------------
    # Add RA/Dec if requested
    # -------------------------------------------------------
    if ref_img_path is not None:
        with fits.open(ref_img_path) as hdu:
            wcs = WCS(hdu[1].header)
    
        ra, dec = wcs.pixel_to_world_values(
            phot_table["x"] - 0.5,
            phot_table["y"] - 0.5
        )
    
        phot_table["ra"] = ra
        phot_table["dec"] = dec  

    # -------------------------------------------------------
    # Filtering
    # -------------------------------------------------------
    mask = np.ones(len(phot_table), dtype=bool)

    for filt in filter.split("_"):
        filt = filt.upper()
        mask &= (
            (phot_table[f"sharpness_{filt}"]**2 <= sharp_cut) &
            (phot_table[f"crowd_{filt}"] <= crowd_cut) &
            (phot_table[f"SNR_{filt}"] >= SNR_min)
        )

    mask &= phot_table["type"] <= type

    phot_table_filt = phot_table[mask]
    phot_table.write(fake_out_fits, overwrite=True)
    phot_table_filt.write(
        os.path.join(
            output_dir,
            f"fake_out_{region_name}_{mag_string}_{out_id}_filt.fits"
        ),
        overwrite=True
    )

    print("ACS Completeness Completed!")
