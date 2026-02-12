import os
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import angular_separation
from astropy import units as u

from .scripts.catalog_filter import box


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

param_dir_default = Path(__file__).parent / "params"
script_dir = Path(__file__).parent / "scripts"


# ---------------------------------------------------------------------
# Clean parameter dictionaries (duplicate key fixed)
# ---------------------------------------------------------------------

sw_params = {
    "RAper": "2",
    "Rchi": "1.5",
    "RSky0": "15",
    "RSky1": "35",
    "RSky2": "3 10",
    "RPSF": "15",
    "apsky": "20 35",
    "aprad": "10",
}

lw_params = {
    "RAper": "3",
    "Rchi": "2",
    "RSky0": "15",
    "RSky1": "35",
    "RSky2": "4 10",
    "RPSF": "15",
    "apsky": "20 35",
    "aprad": "10",
}


# ---------------------------------------------------------------------
# Helper: Safe subprocess
# ---------------------------------------------------------------------

def run_cmd(cmd):
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def nircam_phot(
    input_files,
    filter="f200w",
    output_dir=".",
    ref_img_path=".",
    cat_name="",
    param_file=None,
    sharp_cut=0.01,
    crowd_cut=0.5,
    SNR_min=5,
    type=2,
):

    if not input_files:
        raise ValueError("input_files cannot be empty")

    if not all(i.lower().endswith(".fits") for i in input_files):
        raise ValueError("All input files must be FITS")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_img = Path(f"{ref_img_path}.fits")

    # Apply mask to reference
    run_cmd(["nircammask", str(ref_img)])

    if param_file is None or not Path(param_file).exists():
        logger.info("Using default parameter file")
        param_file = param_dir_default / "nircam_dolphot.param"
        edit_params = True
    else:
        edit_params = False

    out_id = filter + cat_name

    # --------------------------------------------------------------
    # Prepare exposures
    # --------------------------------------------------------------

    exps = []
    filter_wavelengths = []

    for f in input_files:
        f = Path(f)
        exp_dir = output_dir / f.stem
        exp_dir.mkdir(exist_ok=True)

        data_path = exp_dir / "data.fits"
        if not data_path.exists():
            data_path.write_bytes(f.read_bytes())

        with fits.open(data_path) as hdul:
            filter_wavelength = float(hdul[0].header["FILTER"][1:-1])
            filter_wavelengths.append(filter_wavelength)

        run_cmd(["nircammask", str(data_path)])
        run_cmd(["calcsky", str(exp_dir / "data"), "10", "25", "2", "2.25", "2.00"])

        exps.append(exp_dir)

    # --------------------------------------------------------------
    # Build parameter file
    # --------------------------------------------------------------

    if edit_params:

        with open(param_file) as f:
            dat = f.readlines()

        dat[0] = f"Nimg = {len(exps)}\n"
        dat[4] = f"img0_file = {ref_img_path}\n"

        img_lines = []

        for i, exp in enumerate(exps):
            img_lines.append(f"img{i+1}_file = {exp}/data\n")

            param_set = sw_params if filter_wavelengths[i] <= 200 else lw_params

            for key, val in param_set.items():
                img_lines.append(f"img{i+1}_{key} = {val}\n")

        dat[5:6] = img_lines

        param_file = output_dir / f"nircam_dolphot_{out_id}.param"
        with open(param_file, "w") as f:
            f.writelines(dat)

    # --------------------------------------------------------------
    # Run DOLPHOT
    # --------------------------------------------------------------

    phot_file = output_dir / f"{out_id}_photometry.fits"

    if not phot_file.exists():
        run_cmd(["dolphot", str(output_dir / "out"), f"-p{param_file}"])
        run_cmd([
            "python",
            str(script_dir / "to_table.py"),
            "--o", f"{out_id}_photometry",
            "--f", str(output_dir / "out"),
            "--d", "NIRCAM",
        ])

    phot_table = Table.read(phot_file)

    # --------------------------------------------------------------
    # Assign WCS
    # --------------------------------------------------------------

    with fits.open(ref_img) as hdu:
        wcs = WCS(hdu[1].header)

    ra, dec = wcs.pixel_to_world_values(
        phot_table["x"] - 0.5,
        phot_table["y"] - 0.5
    )

    phot_table["ra"] = ra
    phot_table["dec"] = dec

    # --------------------------------------------------------------
    # Filtering (FIXED multi-filter logic)
    # --------------------------------------------------------------

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

    phot_table.write(phot_file, overwrite=True)
    phot_table_filt.write(
        output_dir / f"{out_id}_photometry_filt.fits",
        overwrite=True
    )

    logger.info("NIRCAM Stellar Photometry Completed")

    return phot_table, phot_table_filt

def nircam_phot_comp(
    param_file,
    m=[20],
    filter="f200w",
    region_name="3",
    output_dir=".",
    tab_path=".",
    ref_img_path=None,
    cat_name="",
    sharp_cut=0.01,
    crowd_cut=0.5,
    SNR_min=5,
    type=2,
    ra_col="ra",
    dec_col="dec",
    ra=0,
    dec=0,
    shape="box",
    width=24/3600,
    height=24/3600,
    ang=245,
    nx=10,
    ny=10,
):

    param_file = Path(param_file)
    output_dir = Path(output_dir)
    tab_path = Path(tab_path)

    if not param_file.exists():
        raise ValueError("param_file does not exist")

    output_dir.mkdir(parents=True, exist_ok=True)

    out_id = filter + cat_name

    # --------------------------------------------------------------
    # Load photometry table
    # --------------------------------------------------------------

    tab = Table.read(tab_path)

    # --------------------------------------------------------------
    # Region selection
    # --------------------------------------------------------------

    if shape == "box":
        tab_n = box(tab, ra_col, dec_col, ra, dec, 0, 0,
                    width, height, angle=ang)

    elif shape == "circle":

        tab_n = tab.copy()

        sep = angular_separation(
            tab[ra_col] * u.deg,
            tab[dec_col] * u.deg,
            ra * u.deg,
            dec * u.deg
        ).to(u.deg).value

        tab_n["r"] = sep
        tab_n = tab_n[tab_n["r"] <= width]

        if len(tab_n) == 0:
            raise ValueError("No sources inside circular region")

        x_cen = 0.5 * (tab_n["x"].min() + tab_n["x"].max())
        y_cen = 0.5 * (tab_n["y"].min() + tab_n["y"].max())

        r_pix_max = np.sqrt(
            (tab_n["x"] - x_cen)**2 +
            (tab_n["y"] - y_cen)**2
        ).max()

    else:
        raise ValueError("shape must be 'box' or 'circle'")

    # --------------------------------------------------------------
    # Generate fake star grid
    # --------------------------------------------------------------

    x = tab_n["x"]
    y = tab_n["y"]

    xx, yy = np.meshgrid(
        np.linspace(x.min() + 10, x.max() - 10, nx),
        np.linspace(y.min() + 10, y.max() - 10, ny)
    )

    x = xx.ravel().astype(int)
    y = yy.ravel().astype(int)

    if shape == "circle":
        r_pix = np.sqrt((x - x_cen)**2 + (y - y_cen)**2)
        mask_circle = r_pix <= r_pix_max
        x = x[mask_circle]
        y = y[mask_circle]

    ext = np.ones_like(x)
    chip = np.ones_like(x)

    # Build magnitude grid
    mags = np.array([np.full(len(x), mag) for mag in m]).T

    columns = {
        "ext": ext,
        "chip": chip,
        "x": x,
        "y": y,
    }

    for i in range(len(m)):
        columns[f"mag_{i}"] = mags[:, i]

    mag_label = "_".join(map(str, m))
    fake_file = output_dir / f"fake_{region_name}_{mag_label}_{out_id}.txt"

    df = pd.DataFrame(columns)
    df.to_csv(fake_file, sep=" ", index=False, header=False)

    # --------------------------------------------------------------
    # Edit parameter file for fake stars
    # --------------------------------------------------------------

    with open(param_file) as f:
        dats = f.readlines()

    fake_line_index = None
    for i, line in enumerate(dats):
        if line.strip().startswith("FakeStars"):
            fake_line_index = i
            break

    if fake_line_index is None:
        raise RuntimeError("FakeStars entry not found in parameter file")

    dats[fake_line_index] = f"FakeStars =   {fake_file}\n"
    dats[fake_line_index + 1] = (
        f"FakeOut =    {output_dir}/"
        f"fake_{region_name}_{mag_label}_{out_id}.fake\n"
    )

    param_file_new = param_file.with_name(
        param_file.stem + f"_{region_name}.param"
    )

    with open(param_file_new, "w") as f:
        f.writelines(dats)

    # --------------------------------------------------------------
    # Run DOLPHOT with fake stars
    # --------------------------------------------------------------

    phot_file = output_dir / f"{out_id}_photometry.fits"

    if not phot_file.exists():
        raise RuntimeError("Base photometry must exist before completeness")

    run_cmd([
        "dolphot",
        str(output_dir / "out"),
        f"-p{param_file_new}"
    ])

    fake_output_base = (
        f"fake_out_{region_name}_{mag_label}_{out_id}"
    )

    run_cmd([
        "python",
        str(script_dir / "to_table_fake.py"),
        "--f",
        str(output_dir / f"fake_{region_name}_{mag_label}_{out_id}.fake"),
        "--c",
        str(output_dir / "out.columns"),
        "--o",
        fake_output_base,
    ])

    fake_fits = output_dir / f"{fake_output_base}.fits"
    phot_table = Table.read(fake_fits)

    # --------------------------------------------------------------
    # Assign WCS (optional)
    # --------------------------------------------------------------

    if ref_img_path is not None:

        ref_img = Path(f"{ref_img_path}.fits")

        with fits.open(ref_img) as hdu:
            wcs = WCS(hdu[1].header)

        ra_vals, dec_vals = wcs.pixel_to_world_values(
            phot_table["x"] - 0.5,
            phot_table["y"] - 0.5
        )

        phot_table["ra"] = ra_vals
        phot_table["dec"] = dec_vals

    # --------------------------------------------------------------
    # Filtering (FIXED multi-filter logic)
    # --------------------------------------------------------------

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

    phot_table.write(fake_fits, overwrite=True)
    phot_table_filt.write(
        output_dir / f"{fake_output_base}_filt.fits",
        overwrite=True
    )

    logger.info("NIRCAM Completeness Completed")

    return phot_table, phot_table_filt
   
