from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
import os
import numpy as np
import itertools

def filter_phot(out_dir, region, filt_n):
    os.system(f"python ../scripts/to_table.py --f ../{out_dir}/{region}/{filt_n}/out")
    
    phot_table = Table.read(f"../{out_dir}/{region}/{filt_n}/photometry.fits")
    phot_table.rename_columns(['mag_vega'],[f'mag_vega_{filt_n}'])
    
    hdu = fits.open(f"../{out_dir}/{region}/{filt_n}/data_conv.fits")[1]
    
    wcs = WCS(hdu.header)
    positions = np.transpose([phot_table['x'] - 0.5, phot_table['y']-0.5])
    
    coords = np.array(wcs.pixel_to_world_values(positions))
    phot_table['ra'] = coords[:,0]
    phot_table['dec'] = coords[:,1]
    
    # NIRCAM Filter Short (Warfield et.al)
    if filt_n in ['F115W', 'F150W','F200W']:
        phot_table1 = phot_table[ (phot_table['sharpness']**2 <= 0.01) &
                                    (phot_table['obj_crowd'] <=  0.5) &
                                    (phot_table['flags']     <=    2) &
                                    (phot_table['type']      <=    2)]

        phot_table2 = phot_table[ ~((phot_table['sharpness']**2 <= 0.01) &
                                    (phot_table['obj_crowd'] <=  0.5) &
                                    (phot_table['flags']     <=    2) &
                                    (phot_table['type']      <=    2))]
        print('NIRCAM SHORT')

    if filt_n in ['F277W', 'F335M', 'F444W']:
        # NIRCAM Filter Long
        phot_table1 = phot_table[ (phot_table['sharpness']**2 <= 0.02) &
                                (phot_table['obj_crowd'] <=  0.5) &
                                (phot_table['flags']     <=    2) &
                                (phot_table['type']      <=    2)]

        phot_table2 = phot_table[~((phot_table['sharpness']**2 <= 0.02) &
                            (phot_table['obj_crowd'] <=           0.5) &
                            (phot_table['flags']     <=    2) &
                            (phot_table['type']      <=    2))]
        print('NIRCAM LONG')

    if filt_n in ['F435W', 'F555W', 'F814W']:
        # HST ACS
        phot_table1 = phot_table[ (phot_table['sharpness']**2 <   0.2) &
                                (phot_table['obj_crowd'] <       2.25) &
                                (phot_table['flags']     <=         2) &
                                (phot_table['type']      <=         2) &
                                (phot_table['SNR']       >          4)]

        phot_table2 = phot_table[ ~((phot_table['sharpness']**2 <  0.2) &
                            (phot_table['obj_crowd'] <            2.25) &
                            (phot_table['flags']     <=              2) &
                            (phot_table['type']      <=              2) &
                            (phot_table['SNR']       >               4))]
        print("HST ACS/WFC")

    if filt_n in ['F275W', 'F336W']:
        # HST WFC3
        phot_table1 = phot_table[(phot_table['sharpness']**2 < 0.15) &
                                (phot_table['obj_crowd'] <      1.3) &
                                (phot_table['flags']     <=       2) &
                                (phot_table['type']      <=       2) &
                                (phot_table['SNR']       >        5)]

        phot_table2 = phot_table[~((phot_table['sharpness']**2 < 0.15) &
                            (phot_table['obj_crowd'] <            1.3) &
                            (phot_table['flags']     <=             2) &
                            (phot_table['type']      <=             2) &
                            (phot_table['SNR']       >              5))]
        print("HST WFC3/UVIS")

    phot_table1['flag_phot'] = 1
    phot_table2['flag_phot'] = 0 

    phot_table = vstack([phot_table1, phot_table2])
    phot_table.write(f"../{out_dir}/{region}/{filt_n}/{filt_n}_photometry_filt.fits",overwrite=True)
    with open(f"../{out_dir}/{region}/{filt_n}/{filt_n}.reg",'w+') as f:
        f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs\n""")
        temp = phot_table1[phot_table1['mag_err']<0.2]
        for row in temp:
            ra = np.round(row['ra'], 7)
            dec = np.round(row['dec'],7)
            f.write(f"""circle({ra},{dec},0.03")\n""")
def generate_jwst_color(out_dir, regions, filters):
    for region in regions.keys():
        for filt in itertools.combinations(filters,2):

            in1 = f"../{out_dir}/{region}/{filt[0]}/{filt[0]}_photometry_filt.fits"
            in2 = f"../{out_dir}/{region}/{filt[1]}/{filt[1]}_photometry_filt.fits"
            out1 = f"../{out_dir}/{region}/t1.fits"
            out2 = f"../{out_dir}/{region}/t2.fits"

            out = f"../{out_dir}/{region}/{filt[0].lower()}_{filt[1].lower()}.fits"

            os.system(f"""topcat -stilts tpipe in={in1} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out1}""")

            os.system(f"""topcat -stilts tpipe in={in2} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out2}""")

            os.system(f"topcat -stilts tskymatch2 in1={out1} in2={out2} out={out} \
                        ra1=ra dec1=dec ra2=ra dec2=dec error=0.06")
def generate_acs_color(out_dir, regions, filters):
    for region in regions.keys():    

        for filt in itertools.combinations(filters,2):

            in1 = f"../{out_dir}/{region}/{filt[0]}/{filt[0]}_photometry_filt.fits"
            in2 = f"../{out_dir}/{region}/{filt[1]}/{filt[1]}_photometry_filt.fits"
            out1 = f"../{out_dir}/{region}/t1.fits"
            out2 = f"../{out_dir}/{region}/t2.fits"

            out = f"../{out_dir}/{region}/{filt[0].lower()}_{filt[1].lower()}.fits"

            os.system(f"""topcat -stilts tpipe in={in1} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out1}""")

            os.system(f"""topcat -stilts tpipe in={in2} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out2}""")

            os.system(f"topcat -stilts tskymatch2 in1={out1} in2={out2} out={out} \
                        ra1=ra dec1=dec ra2=ra dec2=dec error=0.1")

def generate_wfc3_color(out_dir, regions, filters):
    for region in regions.keys():    

        for filt in itertools.combinations(filters,2):

            in1 = f"../{out_dir}/{region}/{filt[0]}/{filt[0]}_photometry_filt.fits"
            in2 = f"../{out_dir}/{region}/{filt[1]}/{filt[1]}_photometry_filt.fits"
            out1 = f"../{out_dir}/{region}/t1.fits"
            out2 = f"../{out_dir}/{region}/t2.fits"

            out = f"../{out_dir}/{region}/{filt[0].lower()}_{filt[1].lower()}.fits"

            os.system(f"""topcat -stilts tpipe in={in1} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out1}""")

            os.system(f"""topcat -stilts tpipe in={in2} \
                        cmd='select "flag_phot==1 && mag_err<=0.2"' \
                        out={out2}""")

            os.system(f"topcat -stilts tskymatch2 in1={out1} in2={out2} out={out} \
                          ra1=ra dec1=dec ra2=ra dec2=dec error=0.08")
