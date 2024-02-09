import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
import pista as pt
from pista import data_dir as data_path
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import  AutoMinorLocator
from astropy.modeling import models, fitting


def sim_stars(out_dir, region, filt_n, d=10, mag=22, n=100):
    
    r = d/2
    
    if not os.path.exists(f"../PHOT_OUTPUT_m50/{region}/"):
        os.mkdir(f"../PHOT_OUTPUT_m50/{region}/")
        
    if not os.path.exists(f"../PHOT_OUTPUT_m50/{region}/{filt_n}"):
        os.mkdir(f"../PHOT_OUTPUT_m50/{region}/{filt_n}")
    else:
        os.system(f"rm -r ../PHOT_OUTPUT_m50/{region}/{filt_n}/*")
                
    hdu = fits.open(f"../{out_dir}/{region}/{filt_n}/data_conv.fits")[1]

    data_source = hdu.data
    
    if 'CD1_1' in hdu.header.keys() and 'CD1_2' in hdu.header.keys():
        pixel_scale = np.sqrt(hdu.header['CD1_1']**2 + hdu.header['CD1_2']**2)*3600
        
    elif 'CDELT1' in hdu.header.keys():
        pixel_scale = hdu.header['CDELT1']*3600
        
    phot_table = Table.read(f"../{out_dir}/{region}/{filt_n}/{filt_n}_photometry_filt.fits")

    #psf = np.median(fits.open(f'../data/PSF/epsf/{filt_n}/snap_test_psf.fits')[0].data, axis=(1,0))
    psf = fits.open(f'../data/PSF/epsf/{filt_n}/snap_test_psf.fits')[0].data[0,0]
    psf = psf.reshape(51,5,51,5).sum(axis=(1,3))
    psf /= psf.sum()

    hdu = fits.PrimaryHDU(psf)
    hdul = fits.HDUList([hdu])
    hdul.writeto('psf.fits', overwrite=True)

    tel_params ={
                'aperture'       : 650,
                'pixel_scale'    : pixel_scale,
                'psf_file'       : f'psf.fits',
                'response_funcs' :  [ f'{data_path}/INSIST/UV/Coating.dat,5,100',   # 6 mirrors
                                    ],
                 'coeffs'        : 0.5 ,
                 'theta'         : 0
                }

    df = phot_table[ (phot_table['SNR']<50) ][['ra', 'dec',f'mag_vega_{filt_n}', 'flux','flag_phot']].to_pandas()

    df = df[df['flag_phot']==1]
    zero_flux = (df['flux']/10**(-0.4*df[f'mag_vega_{filt_n}'])).mean()
    zp = 2.5*np.log10(zero_flux)
    
    dZP = {'F115W' : 0.5,
           'F150W' : 0.3,
           'F200W' : 0.1,
           
           'F435W' : 0.3,
           'F555W' : 0.3,
           'F814W' : 0.6,
           
           'F275W' : 0.9,
           'F336W' : 0.7
           
            }
    
    zero_flux = 10**(0.4*(zp + dZP[filt_n]))
    
    df = df.rename(columns = {f'mag_vega_{filt_n}': 'mag'})
    sim = pt.Imager(df=df, tel_params=tel_params, exp_time=2000,
                    n_x=data_source.shape[0], n_y=data_source.shape[1])
                          
    sim.shot_noise = False
                          
    det_params = {'shot_noise' :  'Poisson',
                  'qe_response': [],
                  'qe_mean'    : 1,
                  'G1'         :  1,
                  'bias'       : 10,
                  'PRNU_frac'  :  0.25/100,
                  'DCNU'       :  0.1/100,
                  'RN'         :  3,
                  'T'          :  218,
                  'DN'         :  0.01/100
                  }

    sim(det_params=det_params)
    
    r_pix = r/pixel_scale
    
    x_min  = sim.n_x_sim/2 - r_pix + 26
    x_max  = sim.n_x_sim/2 + r_pix - 26
    
    y_min  = sim.n_y_sim/2 - r_pix + 26
    y_max  = sim.n_y_sim/2 + r_pix - 26
    
    x = np.linspace(x_min,x_max, int(n**0.5))
    y = np.linspace(y_min,y_max, int(n**0.5))
    
    x, y = np.meshgrid(x, y)
    
    x = x.ravel()
    y = y.ravel()
    
    mag_ = np.ones(n)*mag
    
    df_add  = pd.DataFrame(zip(x,y,mag_), columns = ['x','y','mag'])
    out_img = sim.add_stars(data_source, zero_flux, df_add)

    hdu = fits.open(f"../{out_dir}/{region}/{filt_n}/data_conv.fits")
    
    hdu[1].data = out_img
    wcs = WCS(hdu[1].header)
    
    dx = {'F115W' : -1,
          'F150W' : 0,
          'F200W' : 0,
          'F435W' : 0,
          'F555W' : 0,
          'F814W' : 0,
          'F275W' : -1,
          'F336W' : -1}
    
    dy = {'F115W' : -1,
          'F150W' : 0,
          'F200W' : 0,
          'F435W' : -1,
          'F555W' : -1,
          'F814W' : -1,
          'F275W' : -1,
          'F336W' : -1}
            
    coords = np.array(wcs.array_index_to_world_values(y - 50 + dy[filt_n],
                                                      x - 50 + dx[filt_n]))
    
    df_add['x'] = x - 49 
    df_add['y'] = y - 49
    
    df_add['ra'] = coords[0,:]
    df_add['dec'] = coords[1,:]
    
    df_add = Table.from_pandas(df_add)
    
    df_add.write(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/add_stars.fits", overwrite=True)
    hdu.writeto(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/data_conv.fits", output_verify='ignore',
            overwrite=True)
    
def comp_base(out_dir, region, filt_n, r=0.06, mag=22):
    in1 = f"../PHOT_OUTPUT_m50/{region}/{filt_n}/add_stars.fits"
    in2 = f"../PHOT_OUTPUT_m50/{region}/{filt_n}/{filt_n}_photometry_filt.fits"

    out_t = f"../PHOT_OUTPUT_m50/{region}/{filt_n}/matched_t.csv"
    out = f"../PHOT_OUTPUT_m50/{region}/{filt_n}/matched.csv"

    #os.system(f"topcat -stilts tskymatch2 in1={in1} in2={in2} out={out} \
     #             ra1=ra dec1=dec ra2=ra dec2=dec error={r}") join=1and2 find=all 
    
    os.system(f"""topcat -stilts tmatch2 in1={in1} in2={in2} \
               matcher=sky+1d params='{r} 0.5' \
               values1='ra dec mag' values2='ra dec mag_vega_{filt_n}'  \
               out={out_t}""")
       
    df_match = pd.read_csv(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/matched_t.csv") 
    
    if len(df_match)>0:
        os.system(f"""topcat -stilts tpipe in={out_t} \
                        cmd='select "mag_err<=0.2"' \
                        out={out}""")

        df_match = pd.read_csv(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/matched.csv") 
            
    with open(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/{filt_n}_match_0.reg",'w+') as f:
        f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs\n""")
        for i, row in df_match[df_match['flag_phot']==0].iterrows():
            ra = np.round(row['ra_1'], 7)
            dec = np.round(row['dec_1'],7)
            f.write(f"""circle({ra},{dec},0.06")\n""")
        
    df_match = df_match[df_match['flag_phot']==1]
    
    with open(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/{filt_n}_match_1.reg",'w+') as f:
        f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs\n""")
        for i, row in df_match.iterrows():
            ra = np.round(row['ra_1'], 7)
            dec = np.round(row['dec_1'],7)
            f.write(f"""circle({ra},{dec},0.06")\n""")
            
    df_match1 = df_match[abs(df_match['mag'] - df_match[f'mag_vega_{filt_n}'])<0.5]
    df_match2 = df_match[abs(df_match['mag'] - df_match[f'mag_vega_{filt_n}'])>=0.5]
    
    with open(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/{filt_n}_match_2.reg",'w+') as f:
        f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs\n""")
        for i, row in df_match1.iterrows():
            ra = np.round(row['ra_1'], 7)
            dec = np.round(row['dec_1'],7)
            f.write(f"""circle({ra},{dec},0.06")\n""")
            
    with open(f"../PHOT_OUTPUT_m50/{region}/{filt_n}/{filt_n}_match_3.reg",'w+') as f:
        f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs\n""")
        for i, row in df_match2.iterrows():
            ra = np.round(row['ra_1'], 7)
            dec = np.round(row['dec_1'],7)
            f.write(f"""circle({ra},{dec},0.06")\n""")
    
    print(mag, len(df_match1))
    
    return len(df_match1)

def comp_fit(out_dir, region, filt_n, x, y):
    
    init = pritchet()
    fit = fitting.LevMarLSQFitter()
    offset = y.max()
    model = fit(init, x, y/offset)

    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(x,y)

    ax.plot(x, model(x)*offset, '--r')
    ax.set_xlabel('mags')
    ax.set_ylabel(r'$N_{out}/N_{in}$')
    ax.set_title(f"{region} | {filt_n} | "+ r"$\alpha =$" + f" {np.round(model.alpha.value,2)}" + r" | $m_{50}=$" + f"{np.round(model.m_50.value,2)}")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', width=2,direction="in", top = True,right = True,
                   bottom = True, left = True)
    ax.tick_params(which='major', length=7,direction="in")
    ax.tick_params(which='minor', length=4, color='black',direction="in")
    ax.set_ylim(0,1)
    fig.savefig(f"../{out_dir}/{region}/{filt_n}/{filt_n}_comp_{region}.png")
    plt.close(fig)

@models.custom_model
def pritchet(m,alpha=0.5,m_50=30):
    return 0.5*(1 - alpha*(m - m_50)/np.sqrt(1 + alpha**2*(m-m_50)**2))