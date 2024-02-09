from matplotlib.pyplot import plt
from astropy.table import Table
import numpy as np
from astropy.coordinates import angular_separation
import astropy.units as u
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.ticker import AutoMinorLocator, AutoLocator

Av_dict = { 
            'f275w': 2.02499,
            'f336w': 1.67536,
            'f435w': 1.33879,
            'f555w': 1.03065,
            'f814w': 0.59696,
            'f115w': 0.419,
            'f150w': 0.287,
            'f200w': 0.195,
          }

def gen_CMD(filt1='f115w', filt2='f150w',name=None, Av=0.19, r=333, r_in=None, 
            r_out=None, dismod=29.95, xlims=[-0.5,2.5], ylims=[18,28],label=3, met=0.02,
           cmd=None, out_dir='.', Av_ = 3, Av_x=2, Av_y=19, mag_err_lim=0.2,
           gen_kde=False, add_ref=False, plot_regions=['bubble'],add_ext='',
           ra_col = 'ra_1', dec_col= 'dec_1',regions={},
           ref_xpos=-0.25, ages=[7.,8.,9.]):
    
    if met is not None:
        cmd = cmd[cmd['Zini']==met]

    age_lin = []
    for i in ages:
        if i > 6  and i <9:
            i-=6
            age_lin.append(f'{np.ceil(10**i)} Myr')
        elif i >= 9:
            i-=9
            age_lin.append(f'{np.ceil(10**i)} Gyr')
                  
    fig = plt.figure(figsize=(len(plot_regions)*12,10))
    for i,region in enumerate(plot_regions):
        
        ax = fig.add_subplot(1,len(plot_regions),int(i+1))
        
        if name is None:
            tab_bub = Table.read(f"../{out_dir}/{region}/{filt1}_{filt2}{add_ext}.fits")
        else:
            tab_bub = Table.read(f"../{out_dir}/{region}/{name}.fits")
        
        mag_errs = [i for i in tab_bub.keys() if 'mag_err' in i]
        
        for i in mag_errs:
            tab_bub = tab_bub[tab_bub[i]<=mag_err_lim]


        AF1 =  Av_dict[filt1]*Av
        AF2 =  Av_dict[filt2]*Av
        
        ra_cen = regions[region]['ra']
        dec_cen = regions[region]['dec']
        
        tab_bub['r'] = angular_separation(tab_bub[ra_col]*u.deg,tab_bub[dec_col]*u.deg,
                                          ra_cen*u.deg, dec_cen*u.deg).to(u.arcsec).value
        
        if (r_out is None or r_in is None) :
            r_in = 0
            r_out = r
            
        if r_out is not None:    
            tab_bub = tab_bub[ (tab_bub['r']>=r_in) & (tab_bub['r'] <r_out)]

        x = tab_bub[f'mag_vega_{filt1.upper()}'] - tab_bub[f'mag_vega_{filt2.upper()}']
        y = tab_bub[f'mag_vega_{filt2.upper()}'] 
        
        x = x.value.astype(float)
        y = y.value.astype(float)
        
        n_sources = len(x)
        
        if gen_kde:

            # Peform the kernel density estimate
            xx, yy = np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])

            kernel = gaussian_kde(values, bw_method=0.075)
            f = np.reshape(kernel(positions), xx.shape)

            f = f.T
            img = ax.imshow(f, cmap='jet', 
                          extent=[xlims[0], xlims[1], 
                                  ylims[0], ylims[1]],
                           interpolation='nearest', aspect='auto')
            
        else:
            ax.scatter(x,y, s=0.2, color='black', label='data')
            
        ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        ax1.set_ylabel(r'$M_{' + f'{filt2.upper()}' + r'}$')  # we already handled the x-label with ax1

        ax.set_xlabel(f"{filt1.upper()} - {filt2.upper()}")
        ax.set_ylabel(filt2.upper())
        

        ref = tab_bub[f'mag_vega_{filt2.upper()}']
        ref_new = np.arange(np.ceil(y.min()),np.floor(y.max()),0.5)
        
        if 'mag_err_1' in tab_bub.keys():
            mag_err1 = interp1d(ref, tab_bub['mag_err_1'])(ref_new)
            mag_err2 = interp1d(ref, tab_bub['mag_err_2'])(ref_new)
            
        elif 'mag_err_1_1' in tab_bub.keys():
            mag_err1 = interp1d(ref, tab_bub['mag_err_1_1'])(ref_new)
            mag_err2 = interp1d(ref, tab_bub['mag_err_2_1'])(ref_new)

        col_err = np.sqrt(mag_err1**2 + mag_err2**2)

        x = ref_xpos + 0*ref_new
        y = ref_new
        yerr = mag_err1
        xerr = col_err

        ax.errorbar(x, y,yerr,xerr,fmt='o', color = 'red', markersize=0.5, capsize=2) 
        
        if cmd is not None:
            for i,age in enumerate(ages):

                t = cmd[np.round(cmd['logAge'],1)==age].copy()
                
                t_ = t[t['label']<=label]
                x =  (t_[f'{filt1.upper()}mag'] + AF1) - (t_[f'{filt2.upper()}mag'] + AF2)
                y =  t_[f'{filt2.upper()}mag'] + AF2 + dismod
                ax.plot(x,y, linewidth=3, label=age_lin[i], alpha=0.6)
                
        else: 
            met = ' '

        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlim(xlims[0], xlims[1]) 
        
        yticks = ax.get_yticks()
        yticks_n = yticks - dismod
        
        dy = yticks_n - np.floor(yticks_n)
        ax1.set_yticks(yticks + dy, np.floor(yticks_n))
        
        ax1.set_ylim(ylims[0], ylims[1])
        ax1.set_xlim(xlims[0], xlims[1]) 
        
        ax.invert_yaxis()
        ax1.invert_yaxis()
        ax.set_title(region.capitalize() + f" | No of sources: {n_sources} | Z : {met}", fontsize=30)
            
        if add_ref:
            ax.plot([xlims[0], xlims[1]],[25,25], '--r', zorder=200)
            ax.plot([xlims[0], xlims[1]],[24.5,24.5], '--r', zorder=200)
            ax.plot([xlims[0], xlims[1]],[23,23], '--r', zorder=200)
         
            
        ax.legend(fontsize=15)
        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(which='both', length=14,direction="in", bottom=True, top=True,left=True)
        ax.tick_params(which='minor', length=8)
        
        ax1.tick_params(which='both', length=14,direction="in", right=True)
        ax1.tick_params(which='minor', length=8)

        AF1_ =  Av_dict[filt1]*Av_
        AF2_ =  Av_dict[filt2]*Av_

        dx = AF1_ - AF2_
        dy = AF1_

        ax.annotate('', xy=(Av_x, Av_y),
                     xycoords='data',
                     xytext=(Av_x+dx, Av_y+dy),
                     textcoords='data',
                     arrowprops=dict(arrowstyle= '<|-',
                                     color='black',
                                     lw=0.5,
                                     ls='-')
                   )
            
        ax.annotate(f'Av = {Av_}', xy=(Av_x-0.1, Av_y-0.1))
        
    fig.tight_layout()
    return fig, ax