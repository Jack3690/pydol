import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import angular_separation
import astropy.units as u
from scipy.stats import gaussian_kde
from astropy.modeling import models, fitting
import seaborn as sb
from .catalog_filter import box
from matplotlib.colors import LinearSegmentedColormap

sb.set_style('white')

from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize'] = 35
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 35

font1 = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': '15'}
font2 = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': '25'}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

Av_dict = { 
            'f275w': 2.02499,
            'f336w': 1.67536,
            'f435w': 1.33879,
            'f555w': 1.03065,
            'f814w': 0.59696,
            'f115w': 0.419,
            'f150w': 0.287,
            'f200w': 0.195,
    
            'f438w': 1.34148,
            'f606w': 0.90941,
            'f814w': 0.59845
          }

def running_avg(x,y, nbins=100, mode='median'):
    bins = np.linspace(x.min(),x.max(), nbins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(x,bins)
    if mode=='mean':
        running_median = [np.mean(y[idx==k]) for k in range(nbins)]
    elif mode=='median':
        running_median = [np.median(y[idx==k]) for k in range(nbins)]
    else:
        raise Exception(f"""Input mode="{mode}" NOT available""")
    return bins, np.array(running_median)

def gen_CMD(tab, filt1='f115w', filt2='f150w', filt3=None, ra_col = 'ra_1', dec_col= 'dec_1',
            ra_cen=0, dec_cen=0, r_in=0, r_out=24, Av=0.19,  Av_ = 3, Av_x=2, Av_y=19,
             sqr_field=False, ab_dist = True,  dismod=29.7415, xlims=[-0.5,2.5], ylims=[18,28], met=0.02, gen_contours=False,cmd=None,  gen_kde=False, fig = None, ax= None, label_min=None, label_max=None, ages=[7.,8.,9.], alpha=1, s =0.2,lw=3, ref_xpos=-0.25, skip_data = False,
            ang=245.00492, mag_err_cols = ['mag_err_1','mag_err_2'], show_err_model=False):
    
    # Filtering Isochrone evolutionary phases
    if label_min is None or label_max is None:
        label_min = 0
        label_max = 10
        
    if filt3 is None:
        filt3 = filt2
        
    if met is not None and cmd is not None:
        cmd = cmd[cmd['Zini']==met]
        
    age_lin = []
    
    for i in ages:
        if i<6:
            age_lin.append(f'{np.round(10**i,1)} Myr')
        if i >= 6  and i <9:
            i-=6
            age_lin.append(f'{np.round(10**i,1)} Myr')
        elif i >= 9:
            i-=9
            age_lin.append(f'{np.round(10**i,1)} Gyr')
            
    if fig is None or ax is None:        
        fig, ax = plt.subplots(figsize=(12,10))   

    AF1     = Av_dict[filt1]*Av
    AF2     = Av_dict[filt2]*Av
    AF3     = Av_dict[filt3]*Av

    tab['r'] = angular_separation(tab[ra_col]*u.deg,tab[dec_col]*u.deg,
                                      ra_cen*u.deg, dec_cen*u.deg).to(u.arcsec).value

    if not sqr_field:
        tab = tab[(tab['r']>=r_in) & (tab['r'] <=r_out)]
    else:

        tab = box(tab, ra_col, dec_col,  ra_cen, dec_cen,
                      r_out/3600, r_out/3600, ang)

    x = tab[f'mag_vega_{filt1.upper()}'] - tab[f'mag_vega_{filt2.upper()}']
    y = tab[f'mag_vega_{filt3.upper()}'] 

    x = x.value.astype(float)
    y = y.value.astype(float)

    n_sources = len(x)

    if gen_kde and not gen_contours:
        # Peform the kernel density estimate
        xx, yy = np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = gaussian_kde(values, bw_method=0.05)
        f = np.reshape(kernel(positions), xx.shape)

        f = f.T
        img = ax.imshow(f, cmap='gray_r', 
                      extent=[xlims[0], xlims[1], 
                              ylims[0], ylims[1]],
                       interpolation='nearest', aspect='auto')


    elif gen_contours:
        cmap_custom = LinearSegmentedColormap.from_list("custom_grey_to_white", ["grey", "white"], N=256)
        ax.scatter(x,y, s=0.2, color='black', label='data')
        sb.kdeplot(x=x, y=y, levels=[0.1, 0.25,0.5,0.75,0.9,0.95,0.99],
                   ax=ax, fill=True, cmap=cmap_custom)
        sb.kdeplot(x=x, y=y, levels=[0.1, 0.25,0.5,0.75,0.9,0.95,0.99],
                   ax=ax,  color='black')


    elif not skip_data:
        ax.scatter(x,y, s=s, color='black', label='data')

    if ab_dist:

        ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        ax1.set_ylabel(r'$M_{' + f'{filt3.upper()}' + r'}$')  # we already handled the x-label with ax1

    ax.set_xlabel(f"{filt1.upper()} - {filt2.upper()}")
    ax.set_ylabel(filt3.upper())

    # Error bars

    ref = tab[f'mag_vega_{filt3.upper()}']
    ref_new = np.arange(np.ceil(y.min()),np.floor(y.max())+0.5,0.5)

    mag_err1 = tab[mag_err_cols[0]]
    mag_err2 = tab[mag_err_cols[1]]

    if len(mag_err_cols)>2:
        mag_err3 = tab[mag_err_cols[2]]
    else:
        mag_err3 = mag_err2

    col_err = np.sqrt(mag_err1**2 + mag_err2**2)

    if not skip_data:
        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_col = fit(init,ref,col_err)

        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_mag = fit(init,ref,mag_err3)

        x = ref_xpos + 0*ref_new
        y = ref_new
        yerr = model_mag(ref_new)
        xerr = model_col(ref_new)
        
        if show_err_model:
            plt.show()
            plt.scatter(ref, mag_err3)
            plt.plot(ref_new,yerr,'--r')
            plt.show()
            plt.scatter(ref, col_err)
            plt.plot(ref_new,xerr,'--r')
            plt.show()
        ax.errorbar(x, y, yerr,xerr ,fmt='o', color = 'red', markersize=0.5, capsize=2) 

    if cmd is not None:
        for i, age in enumerate(ages):
            t = cmd[np.round(cmd['logAge'],4)==age].copy()

            mags = np.unique(t['Zini'])

            for Z in mags:
                t_ = t[t['Zini']==Z]
                t_ = t_[(t_['label']>=label_min) & (t_['label']<=label_max)]

                x =  (t_[f'{filt1.upper()}mag'] + AF1) - (t_[f'{filt2.upper()}mag'] + AF2)
                y =  t_[f'{filt3.upper()}mag'] + AF3 + dismod

                mask = (y.values[1:]- y.values[:-1])<1
                mask = np.array([True] + list(mask))
                mask = np.where(~mask, np.nan, 1)

                if len(mags)>1:
                    ax.plot(x*mask, y*mask, linewidth=lw, label=age_lin[i]+ f' {Z}', alpha=alpha)
                else:
                    ax.plot(x*mask, y*mask, linewidth=lw, label=age_lin[i], alpha=alpha)

    else: 
        met = ' '

    ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlim(xlims[0], xlims[1]) 

    yticks = ax.get_yticks()
    yticks_n = yticks - dismod - AF3

    dy = yticks_n - np.floor(yticks_n)

    if ab_dist:
        ax1.set_yticks(yticks - dy, np.floor(yticks_n))

        ax1.set_ylim(ylims[0], ylims[1])
        ax1.set_xlim(xlims[0], xlims[1]) 

    ax.invert_yaxis()
    if ab_dist:
        ax1.invert_yaxis()


    ax.legend(fontsize=15)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


    ax.tick_params(which='both', length=15,direction="in", bottom=True, top=True,left=True, width = 3)
    ax.tick_params(which='minor', length=8, width = 3)

    if ab_dist:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which='both', length=15,direction="in", right=True, width = 3)
        ax1.tick_params(which='minor', length=8, width = 3)

    AF1_ =  Av_dict[filt1]*Av_
    AF2_ =  Av_dict[filt2]*Av_
    AF3_ =  Av_dict[filt3]*Av_

    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(Av_x, Av_y),
                 xycoords='data',
                 xytext=(Av_x+dx, Av_y+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color='black',
                                 lw=2,
                                 ls='-')
               )

    ax.annotate(f'Av = {Av_}', xy=(Av_x-0.1, Av_y-0.1), fontsize=25)
        
    fig.tight_layout()
    return fig, ax, tab
                     
def gen_CMD_xcut(tab, filt1='f115w', filt2='f150w', filt3=None, ra_col = 'ra_1', dec_col= 'dec_1',
                 ra_cen=0, dec_cen=0, r_in=0, r_out=24, sqr_field=False, Av=0.19,
                 mag_err_cols = ['mag_err_1','mag_err_2'],  dismod=29.95, mag_err_lim=0.2,label_min=0, 
                 label_max=10, cmd=None,  Av_=3,  Av_x=2, Av_y=22,  xlims=[-0.5,2.5], ylims=[18,30], 
                 ang=245.00492 , age=9.0,met=0.02,  fit_slope=False, cmd_ylo=None, cmd_yhi=None, cmd_xlo = None, 
                 cmd_xhi= None, y_lo = 22, y_hi=26.5, dy=0.5, dx=0.5, rgb_xlo=0.5,rgb_xhi=2,
                 rgb_ylo=23, rgb_yhi=26, fit_isochrone=True, fig=None, ax=None,s=5,lw=3):
        
    if filt3 is None:
        filt3 = filt2
        
    if met is not None:
        cmd = cmd[cmd['Zini']==met]
        
    if cmd_ylo is None or cmd_yhi is None:
        cmd_ylo = y_lo - dy
        cmd_yhi = y_hi + dy
    
    AF1 =  Av_dict[filt1]*Av
    AF2 =  Av_dict[filt2]*Av
    AF3 =  Av_dict[filt3]*Av
    
    tab['r'] = angular_separation(tab[ra_col]*u.deg,tab[dec_col]*u.deg,
                                          ra_cen*u.deg, dec_cen*u.deg).to(u.arcsec).value
    if r_in is None:
            r_in = 0
            r_out = r_out
            
    if not sqr_field:
        tab = tab[ (tab['r']>=r_in) & (tab['r'] <=r_out)]
    else:
        tab = box(tab, ra_col, dec_col,  ra_cen, dec_cen,
                      r_out/3600, r_out/3600, ang)
    
    x = tab[f'mag_vega_{filt1.upper()}'] - tab[f'mag_vega_{filt2.upper()}']
    y = tab[f'mag_vega_{filt3.upper()}'] 

    x = x.value.astype(float)
    y = y.value.astype(float)

    if cmd_xlo is None or cmd_xhi is None:
        cmd_xlo = x.mean() - 0.5

        cmd_xhi = x.mean() + 0.5
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12,10))
    
    theta = np.arctan(AF3/(AF1-AF2)) # Extinction vector
    
    ax.scatter(x,y, s=s, color='black')
    
    if cmd is not None:
        cmd = cmd[(cmd['label']>=label_min) & (cmd['label']<=label_max)]

        t = cmd[np.round(cmd['logAge'],1)==age].copy()

        met = np.array(cmd['Zini'])[0]
        x_i =  (t[f'{filt1.upper()}mag'] + AF1) - (t[f'{filt2.upper()}mag'] + AF2)
        y_i =  t[f'{filt3.upper()}mag'] + AF3 + dismod
        
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        
        x_iso = x_i.copy()
        y_iso = y_i.copy()
        
        x_i = x_i[np.where( (y_i>=cmd_ylo) & (y_i<=cmd_yhi))[0]]
        y_i = y_i[np.where( (y_i>=cmd_ylo) & (y_i<=cmd_yhi))[0]]
        
        ax.plot(x_iso,y_iso, zorder=300,color='green',lw=lw)

        ax.legend(['data', f'age = {age}'], fontsize=20)

    x_l = np.linspace(cmd_xlo, cmd_xhi)    

    # Bin mid points
    y_rgbn = np.arange(y_lo, y_hi, dy)
    
    y_rgb_mid = y_rgbn[:-1] + dy/2   
    
    if fit_isochrone:
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_i, x_i)
        
        ax.plot(model_iso(np.linspace(ylims[0],ylims[1])),
                          np.linspace(ylims[0],ylims[1]),'--r', lw=lw,
               zorder=400)
        if fit_slope:
            theta = np.arctan(-model_iso.slope.value)
        
    else:
        ind = (x>=rgb_xlo) & (x<=rgb_xhi) & (y>=rgb_ylo) & (y<=rgb_yhi)

        y_n, x_n = running_avg(y[ind], x[ind], 100)
        
        ind = ~np.isnan(x_n)
        x_bin = x_n[ind]
        y_bin = y_n[ind]
        
        ax.plot(x_bin,y_bin,color='blue',lw=lw)
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_bin, x_bin)
        
        ax.plot(model_iso(np.linspace(ylims[0],ylims[1])),
                          np.linspace(ylims[0],ylims[1]),'--r',lw=lw, 
               zorder=400)
        
    x_rgb_mid = model_iso(y_rgb_mid)
    
    ax.scatter(x_rgb_mid, y_rgb_mid, c='r' ,zorder = 200,s=s)

    dats = []
    
    dx0 = dx
    dx = cmd_xhi- cmd_xlo
    
    for i,y0 in enumerate(y_rgbn[:-1]):

        # Extinction Vector
        x0 = model_iso(y0)
        x_l = np.linspace(x0-dx0/2, x0+dx0/2)   
        y_Avl = y0 + np.tan(theta)*(x_l-x0)
        
        x01 = model_iso(y0 + dy)
        y_Avu = y0 + dy + np.tan(theta)*(x_l-x01)

        ax.plot(x_l,y_Avl, color='grey',lw=lw)
        ax.plot(x_l,y_Avu, color='grey',lw=lw)

        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        
        model_Avl = fit(init, x_l, y_Avl)
        model_Avu = fit(init, x_l, y_Avu)

        c1 = (y>model_Avl(x)) & (y<=model_Avu(x))
        c2 = (x>=x_rgb_mid[i]-dx/2) & (x<=x_rgb_mid[i]+dx/2) 

        yn = y[np.where(c1&c2)]
        xn = x[np.where(c1&c2)]
        
        dat = np.array([xn, yn])
        dats.append(dat)
    
    ax.set_xlabel(f"{filt1.upper()} - {filt2.upper()}")
    ax.set_ylabel(filt3.upper())

    ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlim(xlims[0], xlims[1])  
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', length=15,direction="in", bottom=True, top=True,left=True, right=True, width = 3)
    ax.tick_params(which='minor', length=8, width = 3)
    
    AF1_ =  Av_dict[filt1]*Av_
    AF2_ =  Av_dict[filt2]*Av_
    AF3_ =  Av_dict[filt3]*Av_
    
    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(Av_x, Av_y),
                 xycoords='data',
                 xytext=(Av_x+dx, Av_y+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color='black',
                                 lw=lw,
                                 ls='-')
               )

    ax.annotate(f'Av = {Av_}', xy=(Av_x-0.1, Av_y-0.1), fontsize=25)
    
    return fig, ax, dats, x_rgb_mid, y_rgb_mid, y_rgbn, model_iso

def gen_CMD_ycut(tab, filt1='f115w', filt2='f150w', filt3=None, ra_col = 'ra_1', dec_col= 'dec_1',
                 ra_cen=0, dec_cen=0, Av=0.19, r=333, r_in=None, r_out=None,  met=0.02, mag_err_lim=0.2,
                 mag_err_cols = ['mag_err_1','mag_err_2'], label_min=0, label_max=3,
                dismod=29.95, cmd=None, xlims=[-0.5,2.5], ylims=[18,30], sqr_field=False,
                age=9.0, cmd_xlo = None, cmd_xhi= None, gen_kde=False, perp_iso=False,
                y_lo = 22, y_hi=26.5, dy=0.5, Av_ = 3,ref_xpos=0.25, rgb_xlo=0.5,rgb_xhi=2,
                rgb_ylo=23, rgb_yhi=26, Av_x=2, Av_y=22, fit_isochrone=True,
                x0=1, y0=None,ang=245.00492, fig = None, ax = None,s=10,lw=1):
    
    if filt3 is None:
        filt3 = filt2
    if met is not None and cmd is not None:
        if 'Zini' in cmd.keys():
            cmd = cmd[cmd['Zini']==met]
        else:
            cmd = cmd[cmd['Zini_1']==met]  
            
    cmd_ylo = y_lo
    cmd_yhi = y_hi
    
    age_lin = []
    for i in [age]:
        if i<6:
            age_lin.append(f'{np.ceil(10**i)} Myr')
        if i >= 6  and i <9:
            i-=6
            age_lin.append(f'{np.ceil(10**i)} Myr')
        elif i >= 9:
            i-=9
            age_lin.append(f'{np.ceil(10**i)} Gyr')
        
    for i in mag_err_cols:
        tab = tab[tab[i]<=mag_err_lim]

    AF1 =  Av_dict[filt1]*Av
    AF2 =  Av_dict[filt2]*Av
    AF3 =  Av_dict[filt3]*Av

    tab['r'] = angular_separation(tab[ra_col]*u.deg,tab[dec_col]*u.deg,
                                          ra_cen*u.deg, dec_cen*u.deg).to(u.arcsec).value

    if r_in is None :
            r_in = 0
            
    if r_out is not None:  
        if not sqr_field:
            tab = tab[ (tab['r']>=r_in) & (tab['r'] <=r_out)]
        else:
            tab = box(tab, ra_col, dec_col,  ra_cen, dec_cen,
                          r_out/3600, r_out/3600, ang)
    
    x = tab[f'mag_vega_{filt1.upper()}'] - tab[f'mag_vega_{filt2.upper()}']
    y = tab[f'mag_vega_{filt3.upper()}'] 
    
    x = x.value.astype(float)
    y = y.value.astype(float)
    
    if cmd_xlo is None or cmd_xhi is None:
        cmd_xlo = x.mean() - 1

        cmd_xhi = x.mean() + 1
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12,10))

    l = []
    
    if gen_kde:

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = gaussian_kde(values, bw_method=0.05)
        f = np.reshape(kernel(positions), xx.shape)

        f = f.T
        img = ax.imshow(f, cmap='jet', 
                      extent=[xlims[0], xlims[1], 
                              ylims[0], ylims[1]],
                       interpolation='nearest', aspect='auto')
            
    else:
        ax.scatter(x,y, s=s, color='black')
        l.append('data')
    M  = None
    c_ = None
    if cmd is not None:
        l_ = [i for i in cmd.keys() if 'logAge' in i][0]
        t = cmd[np.round(cmd[l_],1)==age].copy()
        t = t[(t['label']>=label_min) & (t['label']<=label_max)]
        x_i =  (t[f'{filt1.upper()}mag'] + AF1) - (t[f'{filt2.upper()}mag'] + AF2)
        y_i =  t[f'{filt3.upper()}mag']
        # Max mag and Color
        M = y_i.min()
        c_ = x_i[y_i==y_i.min()].values[0]
        
        y_i +=   AF3 + dismod
        
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        
        x_iso = x_i.copy()
        y_iso = y_i.copy()
        

        x_i = x_i[np.where( (y_i>=cmd_ylo-0.5) & (y_i<=cmd_yhi+0.5))[0]]
        y_i = y_i[np.where( (y_i>=cmd_ylo-0.5) & (y_i<=cmd_yhi+0.5))[0]]

        y_i = y_i[np.where( (x_i>=cmd_xlo-0.2) & (x_i<=cmd_xhi+0.2))[0]]
        x_i = x_i[np.where( (x_i>=cmd_xlo-0.2) & (x_i<=cmd_xhi+0.2))[0]]
        
        ax.plot(x_iso,y_iso, zorder=200, color='black',lw=lw)
        
        l.append(f'Age = {age_lin[0]}')

    x_l = np.linspace(0, 2)    

    # Bin mid points
    x_rgbn = np.arange(cmd_xlo, cmd_xhi, dy)
    
    x_rgb_mid = x_rgbn[:-1]+dy/2   
    
    if fit_isochrone:
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, x_i, y_i)
        slope = 1/model_iso.slope.value
    
    elif perp_iso:
        slope=0
        
    else:
        ind = (x>=rgb_xlo) & (x<=rgb_xhi) & (y>=rgb_ylo) & (y<=rgb_yhi)

        y_n, x_n = running_avg(y[ind], x[ind], 100)
        
        ind = ~np.isnan(x_n)
        x_bin = x_n[ind]
        y_bin = y_n[ind]
        
        ax.plot(x_bin,y_bin, color='blue', zorder=390)
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_bin, x_bin)
        
        ax.plot(model_iso(np.linspace(ylims[0],ylims[1])),
                          np.linspace(ylims[0],ylims[1]),'--r', lw=lw,
               zorder=400)
        
        slope = model_iso.slope.value
    
    if y0 is None:
        y0 = y.mean()
    y_rgb_mid = y0 + x_rgb_mid*0

    dats = []
    
    init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    for x0 in x_rgbn[:-1]:
        
        y_l = np.linspace(cmd_ylo, cmd_yhi)
        x_l = slope*(y_l - y0) + x0
        
        y_r = np.linspace(cmd_ylo, cmd_yhi)
        x_r = slope*(y_r - y0) + x0 + dy

        ax.plot(x_l,y_l, color='red', lw=lw)
        ax.plot(x_r,y_r, color='red', lw=lw)

        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        
        model_l = fit(init, y_l,x_l)
        model_r = fit(init, y_r,x_r)

        c1 = (x>model_l(y)) & (x<=model_r(y))

        yn = y[np.where(c1)]
        xn = x[np.where(c1)]
        if not gen_kde:
            ax.scatter(xn,yn, s =s, color='green', zorder=100)
        dat = np.array([xn, yn])
        dats.append(dat)
    
    ref = tab[f'mag_vega_{filt2.upper()}']
    ref_new = np.arange(np.ceil(y.min()),np.ceil(y.max()) + 0.5,0.5)

    mag_err1 = tab[mag_err_cols[0]]
    mag_err2 = tab[mag_err_cols[1]]

    if len(mag_err_cols)>2:
        mag_err3 = tab[mag_err_cols[2]]
    else:
        mag_err3 = mag_err2

    col_err = np.sqrt(mag_err1**2 + mag_err2**2)

    init = models.Exponential1D()
    fit = fitting.LevMarLSQFitter()
    model_col = fit(init,ref,col_err)

    init = models.Exponential1D()
    fit = fitting.LevMarLSQFitter()
    model_mag = fit(init,ref,mag_err3)

    x = ref_xpos + 0*ref_new
    y = ref_new
    yerr = model_mag(ref_new)
    xerr = model_col(ref_new)

    ax.errorbar(x, y, yerr,xerr ,fmt='o', color = 'red', markersize=0.5, capsize=2) 
    
    AF1_ =  Av_dict[filt1]*Av_
    AF2_ =  Av_dict[filt2]*Av_
    AF3_ =  Av_dict[filt3]*Av_
    
    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(Av_x, Av_y),
                 xycoords='data',
                 xytext=(Av_x+dx, Av_y+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color='black',
                                 lw=lw,
                                 ls='-')
               )

    ax.annotate(f'Av = {Av_}', xy=(Av_x-0.1, Av_y-0.1), fontsize=25)
    
    ax.set_xlabel(f"{filt1.upper()} - {filt2.upper()}")
    ax.set_ylabel(filt3.upper())

    ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlim(xlims[0], xlims[1])  
    ax.invert_yaxis()
    title = f"Z : {met} | " + "$M_" + "{" + f"{filt3.upper()}" +r"}^{TRGB}$ : " + f"{M} | "
    title += r"$A_{" + f"{filt3.upper()}" + r"}$ : " + f"{np.round(AF3,3)}"
    ax.set_title(title)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', length=15,direction="in", bottom=True, top=True,left=True, right=True, width=3)
    ax.tick_params(which='minor', length=8, width = 3)
    ax.legend(l,fontsize=20)
    
    return fig, ax, dats, x_rgb_mid, y_rgb_mid, x_rgbn, [M, AF3, c_]
