import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import angular_separation
import astropy.units as u
from scipy.stats import gaussian_kde
from astropy.modeling import models, fitting
import seaborn as sb
from .catalog_filter import box, ellipse, polygon
from matplotlib.colors import LinearSegmentedColormap
import subprocess
import pandas as pd

sb.set_style('white')

from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize'] = 35
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 35
  

Av_dict = { # WFC3
            'f275w': 2.02499,
            'f336w': 1.67536,
            'f438w': 1.34148,
            'f606w': 0.90941,
            'uf814w': 0.59845,
            
            # JWST-NIRCAM
            'f090w': 0.583,
            'f115w': 0.419,
            'f140m': 0.315,
            'f150w': 0.287,
            'f200w': 0.195,
            'f212n': 0.176,
            'f356w': 0.099,
            'f444w': 0.083,
            # WFC2
            'f435w': 1.33879,
            'f555w': 1.03065,
            'f814w': 0.59696,
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

def gen_CMD(
    tab,
    df_iso = None,
    filters={'filt1': 'f115w', 'filt2': 'f150w'},
    positions={'ra_col': 'ra', 'dec_col': 'dec', 'ra_cen': 0, 'dec_cen': 0},
    region={'r_in': 0, 'r_out': 24, 'spatial_filter': 'circle','ang': 245.00492},
    extinction={'Av': 0.19, 'Av_': 3, 'Av_x': 2, 'Av_y': 19},
    distance_modulus=29.7415,
    axis_limits={'xlims': [-0.5, 2.5], 'ylims': [18, 28]},
    isochrone_params={'met': 0.02, 'ages': [7., 8., 9.]},
    plot_settings={'alpha': 1, 's': 0.2, 'lw': 3},
    error_settings={ 'mag_err_lim': 0.2, 'show_err_model': False, 'ref_xpos': -0.5},
    kde_contours={'gen_kde': False, 'gen_contours': False},
    other_settings={'ab_dist': True, 'skip_data': False, 'show_err_model':False},
    fig=None,
    ax=None):
    
    """
    Generate a Color-Magnitude Diagram (CMD) with optional KDE or contour overlays.

    Parameters
    ----------
    tab : DataFrame
        Input data table containing magnitudes, positions, and errors for sources.
        
    df_iso : DataFrame, optional
        Isochrone data for overlay.

    filters : dict, optional
        Filters used in CMD. Keys:
        - 'filt1': Primary filter for color calculation.
        - 'filt2': Secondary filter for color calculation.
        - 'filt3': Filter for magnitude axis. Defaults to 'filt2'.

    positions : dict, optional
        Positional parameters. Keys:
        - 'ra_col': RA column name.
        - 'dec_col': DEC column name.
        - 'ra_cen': Central RA (in degrees).
        - 'dec_cen': Central DEC (in degrees).

    region : dict, optional
        Region parameters for filtering sources. Keys:
        - 'r_in': Inner radius for selection (arcseconds) (used for circular filters).
        - 'r_out': Outer radius for selection (arcseconds) (used for circular filters).
        - 'spatial_filter': Type of spatial filtering ('circle', 'box', 'ellipse').
        - 'ang': Orientation angle (used for box or ellipse filters).
        - 'width_in', 'height_in': Inner box dimensions (arcseconds) (used for box filters).
        - 'width_out', 'height_out': Outer box dimensions (arcseconds) (used for box filters).
        - 'a1', 'b1': Inner semi-major and semi-minor axes (arcseconds) (used for ellipse filters).
        - 'a2', 'b2': Outer semi-major and semi-minor axes (arcseconds) (used for ellipse filters).

    extinction : dict, optional
        Extinction parameters. Keys:
        - 'Av': Extinction value.
        - 'Av_': Annotation extinction value.
        - 'Av_x', 'Av_y': Annotation arrow position.

    distance_modulus : float, optional
        Distance modulus for CMD adjustments. Default is 29.7415.

    axis_limits : dict, optional
        Plot axis limits. Keys:
        - 'xlims': Limits for x-axis (color).
        - 'ylims': Limits for y-axis (magnitude).

    isochrone_params : dict, optional
        Isochrone parameters for plotting. Keys:
        - 'met': Metallicity for isochrones.
        - 'label_min': Minimum label value for filtering.
        - 'label_max': Maximum label value for filtering.
        - 'ages': List of log ages to plot.

    plot_settings : dict, optional
        General plot settings. Keys:
        - 'alpha': Transparency for isochrone lines.
        - 's': Marker size for scatter plots.
        - 'lw': Line width for isochrones.

    error_settings : dict, optional
        Settings for error handling and plotting. Keys:
        - 'mag_err_cols': Columns for magnitude errors. Defaults to filter-based columns.
        - 'mag_err_lim': Maximum allowable magnitude error.
        - 'show_err_model': Show error models during plotting.
        - 'ref_xpos': Reference x-position for error bars.

    kde_contours : dict, optional
        Settings for KDE or contour plots. Keys:
        - 'gen_kde': Generate KDE overlay.
        - 'gen_contours': Generate contour overlay.

    other_settings : dict, optional
        Miscellaneous settings. Keys:
        - 'ab_dist': Include absolute distance modulus adjustments.
        - 'skip_data': Skip scatter plot of source data.

    fig : matplotlib.figure.Figure, optional
        Existing figure object. If None, a new figure is created.

    ax : matplotlib.axes.Axes, optional
        Existing axis object. If None, a new axis is created.


    Returns
    -------
    tuple
        (fig, ax, tab) where:
        - fig: The figure object.
        - ax: The axis object.
        - tab: The filtered input data table after spatial and error-based selection.
    """
    
    # Fill in default values for nested dictionaries
    filters.setdefault('filt1','f115w')
    filters.setdefault('filt2','f200w')
    filters.setdefault('filt3', filters['filt2'])
    
    positions.setdefault('ra_col','ra')
    positions.setdefault('dec_col','dec')
    positions.setdefault('ra_cen',0)
    positions.setdefault('dec_cen',0)
    
    region.setdefault('r_in',0)
    region.setdefault('r_out',10)
    region.setdefault('spatial_filter','circle')
    
    extinction.setdefault('Av',0.19)
    extinction.setdefault('Av_',3)
    extinction.setdefault('Av_x',3)
    extinction.setdefault('Av_y',19)
    
    axis_limits.setdefault('xlims', [-0.5, 2.5])
    axis_limits.setdefault('ylims', [18, 28])
    
    isochrone_params.setdefault('label_min', 0)
    isochrone_params.setdefault('label_max', 10)
    isochrone_params.setdefault('met', [0.02])
    isochrone_params.setdefault('age', [7,8,9])
    
    plot_settings.setdefault('Av.fontsize',15)
    plot_settings.setdefault('legend.fontsize',15)
    plot_settings.setdefault('lw',3)
    plot_settings.setdefault('s',0.2)
    plot_settings.setdefault('alpha',1)
    plot_settings.setdefault('print_met',False)
    plot_settings.setdefault('legend.ncols',1)
    
    
    error_settings.setdefault('mag_err_cols', [
        f'mag_err_{filters["filt1"].upper()}',
        f'mag_err_{filters["filt2"].upper()}',
        f'mag_err_{filters["filt3"].upper()}',])
    
    error_settings.setdefault('mag_err_lim',0.2)
    error_settings.setdefault('ref_xpos',-0.25)
    
    kde_contours.setdefault('gen_kde',False)
    kde_contours.setdefault('gen_contours',False)
    kde_contours.setdefault('kde_bin',100j)
    kde_contours.setdefault('cmap','jet')
    kde_contours.setdefault('bw', 0.05)
    
    other_settings.setdefault('ab_dist',True)
    other_settings.setdefault('skip_data',False)
    other_settings.setdefault('show_err_model',False)

    # Filter table by magnitude errors
    for col in error_settings['mag_err_cols']:
        tab = tab[tab[col] <= error_settings['mag_err_lim']]

    # Compute angular separation or define square field
    tab['r'] = angular_separation(
        tab[positions['ra_col']] * u.deg,
        tab[positions['dec_col']] * u.deg,
        positions['ra_cen'] * u.deg,
        positions['dec_cen'] * u.deg).to(u.arcsec).value

    if region['spatial_filter']=='circle':
        tab = tab[(tab['r'] >= region['r_in'])
                  & (tab['r'] <= region['r_out'])]
        
    elif region['spatial_filter']=='box':   
        tab = box(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['width_in'] / 3600, region['height_in'] / 3600,
                  region['width_out'] / 3600, region['height_out'] / 3600,
                  region['ang'])

    elif region['spatial_filter']=='ellipse':
        tab = ellipse(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['ang'], 
                  region['a1'] / 3600,region['b1'] / 3600,
                  region['a2'] / 3600,region['b2'] / 3600)
    elif region['spatial_filter']=='polygon':
        tab = polygon(tab, positions['ra_col'], positions['dec_col'], region['points'])

    # Compute magnitudes and colors
    x = tab[f'mag_vega_{filters["filt1"].upper()}'] - tab[f'mag_vega_{filters["filt2"].upper()}']
    y = tab[f'mag_vega_{filters["filt3"].upper()}']

    x = x.value.astype(float)
    y = y.value.astype(float)

    # Initialize figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Extinction corrections
    AF1 = Av_dict[filters['filt1']] * extinction['Av']
    AF2 = Av_dict[filters['filt2']] * extinction['Av']
    AF3 = Av_dict[filters['filt3']] * extinction['Av']

    # Kernel density estimation or scatter plot
    tick_color = 'black'
    if kde_contours['gen_kde'] and not kde_contours['gen_contours']:
        xx, yy = np.mgrid[
            axis_limits['xlims'][0]:axis_limits['xlims'][1]:kde_contours['kde_bin'],
            axis_limits['ylims'][0]:axis_limits['ylims'][1]:kde_contours['kde_bin']]
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = gaussian_kde(values, bw_method=kde_contours['bw'])
        f = np.reshape(kernel(positions), xx.shape)
        tick_color='white'
        ax.imshow(f.T, cmap=kde_contours['cmap'], extent=(*axis_limits['xlims'], *axis_limits['ylims']),
                  interpolation='nearest', aspect='auto')

    elif kde_contours['gen_contours']:
        ax.scatter(x, y, s=plot_settings['s'], color='black', label='data')
        cmap_custom = LinearSegmentedColormap.from_list("custom_grey_to_white", ["grey", "white"], N=256)
        sb.kdeplot(x=x, y=y, levels=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                   ax=ax, fill=True, cmap=cmap_custom)
        
        sb.kdeplot(x=x, y=y, levels=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                   ax=ax, color='black')

    elif not other_settings['skip_data']:
        ax.scatter(x, y, s=plot_settings['s'], color='black', label='data')
        
    ax.set_xlim(axis_limits['xlims'][0],axis_limits['xlims'][1])
    ax.set_ylim(axis_limits['ylims'][0],axis_limits['ylims'][1])
    
    ax.tick_params(which='both', length=15,direction="in", 
                   bottom=True, top=True,left=True, width = 3,
                   color=tick_color)
    
    ax.tick_params(which='minor', length=8, width = 3, color=tick_color)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Handle isochrones
        
    age_lin = []
    for i in isochrone_params['ages']:
        if i < 6:
            age_lin.append(f'{np.round(10**i,1)} Myr')
        if i >= 6  and i < 9:
            i -= 6
            age_lin.append(f'{np.round(10**i,1)} Myr')
        elif i >= 9:
            i -= 9
            age_lin.append(f'{np.round(10**i,1)} Gyr')
            
    if df_iso is not None:
        df_iso = df_iso[(df_iso['label']>=isochrone_params['label_min'])
                       & (df_iso['label']<=isochrone_params['label_max'])]
                        
        for i,age in enumerate(isochrone_params['ages']):
            t = df_iso[(np.round(df_iso['logAge'],1) == age)]
            for Z in isochrone_params['met']:
                subset = t[t['Zini'] == Z]
                x_iso = subset[f"{filters['filt1'].upper()}mag"] + AF1 - (
                        subset[f"{filters['filt2'].upper()}mag"] + AF2)
                y_iso = subset[f"{filters['filt3'].upper()}mag"] + AF3 + distance_modulus
                       
                mask = (y_iso.values[1:]- y_iso.values[:-1])<1
                mask = np.array([True] + list(mask))
                mask = np.where(~mask, np.nan, 1)
                
                if len(isochrone_params['met'])>1 or plot_settings['print_met']:
                    label = label=age_lin[i]+ f' {Z}'
                else:
                    label = label=age_lin[i]
                               
                ax.plot(x_iso*mask, y_iso*mask, lw=plot_settings['lw'],
                        label=label,alpha=plot_settings['alpha'])

    # Absolute magnitude
    if other_settings['ab_dist']:
        yticks = ax.get_yticks()
        yticks_n = yticks - distance_modulus - AF3
        
        dy = yticks_n - np.floor(yticks_n)
        ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis            
        ax1.set_ylabel(r'$M_{' + f"{filters['filt3'].upper()}" + r'}$')  # we already handled the x-label with ax1
        ax1.set_yticks(yticks - dy, np.floor(yticks_n), fontsize=30)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax1.set_xlim(axis_limits['xlims'][0],axis_limits['xlims'][1])
        ax1.set_ylim(axis_limits['ylims'][0],axis_limits['ylims'][1])
        
        ax1.tick_params(which='both', length=15,direction="in",
                        right=True, width = 3, color=tick_color)
        ax1.tick_params(which='minor', length=8, width = 3, color=tick_color)
        
        
        ax1.invert_yaxis()

    # Extinction Vector
    AF1_ = Av_dict[filters['filt1']] * extinction['Av_']
    AF2_ = Av_dict[filters['filt2']] * extinction['Av_']
    AF3_ = Av_dict[filters['filt3']] * extinction['Av_']

    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(extinction['Av_x'], extinction['Av_y']),
                 xycoords='data',
                 xytext=(extinction['Av_x']+dx, 
                         extinction['Av_y']+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color=tick_color,
                                 lw=plot_settings['lw'],
                                 ls='-')
               )

    ax.annotate(f"Av = {extinction['Av_']}",
                xy=(extinction['Av_x']-0.1, extinction['Av_y']-0.1)
                ,fontsize=plot_settings['Av.fontsize'],
                color=tick_color)
    
    # Error models
    if not other_settings['skip_data']:
        ref = tab[f"mag_vega_{filters['filt3'].upper()}"]
        ref_new = np.arange(np.ceil(y.min()),np.floor(y.max())+0.5,0.5)

        mag_err1 = tab[error_settings['mag_err_cols'][0]]
        mag_err2 = tab[error_settings['mag_err_cols'][1]]

        if len(error_settings['mag_err_cols'])>2:
            mag_err3 = tab[error_settings['mag_err_cols'][2]]
        else:
            mag_err3 = mag_err2

        col_err = np.sqrt(mag_err1**2 + mag_err2**2)
        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_col = fit(init,ref,col_err)

        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_mag = fit(init,ref,mag_err3)

        x = error_settings['ref_xpos'] + 0*ref_new
        y = ref_new
        yerr = model_mag(ref_new)
        xerr = model_col(ref_new)
        
        if other_settings['show_err_model']:
            plt.show()
            plt.scatter(ref, mag_err3)
            plt.plot(ref_new,yerr,'--r')
            plt.show()
            plt.scatter(ref, col_err)
            plt.plot(ref_new,xerr,'--r')
            plt.show()
        ax.errorbar(x, y, yerr, xerr ,fmt='o', color = 'red', markersize=0.5, capsize=2) 
    
    # Labels, ticks, and legend
    ax.invert_yaxis()
    ax.set_xlabel(f"{filters['filt1'].upper()} - {filters['filt2'].upper()}")
    ax.set_ylabel(filters['filt3'].upper())
    ax.legend(fontsize=plot_settings['legend.fontsize'], ncols = plot_settings['legend.ncols'])

    fig.tight_layout()
    return fig, ax, tab
                     
def gen_CMD_xcut(tab,
                df_iso = None,
                filters={'filt1': 'f115w', 'filt2': 'f150w'},
                positions={'ra_col': 'ra', 'dec_col': 'dec', 'ra_cen': 0, 'dec_cen': 0},
                region={'r_in': 0, 'r_out': 24, 'spatial_filter': 'circle','ang': 245.00492},
                extinction={'Av': 0.19, 'Av_': 3, 'Av_x': 2, 'Av_y': 19},
                distance_modulus=29.7415,
                axis_limits={'xlims': [-0.5, 2.5], 'ylims': [18, 28]},
                isochrone_params={'met': 0.02, 'ages': [7., 8., 9.]},
                plot_settings={'alpha': 1, 's': 0.2, 'lw': 3, 'cmap':'jet'},
                error_settings={ 'mag_err_lim': 0.2, 'show_err_model': False, 'ref_xpos': -0.5},
                kde_contours={'gen_kde': False, 'gen_contours': False, 'kde_bin': 200j},
                other_settings={'ab_dist': True, 'skip_data': False, 'show_err_model':False},
                x_cut_settings= {},                  
                fig=None,
                ax=None):
    
    filters.setdefault('filt1','f115w')
    filters.setdefault('filt2','f200w')
    filters.setdefault('filt3', filters['filt2'])
    
    positions.setdefault('ra_col','ra')
    positions.setdefault('dec_col','dec')
    positions.setdefault('ra_cen',0)
    positions.setdefault('dec_cen',0)
    
    region.setdefault('r_in',0)
    region.setdefault('r_out',10)
    region.setdefault('spatial_filter','circle')
    
    extinction.setdefault('Av',0.19)
    extinction.setdefault('Av_',3)
    extinction.setdefault('Av_x',3)
    extinction.setdefault('Av_y',19)
    
    axis_limits.setdefault('xlims', [-0.5, 2.5])
    axis_limits.setdefault('ylims', [18, 28])
    
    isochrone_params.setdefault('label_min', 0)
    isochrone_params.setdefault('label_max', 10)
    isochrone_params.setdefault('met', 0.002)
    isochrone_params.setdefault('age', 10)
    
    plot_settings.setdefault('Av.fontsize',15)
    plot_settings.setdefault('legend.fontsize',15)
    plot_settings.setdefault('lw',3)
    plot_settings.setdefault('s',0.2)
    plot_settings.setdefault('alpha',1)
    plot_settings.setdefault('print_met',False)
    plot_settings.setdefault('legend.ncols',1)
    plot_settings.setdefault('cmap','jet')
    
    
    error_settings.setdefault('mag_err_cols', [
        f'mag_err_{filters["filt1"].upper()}',
        f'mag_err_{filters["filt2"].upper()}',
        f'mag_err_{filters["filt3"].upper()}',])
    
    error_settings.setdefault('mag_err_lim',0.2)
    error_settings.setdefault('ref_xpos',-0.25)
    
    kde_contours.setdefault('gen_kde',False)
    kde_contours.setdefault('gen_contours',False)
    kde_contours.setdefault('kde_bin',100j)
    
    other_settings.setdefault('ab_dist',True)
    other_settings.setdefault('skip_data',False)
    other_settings.setdefault('show_err_model',False)

    x_cut_settings.setdefault('cmd_xlo', None)
    x_cut_settings.setdefault('cmd_xhi', None)
    x_cut_settings.setdefault('cmd_ylo', None)
    x_cut_settings.setdefault('cmd_yhi', None)
    x_cut_settings.setdefault('perp_iso', False)
    x_cut_settings.setdefault('y_lo', 22)
    x_cut_settings.setdefault('y_hi', 26.5)
    x_cut_settings.setdefault('dy', 0.5)
    x_cut_settings.setdefault('dx', 0.5)
    x_cut_settings.setdefault('rgb_xlo', 0.5)
    x_cut_settings.setdefault('rgb_xhi', 2)
    x_cut_settings.setdefault('rgb_ylo', 23)
    x_cut_settings.setdefault('rgb_yhi', 26)
    x_cut_settings.setdefault('fit_isochrone', True)
    x_cut_settings.setdefault('fit_rgb', False)
    x_cut_settings.setdefault('x0', 1)
    x_cut_settings.setdefault('y0', None)
    x_cut_settings.setdefault('ref_dy', 0.5)
    x_cut_settings.setdefault('rgb_fit_bin', 100)
    x_cut_settings.setdefault('theta', None)
    x_cut_settings.setdefault('slope', None)
    x_cut_settings.setdefault('intercept', None)
    x_cut_settings.setdefault('color', 'grey')

    # Filter table by magnitude errors
    for col in error_settings['mag_err_cols']:
        tab = tab[tab[col] <= error_settings['mag_err_lim']]
        
    df_iso = df_iso[df_iso['Zini']==isochrone_params['met']]
    df_iso = df_iso[np.round(df_iso['logAge'],1)==isochrone_params['age']]


    if x_cut_settings['cmd_ylo'] is None or x_cut_settings['cmd_yhi'] is None:
        x_cut_settings['cmd_ylo'] = x_cut_settings['y_lo'] - x_cut_settings['dy']
        x_cut_settings['cmd_yhi'] = x_cut_settings['y_hi'] + x_cut_settings['dy']
    
    age = isochrone_params['age'] 
    if age <6:
        age_lin  = f'{np.ceil(10**age)} Myr'
    if age  >= 6  and age < 9:
        age -=6
        age_lin = f'{np.ceil(10**age)} Myr'
    elif age >= 9:
        age-=9
        age_lin = f'{np.ceil(10**age)} Gyr'

    if region['spatial_filter']=='circle':
        tab['r'] = angular_separation(
        tab[positions['ra_col']] * u.deg,
        tab[positions['dec_col']] * u.deg,
        positions['ra_cen'] * u.deg,
        positions['dec_cen'] * u.deg).to(u.arcsec).value
        
        tab = tab[(tab['r'] >= region['r_in'])
                  & (tab['r'] <= region['r_out'])]
        
    elif region['spatial_filter']=='box':   
        tab = box(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['width_in'] / 3600, region['height_in'] / 3600,
                  region['width_out'] / 3600, region['height_out'] / 3600,
                  region['ang'])

    elif region['spatial_filter']=='ellipse':
        tab = ellipse(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['ang'], 
                  region['a1'] / 3600,region['b1'] / 3600,
                  region['a2'] / 3600,region['b2'] / 3600)


    # Extinction corrections
    AF1 = Av_dict[filters['filt1']] * extinction['Av']
    AF2 = Av_dict[filters['filt2']] * extinction['Av']
    AF3 = Av_dict[filters['filt3']] * extinction['Av']
    
        # Compute magnitudes and colors
    x = tab[f'mag_vega_{filters["filt1"].upper()}'] - tab[f'mag_vega_{filters["filt2"].upper()}']
    y = tab[f'mag_vega_{filters["filt3"].upper()}']

    x = x.value.astype(float)
    y = y.value.astype(float)

    if x_cut_settings['cmd_xlo'] is None or x_cut_settings['cmd_xhi'] is None:
        x_cut_settings['cmd_xlo'] = x.mean() - 0.5

        x_cut_settings['cmd_xhi'] = x.mean() + 0.5
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12,10))
    # Extinction vector
    legends = []
    ax.scatter(x,y, s=plot_settings['s'], color='black')
    
    if df_iso is not None:
        df_iso = df_iso[(df_iso['label']>=isochrone_params['label_min']) & 
                        (df_iso['label']<=isochrone_params['label_max'])]
        
        x_i =  (df_iso[f"{filters['filt1'].upper()}mag"] + AF1) - (df_iso[f"{filters['filt2'].upper()}mag"] + AF2)
        y_i =  df_iso[f"{filters['filt3'].upper()}mag"]
        
        # Max mag and Color
        trgb_mag_iso = y_i.min()
        trgb_col_iso = x_i[y_i==y_i.min()].values[0]
        
        y_i +=  AF3 + distance_modulus
        
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        
        x_iso = x_i.copy()
        y_iso = y_i.copy()

        ind = ((y_i>=x_cut_settings['cmd_ylo']) & 
              (y_i<=x_cut_settings['cmd_yhi']) &
              (x_i>=x_cut_settings['cmd_xlo']) & 
              (x_i<=x_cut_settings['cmd_xhi']))
        
        x_i = x_i[ind]  
        y_i = y_i[ind]
        
        ax.plot(x_iso,y_iso, zorder=200, color='green',lw=plot_settings['lw'])
        
        legends.append(f'Age = {age_lin}')

    x_l = np.linspace(x_cut_settings['cmd_xlo'], x_cut_settings['cmd_xhi'])    

    # Bin mid points
    y_rgbn = np.arange(x_cut_settings['y_lo'], 
                       x_cut_settings['y_hi'], 
                       x_cut_settings['dy'])
    
    y_rgb_mid = y_rgbn[:-1] + x_cut_settings['dy']/2   
    
    if  x_cut_settings['fit_isochrone'] and not x_cut_settings['fit_rgb']:
        print("Fitting Isochrone")
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_i, x_i)
        slope = model_iso.slope.value

        y_plot  = np.linspace(axis_limits['ylims'][0],axis_limits['ylims'][1])
        ax.plot(model_iso(y_plot), y_plot,
                '--r', lw=plot_settings['lw'],  zorder=400)
        
    elif x_cut_settings['fit_rgb']:
        print("Fitting RGB stars")
        ind = ((x>=x_cut_settings['rgb_xlo']) & 
              (x<=x_cut_settings['rgb_xhi']) & 
              (y>=x_cut_settings['rgb_ylo']) & 
              (y<=x_cut_settings['rgb_yhi']))

        ind_out = ind
        y_n, x_n = running_avg(y[ind], x[ind], x_cut_settings['rgb_fit_bin'])
        
        ind = ~np.isnan(x_n)
        x_bin = x_n[ind]
        y_bin = y_n[ind]
        
        ax.plot(x_bin,y_bin, color='blue', zorder=390)
        
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_bin, x_bin)

        y_plot  = np.linspace(axis_limits['ylims'][0],axis_limits['ylims'][1])
        ax.plot(model_iso(y_plot), y_plot,
                '--r', lw=plot_settings['lw'],  zorder=400)
        
        slope = model_iso.slope.value

    elif x_cut_settings['slope'] is not None and x_cut_settings['intercept'] is not None:
        model_iso = models.Linear1D(slope=x_cut_settings['slope'],
                                    intercept=x_cut_settings['intercept'])

    if x_cut_settings['theta'] is None:
      theta=np.arctan(AF3/(AF1-AF2)) 
    else:
      theta = x_cut_settings['theta']
        
    x_rgb_mid = model_iso(y_rgb_mid)
    
    ax.scatter(x_rgb_mid, y_rgb_mid, c='r' ,zorder = 200,s=plot_settings['s'])

    dats = []
    
    dx0 = x_cut_settings['dx']
    dx  = x_cut_settings['cmd_xhi']- x_cut_settings['cmd_xlo']
    dy  = x_cut_settings['dy']
    
    for i, y0 in enumerate(y_rgbn[:-1]):

        # Extinction Vector
        x0 = model_iso(y0)
        x_l = np.linspace(x0-dx0/2, x0+dx0/2)   
        y_Avl = y0 + np.tan(theta)*(x_l-x0)
        
        x01 = model_iso(y0 + dy)
        y_Avu = y0 + dy + np.tan(theta)*(x_l-x01)

        ax.plot(x_l,y_Avl, color=x_cut_settings['color'],lw=plot_settings['lw'])
        ax.plot(x_l,y_Avu, color=x_cut_settings['color'],lw=plot_settings['lw'])

        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        
        model_Avl = fit(init, x_l, y_Avl)
        model_Avu = fit(init, x_l, y_Avu)

        c1 = (y>model_Avl(x)) & (y<=model_Avu(x))
        c2 = (x>=x_rgb_mid[i]-dx/2) & (x<=x_rgb_mid[i]+dx/2) 

        yn = y[c1&c2]
        xn = x[c1&c2]
        
        dat = np.array([xn, yn])
        dats.append(dat)
    
    # Extinction Vector
    AF1_ = Av_dict[filters['filt1']] * extinction['Av_']
    AF2_ = Av_dict[filters['filt2']] * extinction['Av_']
    AF3_ = Av_dict[filters['filt3']] * extinction['Av_']

    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(extinction['Av_x'], extinction['Av_y']),
                 xycoords='data',
                 xytext=(extinction['Av_x']+dx, 
                         extinction['Av_y']+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color='black',
                                 lw=5,
                                 ls='-')
               )

    ax.annotate(f"Av = {extinction['Av_']}",
                xy=(extinction['Av_x']-0.1, extinction['Av_y']-0.1)
                ,fontsize=plot_settings['Av.fontsize'])
    
    # Error models
    if not other_settings['skip_data']:
        ref = tab[f"mag_vega_{filters['filt3'].upper()}"]
        ref_new = np.arange(np.ceil(y.min()),np.floor(y.max())+0.5,0.5)

        mag_err1 = tab[error_settings['mag_err_cols'][0]]
        mag_err2 = tab[error_settings['mag_err_cols'][1]]

        if len(error_settings['mag_err_cols'])>2:
            mag_err3 = tab[error_settings['mag_err_cols'][2]]
        else:
            mag_err3 = mag_err2

        col_err = np.sqrt(mag_err1**2 + mag_err2**2)
        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_col = fit(init,ref,col_err)

        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_mag = fit(init,ref,mag_err3)

        x1 = error_settings['ref_xpos'] + 0*ref_new
        y1 = ref_new
        yerr = model_mag(ref_new)
        xerr = model_col(ref_new)
        
        if other_settings['show_err_model']:
            plt.show()
            plt.scatter(ref, mag_err3)
            plt.plot(ref_new,yerr,'--r')
            plt.show()
            plt.scatter(ref, col_err)
            plt.plot(ref_new,xerr,'--r')
            plt.show()
        ax.errorbar(x1, y1, yerr, xerr ,fmt='o', color = 'red', markersize=0.5, capsize=2) 

    ax.set_xlim(axis_limits['xlims'][0],axis_limits['xlims'][1])
    ax.set_ylim(axis_limits['ylims'][0],axis_limits['ylims'][1])
    
    ax.tick_params(which='both', length=15,direction="in", 
                   bottom=True, top=True,left=True, width = 3)
    
    ax.tick_params(which='minor', length=8, width = 3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.set_xlabel(f"{filters['filt1'].upper()} - {filters['filt2'].upper()}")
    ax.set_ylabel(filters['filt3'].upper())
    ax.invert_yaxis()
    fig.tight_layout()
    # Labels, ticks, and legend 
    ax.legend(legends, fontsize=plot_settings['legend.fontsize'], ncols = plot_settings['legend.ncols'])
    
    return fig, ax, dats, x_rgb_mid, y_rgb_mid, y_rgbn, model_iso

def gen_CMD_ycut(tab,
                df_iso = None,
                filters={'filt1': 'f115w', 'filt2': 'f150w'},
                positions={'ra_col': 'ra', 'dec_col': 'dec', 'ra_cen': 0, 'dec_cen': 0},
                region={'r_in': 0, 'r_out': 24, 'spatial_filter': 'circle','ang': 245.00492},
                extinction={'Av': 0.19, 'Av_': 3, 'Av_x': 2, 'Av_y': 19},
                distance_modulus=29.7415,
                axis_limits={'xlims': [-0.5, 2.5], 'ylims': [18, 28]},
                isochrone_params={'met': 0.02, 'ages': [7., 8., 9.]},
                plot_settings={'alpha': 1, 's': 0.2, 'lw': 3, 'cmap':'jet'},
                error_settings={ 'mag_err_lim': 0.2, 'show_err_model': False, 'ref_xpos': -0.5},
                kde_contours={'gen_kde': False, 'gen_contours': False, 'kde_bin': 200j},
                other_settings={'ab_dist': True, 'skip_data': False, 'show_err_model':False},
                y_cut_settings= {},                  
                fig=None,
                ax=None):
    
    filters.setdefault('filt1','f115w')
    filters.setdefault('filt2','f200w')
    filters.setdefault('filt3', filters['filt2'])
    
    positions.setdefault('ra_col','ra')
    positions.setdefault('dec_col','dec')
    positions.setdefault('ra_cen',0)
    positions.setdefault('dec_cen',0)
    
    region.setdefault('r_in',0)
    region.setdefault('r_out',10)
    region.setdefault('spatial_filter','circle')
    
    extinction.setdefault('Av',0.19)
    extinction.setdefault('Av_',3)
    extinction.setdefault('Av_x',3)
    extinction.setdefault('Av_y',19)
    
    axis_limits.setdefault('xlims', [-0.5, 2.5])
    axis_limits.setdefault('ylims', [18, 28])
    
    isochrone_params.setdefault('label_min', 0)
    isochrone_params.setdefault('label_max', 10)
    isochrone_params.setdefault('met', 0.002)
    isochrone_params.setdefault('age', 10)
    
    plot_settings.setdefault('Av.fontsize',15)
    plot_settings.setdefault('legend.fontsize',15)
    plot_settings.setdefault('lw',3)
    plot_settings.setdefault('s',0.2)
    plot_settings.setdefault('alpha',1)
    plot_settings.setdefault('print_met',False)
    plot_settings.setdefault('legend.ncols',1)
    plot_settings.setdefault('cmap','jet')
    
    
    error_settings.setdefault('mag_err_cols', [
        f'mag_err_{filters["filt1"].upper()}',
        f'mag_err_{filters["filt2"].upper()}',
        f'mag_err_{filters["filt3"].upper()}',])
    
    error_settings.setdefault('mag_err_lim',0.2)
    error_settings.setdefault('ref_xpos',-0.25)
    
    kde_contours.setdefault('gen_kde',False)
    kde_contours.setdefault('gen_contours',False)
    kde_contours.setdefault('kde_bin',100j)
    
    other_settings.setdefault('ab_dist',True)
    other_settings.setdefault('skip_data',False)
    other_settings.setdefault('show_err_model',False)

    y_cut_settings.setdefault('cmd_xlo', None)
    y_cut_settings.setdefault('cmd_xhi', None)
    y_cut_settings.setdefault('cmd_ylo', None)
    y_cut_settings.setdefault('cmd_yhi', None)
    y_cut_settings.setdefault('perp_iso', False)
    y_cut_settings.setdefault('y_lo', 22)
    y_cut_settings.setdefault('y_hi', 26.5)
    y_cut_settings.setdefault('dx', 0.5)
    y_cut_settings.setdefault('rgb_xlo', 0.5)
    y_cut_settings.setdefault('rgb_xhi', 2)
    y_cut_settings.setdefault('rgb_ylo', 23)
    y_cut_settings.setdefault('rgb_yhi', 26)
    y_cut_settings.setdefault('fit_isochrone', True)
    y_cut_settings.setdefault('fit_rgb', False)
    y_cut_settings.setdefault('x0', 1)
    y_cut_settings.setdefault('y0', None)
    y_cut_settings.setdefault('ref_dy', 0.5)
    y_cut_settings.setdefault('rgb_fit_bin', 100)

    # Filter table by magnitude errors
    for col in error_settings['mag_err_cols']:
        tab = tab[tab[col] <= error_settings['mag_err_lim']]
        
    df_iso = df_iso[df_iso['Zini']==isochrone_params['met']]
    df_iso = df_iso[np.round(df_iso['logAge'],1)==isochrone_params['age']]

    y_cut_settings['cmd_ylo'] = y_cut_settings['y_lo']
    y_cut_settings['cmd_yhi'] = y_cut_settings['y_hi']
    
    age = isochrone_params['age'] 
    if age <6:
        age_lin  = f'{np.ceil(10**age)} Myr'
    if age  >= 6  and age < 9:
        age -=6
        age_lin = f'{np.ceil(10**age)} Myr'
    elif age >= 9:
        age-=9
        age_lin = f'{np.ceil(10**age)} Gyr'

    if region['spatial_filter']=='circle':
        tab['r'] = angular_separation(
        tab[positions['ra_col']] * u.deg,
        tab[positions['dec_col']] * u.deg,
        positions['ra_cen'] * u.deg,
        positions['dec_cen'] * u.deg).to(u.arcsec).value
        
        tab = tab[(tab['r'] >= region['r_in'])
                  & (tab['r'] <= region['r_out'])]
        
    elif region['spatial_filter']=='box':   
        tab = box(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['width_in'] / 3600, region['height_in'] / 3600,
                  region['width_out'] / 3600, region['height_out'] / 3600,
                  region['ang'])

    elif region['spatial_filter']=='ellipse':
        tab = ellipse(tab, positions['ra_col'], positions['dec_col'],
                  positions['ra_cen'], positions['dec_cen'],
                  region['ang'], 
                  region['a1'] / 3600,region['b1'] / 3600,
                  region['a2'] / 3600,region['b2'] / 3600)


    # Extinction corrections
    AF1 = Av_dict[filters['filt1']] * extinction['Av']
    AF2 = Av_dict[filters['filt2']] * extinction['Av']
    AF3 = Av_dict[filters['filt3']] * extinction['Av']
    
        # Compute magnitudes and colors
    x = tab[f'mag_vega_{filters["filt1"].upper()}'] - tab[f'mag_vega_{filters["filt2"].upper()}']
    y = tab[f'mag_vega_{filters["filt3"].upper()}']

    x = x.value.astype(float)
    y = y.value.astype(float)

    if y_cut_settings['cmd_xlo'] is None or y_cut_settings['cmd_xhi'] is None:
        y_cut_settings['cmd_xlo'] = x.mean() - 1
        y_cut_settings['cmd_xhi'] = x.mean() + 1
    

    legends = []

    # Initialize figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    
    if kde_contours['gen_kde']:

        # Peform the kernel density estimate
        xx, yy = np.mgrid[
            axis_limits['xlims'][0]:axis_limits['xlims'][1]:kde_contours['kde_bin'],
            axis_limits['ylims'][0]:axis_limits['ylims'][1]:kde_contours['kde_bin']]
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = gaussian_kde(values, bw_method=0.05)
        f = np.reshape(kernel(positions), xx.shape)
        tick_color='white'
        ax.imshow(f.T, cmap=plot_settings['cmap'], 
                  extent=(*axis_limits['xlims'], *axis_limits['ylims']),
                  interpolation='nearest', aspect='auto', origin='lower')
            
    else:
        ax.scatter(x,y, s=plot_settings['s'], color='black')
        legends.append('data')
        
    trgb_mag_iso  = None
    trgb_col_iso = None

    if df_iso is not None:
        df_iso = df_iso[(df_iso['label']>=isochrone_params['label_min']) & 
                        (df_iso['label']<=isochrone_params['label_max'])]
        
        x_i =  (df_iso[f"{filters['filt1'].upper()}mag"] + AF1) - (df_iso[f"{filters['filt2'].upper()}mag"] + AF2)
        y_i =  df_iso[f"{filters['filt3'].upper()}mag"]
        
        # Max mag and Color
        trgb_mag_iso = y_i.min()
        trgb_col_iso = x_i[y_i==y_i.min()].values[0]
        
        y_i +=  AF3 + distance_modulus
        
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        
        x_iso = x_i.copy()
        y_iso = y_i.copy()

        ind = ((y_i>=y_cut_settings['cmd_ylo']-0.5) & 
              (y_i<=y_cut_settings['cmd_yhi']+0.5) &
              (x_i>=y_cut_settings['cmd_xlo']-0.2) & 
              (x_i<=y_cut_settings['cmd_xhi']+0.2))
        
        x_i = x_i[ind]  
        y_i = y_i[ind]
        
        ax.plot(x_iso,y_iso, zorder=200, color='black',lw=plot_settings['lw'])
        
        legends.append(f'Age = {age_lin}')

    x_l = np.linspace(0, 2)    

    # Bin mid points
    x_rgbn = np.arange(y_cut_settings['cmd_xlo'], 
                       y_cut_settings['cmd_xhi'] + y_cut_settings['dx']/2, 
                       y_cut_settings['dx'])
    
    x_rgb_mid = x_rgbn[:-1] +  y_cut_settings['dx']/2   
    
    if  y_cut_settings['fit_isochrone'] and not y_cut_settings['fit_rgb']:
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, x_i, y_i)
        slope = 1/model_iso.slope.value
    
    elif y_cut_settings['fit_rgb']:

        ind = ((x>=y_cut_settings['rgb_xlo']) & 
              (x<=y_cut_settings['rgb_xhi']) & 
              (y>=y_cut_settings['rgb_ylo']) & 
              (y<=y_cut_settings['rgb_yhi']))

        y_n, x_n = running_avg(y[ind], x[ind], y_cut_settings['rgb_fit_bin'])
        
        ind = ~np.isnan(x_n)
        x_bin = x_n[ind]
        y_bin = y_n[ind]
        
        ax.plot(x_bin,y_bin, color='blue', zorder=390)
        
        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        model_iso = fit(init, y_bin, x_bin)

        y_plot  = np.linspace(axis_limits['ylims'][0],axis_limits['ylims'][1])
        ax.plot(model_iso(y_plot), y_plot,
                '--r', lw=plot_settings['lw'],  zorder=400)
        
        slope = model_iso.slope.value
    else :
        slope=0

    y0 = y_cut_settings['y0']
    if y0 is None:
        y0 = y.mean()
        
    y_rgb_mid = y0 + x_rgb_mid*0

    dats = []
    
    init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    for x0 in x_rgbn[:-1]:
        
        y_l = np.linspace(y_cut_settings['cmd_ylo'], y_cut_settings['cmd_yhi'])
        x_l = slope*(y_l - y0) + x0
        
        y_r = np.linspace(y_cut_settings['cmd_ylo'], y_cut_settings['cmd_yhi'])
        x_r = slope*(y_r - y0) + x0 + y_cut_settings['dx']

        ax.plot(x_l,y_l, color='red', lw=plot_settings['lw'])
        ax.plot(x_r,y_r, color='red', lw=plot_settings['lw'])

        init = models.Linear1D()
        fit = fitting.LinearLSQFitter()
        
        model_l = fit(init, y_l,x_l)
        model_r = fit(init, y_r,x_r)

        c1 = (x>model_l(y)) & (x<=model_r(y))

        yn = y[np.where(c1)]
        xn = x[np.where(c1)]
        
        if not kde_contours['gen_kde']:
            ax.scatter(xn,yn, s =plot_settings['s'], color='green', zorder=100)
        dat = np.array([xn, yn])
        dats.append(dat)
    
     # Extinction Vector
    AF1_ = Av_dict[filters['filt1']] * extinction['Av_']
    AF2_ = Av_dict[filters['filt2']] * extinction['Av_']
    AF3_ = Av_dict[filters['filt3']] * extinction['Av_']

    dx = AF1_ - AF2_
    dy = AF3_

    ax.annotate('', xy=(extinction['Av_x'], extinction['Av_y']),
                 xycoords='data',
                 xytext=(extinction['Av_x']+dx, 
                         extinction['Av_y']+dy),
                 textcoords='data',
                 arrowprops=dict(arrowstyle= '<|-',
                                 color='black',
                                 lw=5,
                                 ls='-')
               )

    ax.annotate(f"Av = {extinction['Av_']}",
                xy=(extinction['Av_x']-0.1, extinction['Av_y']-0.1)
                ,fontsize=plot_settings['Av.fontsize'])
    
    # Error models
    if not other_settings['skip_data']:
        ref = tab[f"mag_vega_{filters['filt3'].upper()}"]
        ref_new = np.arange(np.ceil(y.min()),np.floor(y.max())+0.5,0.5)

        mag_err1 = tab[error_settings['mag_err_cols'][0]]
        mag_err2 = tab[error_settings['mag_err_cols'][1]]

        if len(error_settings['mag_err_cols'])>2:
            mag_err3 = tab[error_settings['mag_err_cols'][2]]
        else:
            mag_err3 = mag_err2

        col_err = np.sqrt(mag_err1**2 + mag_err2**2)
        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_col = fit(init,ref,col_err)

        init = models.Exponential1D()
        fit = fitting.LevMarLSQFitter()
        model_mag = fit(init,ref,mag_err3)

        x = error_settings['ref_xpos'] + 0*ref_new
        y = ref_new
        yerr = model_mag(ref_new)
        xerr = model_col(ref_new)
        
        if other_settings['show_err_model']:
            plt.show()
            plt.scatter(ref, mag_err3)
            plt.plot(ref_new,yerr,'--r')
            plt.show()
            plt.scatter(ref, col_err)
            plt.plot(ref_new,xerr,'--r')
            plt.show()
        ax.errorbar(x, y, yerr, xerr ,fmt='o', color = 'red', markersize=0.5, capsize=2) 

    ax.set_xlim(axis_limits['xlims'][0],axis_limits['xlims'][1])
    ax.set_ylim(axis_limits['ylims'][0],axis_limits['ylims'][1])
    
    ax.tick_params(which='both', length=15,direction="in", 
                   bottom=True, top=True,left=True, width = 3)
    
    ax.tick_params(which='minor', length=8, width = 3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.set_xlabel(f"{filters['filt1'].upper()} - {filters['filt2'].upper()}")
    ax.set_ylabel(filters['filt3'].upper())
    ax.invert_yaxis()
    fig.tight_layout()
    # Labels, ticks, and legend 
    ax.legend(legends, fontsize=plot_settings['legend.fontsize'], ncols = plot_settings['legend.ncols'])
    
    return fig, ax, dats, x_rgb_mid, y_rgb_mid, x_rgbn, [trgb_mag_iso, AF3, trgb_col_iso]
