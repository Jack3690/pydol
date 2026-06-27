import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from pydol.photometry.scripts.catalog_filter import ellipse, box, polygon
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QLabel,
    QListWidget,
    QComboBox,
    QFileDialog,
    QAction,
    QMessageBox,
    QSizePolicy,
    QLineEdit,
    QProgressDialog,
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches

from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.modeling import models, fitting
from astropy.visualization import simple_norm
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from astropy.wcs.utils import proj_plane_pixel_scales

# Minimalistic seaborn style
sb.set_theme(style="white")

# ==========================================================
# BACKGROUND WORKERS  (keeps UI responsive)
# ==========================================================

class CatalogWorker(QObject):
    """Load an astropy Table in a background thread.

    Strategy by file type:
    - .fits  -> fits.open(memmap=False) avoids the mmap warning on filesystems
                that don't support it (NFS, WSL, certain network mounts).
    - .csv / .txt -> pandas read_csv (3-10x faster than astropy for large text
                files) then converted back to an astropy Table.
    - other  -> astropy Table.read generic fallback.
    """
    finished = pyqtSignal(object)   # emits the Table on success
    error    = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            ext = os.path.splitext(self.filename)[1].lower()

            if ext in ('.fits', '.fit'):
                # memmap=False reads the HDU into RAM immediately, avoiding the
                # 'falling back to denywrite' warning entirely.
                with fits.open(self.filename, memmap=False) as hdul:
                    tab = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.dtype.names:
                            tab = Table(hdu.data)
                            break
                    if tab is None:
                        tab = Table.read(self.filename)

            elif ext in ('.csv', '.txt', '.tsv'):
                sep = '\t' if ext == '.tsv' else ','
                df  = pd.read_csv(self.filename, sep=sep, comment='#',
                                   low_memory=False)
                tab = Table.from_pandas(df)

            else:
                tab = Table.read(self.filename)

            self.finished.emit(tab)
        except Exception as exc:
            self.error.emit(str(exc))


class CMDWorker(QObject):
    """Run gen_CMD in a background thread."""
    finished = pyqtSignal(object, object, object)  # fig, ax, tab
    error    = pyqtSignal(str)

    def __init__(self, tab, df_iso, params, fig, ax):
        super().__init__()
        self.tab    = tab
        self.df_iso = df_iso
        self.params = params
        self.fig    = fig
        self.ax     = ax

    def run(self):
        try:
            p = self.params
            fig, ax, tab1 = gen_CMD(
                self.tab,
                self.df_iso,
                p['filters'],
                p['positions'],
                p['region'],
                p['extinction'],
                p['distance_modulus'],
                p['axis_limits'],
                p['isochrone_params'],
                plot_settings=p['plot_settings'],
                error_settings=p['error_settings'],
                kde_contours=p['kde_contours'],
                other_settings={'ab_dist': True},
                fig=self.fig,
                ax=self.ax,
            )
            self.finished.emit(fig, ax, tab1)
        except Exception as exc:
            self.error.emit(str(exc))

# mpl.rcParams.update({
#     #"text.usetex": False,                # If using LaTeX for labels
#   #  "font.family": "serif",
#   #  "font.serif": ["Computer Modern Roman"],
#     "axes.labelsize": 15,
#     "font.size": 15,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "axes.titlesize": 10,
#     "lines.linewidth": 1.0,
#     "lines.markersize": 3,
#     "figure.dpi": 100,                  # High-quality output
#     "savefig.dpi": 100,
#     "axes.grid": False,                 # Avoid grids unless needed
#     "legend.frameon": False             # No legend frame
# })

# plt.rcParams['axes.titlesize']  = plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 10
# ==========================================================
# CMD CANVAS
# ==========================================================

Av_dict = { 
            'f275w': 2.02499,
            'f336w': 1.67536,
            'f435w': 1.33879,
            'f555w': 1.03065,
            'f814w': 0.59696,
    
            'f090w': 0.583,
            'f115w': 0.419,
            'f150w': 0.287,
            'f200w': 0.195,
    
            'f438w': 1.34148,
            'f606w': 0.90941,
            'f814w': 0.59845
          }

def _is_float(s):
    """Return True if string s can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

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
    
    plot_settings.setdefault('Av.fontsize',8)
    plot_settings.setdefault('legend.fontsize',8)
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

    # Compute angular separation – fully vectorised (no per-row unit overhead)
    ra_rad   = np.deg2rad(np.asarray(tab[positions['ra_col']],  dtype=np.float64))
    dec_rad  = np.deg2rad(np.asarray(tab[positions['dec_col']], dtype=np.float64))
    ra_c     = np.deg2rad(float(positions['ra_cen']))
    dec_c    = np.deg2rad(float(positions['dec_cen']))
    tab['r'] = np.rad2deg(
        angular_separation(ra_rad, dec_rad, ra_c, dec_c)
    ) * 3600.0   # degrees → arcseconds

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

    # Compute magnitudes and colors – extract to plain float64 arrays immediately
    x = (np.asarray(tab[f'mag_vega_{filters["filt1"].upper()}'], dtype=np.float64)
       - np.asarray(tab[f'mag_vega_{filters["filt2"].upper()}'], dtype=np.float64))
    y =  np.asarray(tab[f'mag_vega_{filters["filt3"].upper()}'], dtype=np.float64)

    # Initialize figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Extinction corrections
    AF1 = Av_dict[filters['filt1']] * extinction['Av']
    AF2 = Av_dict[filters['filt2']] * extinction['Av']
    AF3 = Av_dict[filters['filt3']] * extinction['Av']

    # Kernel density estimation or scatter plot
    tick_color = 'black'
    _KDE_MAX_SAMPLES = 50_000   # subsample for speed on large catalogs
    if kde_contours['gen_kde'] and not kde_contours['gen_contours']:
        xx, yy = np.mgrid[
            axis_limits['xlims'][0]:axis_limits['xlims'][1]:kde_contours['kde_bin'],
            axis_limits['ylims'][0]:axis_limits['ylims'][1]:kde_contours['kde_bin']]
        
        positions_kde = np.vstack([xx.ravel(), yy.ravel()])
        # subsample for large datasets to keep KDE tractable
        n_pts = len(x)
        if n_pts > _KDE_MAX_SAMPLES:
            idx = np.random.choice(n_pts, _KDE_MAX_SAMPLES, replace=False)
            xs, ys = x[idx], y[idx]
        else:
            xs, ys = x, y
        values = np.vstack([xs, ys])

        kernel = gaussian_kde(values, bw_method=kde_contours['bw'])
        f = np.reshape(kernel(positions_kde), xx.shape)
        tick_color='white'
        perc_cut = np.percentile(f.ravel(), 84)

        f[f<=perc_cut] = np.nan
        norm = simple_norm(f.T, 'log', min_percent=10)
        ax.imshow(f.T, cmap=kde_contours['cmap'], extent=(*axis_limits['xlims'], *axis_limits['ylims']),
                  interpolation='nearest', aspect='auto', norm=norm, zorder=100, origin='lower')

    elif kde_contours['gen_contours']:
        ax.scatter(x, y, s=plot_settings['s'], color='black', label='data')
        cmap_custom = LinearSegmentedColormap.from_list("custom_grey_to_white", ["grey", "white"], N=256)
        # subsample seaborn kdeplot for large datasets
        n_pts = len(x)
        if n_pts > _KDE_MAX_SAMPLES:
            idx = np.random.choice(n_pts, _KDE_MAX_SAMPLES, replace=False)
            xs, ys = x[idx], y[idx]
        else:
            xs, ys = x, y
        sb.kdeplot(x=xs, y=ys, levels=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                   ax=ax, fill=True, cmap=cmap_custom)
        
        sb.kdeplot(x=xs, y=ys, levels=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                   ax=ax, color='black')

    if not other_settings['skip_data']:
        ax.scatter(x, y, s=plot_settings['s'], color='black', label='data', zorder=50)
        
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
                       
                mask = (y_iso.values[1:]- y_iso.values[:-1])<3
                mask = np.array([True] + list(mask))
                mask = np.where(~mask, np.nan, 1)
                
                if len(isochrone_params['met'])>1 or plot_settings['print_met']:
                    label = label=age_lin[i]+ f' {Z}'
                else:
                    label = label=age_lin[i]
                               
                ax.plot(x_iso*mask, y_iso*mask, lw=plot_settings['lw'],
                        label=label,alpha=plot_settings['alpha'], zorder=200)

    # Absolute magnitude
    if other_settings['ab_dist']:
        yticks = ax.get_yticks()
        yticks_n = yticks - distance_modulus - AF3
        
        dy = yticks_n - np.floor(yticks_n)
        ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis            
        ax1.set_ylabel(r'$M_{' + f"{filters['filt3'].upper()}" + r'}$')  # we already handled the x-label with ax1
        ax1.set_yticks(yticks - dy, np.floor(yticks_n))
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
class CMDCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.fig = Figure(figsize=(12, 10))
        super().__init__(self.fig)

        self.fig.tight_layout()

    def plot_cmd(self, tab, params):
        """Legacy synchronous path (kept for compatibility). Prefer threaded path."""
        self.ax = self.fig.add_subplot(111)
        while len(self.fig.axes) > 1:
            self.fig.axes[-1].remove()
        self.fig.axes[0].clear()
        self.ax = self.fig.axes[0]
        df_iso = getattr(self, 'df_iso', None)
        p = params
        gen_CMD(tab, df_iso,
            p['filters'], p['positions'], p['region'], p['extinction'],
            p['distance_modulus'], p['axis_limits'], p['isochrone_params'],
            plot_settings=p['plot_settings'], error_settings=p['error_settings'],
            kde_contours=p['kde_contours'], other_settings={'ab_dist': True},
            fig=self.fig, ax=self.ax)
        self.draw()



class FITSCanvas(FigureCanvasQTAgg):
    """FITS image canvas with level-of-detail (LOD) rendering.

    The full-resolution float32 array is always kept in ``self._full_data``.
    Whenever the axes viewport changes (zoom / pan / resize) ``_refresh_lod``
    is called: it clips the visible pixel rectangle, block-averages it down to
    at most ``_SCREEN_PX`` pixels in each dimension, and replaces the imshow
    data in-place — no full redraw needed.  WCS coordinates are preserved
    because we never touch the axes transform; only the image array changes.
    """

    _SCREEN_PX = 1024   # target max pixels per axis for the displayed tile

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.fig.clear()

        self._full_data  = None   # float32 full-res array
        self._im         = None   # the AxesImage kept alive for set_data
        self._lod_cid    = None   # mpl callback id for xlim_changed
        self._resize_cid = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tile(data, x0, x1, y0, y1):
        """Clip pixel bounds to array and return the sub-array + actual bounds."""
        H, W = data.shape
        px0 = max(0,   int(np.floor(x0)))
        px1 = min(W,   int(np.ceil(x1)))
        py0 = max(0,   int(np.floor(y0)))
        py1 = min(H,   int(np.ceil(y1)))
        # guard against degenerate view (e.g. during init)
        if px1 <= px0 or py1 <= py0:
            return data, 0, W, 0, H
        return data[py0:py1, px0:px1], px0, px1, py0, py1

    @staticmethod
    def _block_avg(arr, max_px):
        """Block-average arr so neither dimension exceeds max_px."""
        h, w = arr.shape
        ry = max(1, int(np.ceil(h / max_px)))
        rx = max(1, int(np.ceil(w / max_px)))
        if ry == 1 and rx == 1:
            return arr
        h2 = (h // ry) * ry
        w2 = (w // rx) * rx
        return arr[:h2, :w2].reshape(h2 // ry, ry, w2 // rx, rx).mean(axis=(1, 3))

    def _get_pixel_viewport(self):
        """Return current axes viewport in *pixel* coordinates (x0,x1,y0,y1)."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if self.wcs is not None:
            # Axes are in world coords (RA/Dec degrees) — convert to pixels.
            # wcs_world2pix expects [[ra, dec], ...] and returns [[x, y], ...]
            try:
                corners_world = np.array([
                    [xlim[0], ylim[0]],
                    [xlim[1], ylim[1]],
                ])
                corners_pix = self.wcs.wcs_world2pix(corners_world, 0)
                x0 = min(corners_pix[:, 0])
                x1 = max(corners_pix[:, 0])
                y0 = min(corners_pix[:, 1])
                y1 = max(corners_pix[:, 1])
                return x0, x1, y0, y1
            except Exception:
                pass

        # No WCS, or conversion failed — axes are already in pixel coords.
        x0, x1 = sorted(xlim)
        y0, y1 = sorted(ylim)
        return x0, x1, y0, y1

    # ------------------------------------------------------------------
    # LOD refresh — called on every zoom/pan/resize
    # ------------------------------------------------------------------

    def _refresh_lod(self, *_):
        if self._full_data is None or self._im is None:
            return

        x0, x1, y0, y1 = self._get_pixel_viewport()
        tile, tx0, tx1, ty0, ty1 = self._extract_tile(
            self._full_data, x0, x1, y0, y1
        )

        # Downsample to screen resolution
        display = self._block_avg(tile.astype(np.float32, copy=False),
                                   self._SCREEN_PX)

        # Update image data and extent so it maps to correct world coords.
        # extent is (left, right, bottom, top) in *axes* data units.
        if self.wcs is not None:
            # Convert pixel corners back to world coords for the extent.
            try:
                corners_pix = np.array([
                    [tx0, ty0],
                    [tx1, ty1],
                ])
                corners_world = self.wcs.wcs_pix2world(corners_pix, 0)
                ext_x0 = corners_world[0, 0]
                ext_x1 = corners_world[1, 0]
                ext_y0 = corners_world[0, 1]
                ext_y1 = corners_world[1, 1]
            except Exception:
                ext_x0, ext_x1, ext_y0, ext_y1 = tx0, tx1, ty0, ty1
        else:
            ext_x0, ext_x1, ext_y0, ext_y1 = tx0, tx1, ty0, ty1

        self._im.set_data(display)
        self._im.set_extent([ext_x0, ext_x1, ext_y0, ext_y1])
        # Redraw only the image artist — much cheaper than full draw()
        self.ax.draw_artist(self._im)
        self.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_fits(self, data, wcs=None, title=None):
        """Load a new image.  Always stores full-res; LOD kicks in on interaction."""

        # Disconnect old LOD callbacks
        for cid_attr in ('_lod_cid', '_resize_cid'):
            cid = getattr(self, cid_attr, None)
            if cid is not None:
                try:
                    self.ax.callbacks.disconnect(cid)
                except Exception:
                    pass
                try:
                    self.mpl_disconnect(cid)
                except Exception:
                    pass
            setattr(self, cid_attr, None)

        # Store full-res data
        self._full_data = data.astype(np.float32, copy=False)
        H, W = self._full_data.shape

        # Rebuild axes (needed when WCS changes projection)
        self.fig.clear()
        self.wcs = wcs
        if wcs is not None:
            self.ax = self.fig.add_subplot(111, projection=wcs)
        else:
            self.ax = self.fig.add_subplot(111)

        # Initial display: downsample full image to screen size
        initial = self._block_avg(self._full_data, self._SCREEN_PX)

        # Compute full-image extent in axes coords
        if wcs is not None:
            try:
                corners = wcs.wcs_pix2world(
                    np.array([[0, 0], [W, H]]), 0
                )
                extent = [corners[0,0], corners[1,0], corners[0,1], corners[1,1]]
            except Exception:
                extent = [-0.5, W - 0.5, -0.5, H - 0.5]
        else:
            extent = [-0.5, W - 0.5, -0.5, H - 0.5]

        self._im = self.ax.imshow(
            initial,
            origin='lower',
            cmap='jet',
            norm=LogNorm(vmin=self.vmin, vmax=self.vmax),
            interpolation='nearest',
            extent=extent,
            aspect='equal',
        )

        if title:
            self.ax.set_title(title)

        self.fig.tight_layout()
        self.ax.grid(False)

        # Connect LOD refresh to viewport-change events
        self._lod_cid    = self.ax.callbacks.connect(
            'xlim_changed', self._refresh_lod
        )
        self._resize_cid = self.mpl_connect(
            'resize_event', self._refresh_lod
        )

        self.draw()


# ==========================================================
# MAIN WINDOW
# ==========================================================

class StellarPopulationGUI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Interactive CMD Explorer"
        )

        self.resize(1800, 900)
        self.catalog_table = None

        self.init_ui()

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------

    def init_ui(self):

        self.create_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --------------------------------------------
        # LEFT PANEL
        # --------------------------------------------

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # FITS image viewer

        img_label = QLabel("Image Viewer")
        img_font = img_label.font()
        img_font.setPointSize(14)
        img_font.setBold(True)
        img_label.setFont(img_font)

        self.image_canvas = FITSCanvas()
        self.toolbar = NavigationToolbar2QT(self.image_canvas, self)
        left_layout.addWidget(img_label)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.image_canvas)

        # --------------------------------------------
        # Region controls
        # --------------------------------------------

        region_panel = QWidget()

        region_layout = QVBoxLayout()
        region_panel.setLayout(region_layout)

        region_layout.addWidget(QLabel("Regions"))

        self.btn_rectangle = QPushButton("Add Rectangle")
        self.btn_ellipse = QPushButton("Add Ellipse")
        self.btn_circle = QPushButton("Add Circle")
        self.btn_annulus = QPushButton("Add Annulus")

        # connect region buttons
        self.btn_rectangle.clicked.connect(self.start_rectangle_mode)
        region_layout.addWidget(self.btn_rectangle)
        region_layout.addWidget(self.btn_ellipse)
        region_layout.addWidget(self.btn_circle)
        region_layout.addWidget(self.btn_annulus)

        region_layout.addWidget(QLabel("Current Regions"))

        self.region_list = QListWidget()

        region_layout.addWidget(self.region_list)

        # interactive rectangle controls are handled directly on the image (no manual inputs)

        left_layout.addWidget(region_panel)

        # --------------------------------------------
        # RIGHT PANEL (CMD)
        # --------------------------------------------

        right_widget = QWidget()

        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        cmd_label = QLabel("Color-Magnitude Diagram")
        cmd_font = cmd_label.font()
        cmd_font.setPointSize(14)
        cmd_font.setBold(True)
        cmd_label.setFont(cmd_font)

        filter_panel = QWidget()
        filter_layout = QHBoxLayout()
        filter_panel.setLayout(filter_layout)

        filter_layout.addWidget(QLabel("Filter 1:"))
        self.filter1_combo = QComboBox()
        self.filter1_combo.setEnabled(False)
        filter_layout.addWidget(self.filter1_combo)

        filter_layout.addWidget(QLabel("Filter 2:"))
        self.filter2_combo = QComboBox()
        self.filter2_combo.setEnabled(False)
        filter_layout.addWidget(self.filter2_combo)

        filter_layout.addWidget(QLabel("Magnitude:"))
        self.filter3_combo = QComboBox()
        self.filter3_combo.setEnabled(False)
        filter_layout.addWidget(self.filter3_combo)

        self.cmd_canvas = CMDCanvas()

        self.toolbar2 = NavigationToolbar2QT(self.cmd_canvas, self)
        right_layout.addWidget(cmd_label)
        right_layout.addWidget(self.toolbar2)
        right_layout.addWidget(self.cmd_canvas)
        right_layout.addWidget(filter_panel)
        
        # --------------------------------------------
        # AXIS LIMITS PANEL
        # --------------------------------------------
        
        axis_limits_panel = QWidget()
        axis_limits_layout = QHBoxLayout()
        axis_limits_panel.setLayout(axis_limits_layout)
        
        # X-axis limits
        axis_limits_layout.addWidget(QLabel("X-min:"))
        self.xmin_input = QLineEdit()
        self.xmin_input.setText("-1")
        axis_limits_layout.addWidget(self.xmin_input)
        
        axis_limits_layout.addWidget(QLabel("X-max:"))
        self.xmax_input = QLineEdit()
        self.xmax_input.setText("5.2")
        axis_limits_layout.addWidget(self.xmax_input)
        
        # Y-axis limits
        axis_limits_layout.addWidget(QLabel("Y-min:"))
        self.ymin_input = QLineEdit()
        self.ymin_input.setText("17")
        axis_limits_layout.addWidget(self.ymin_input)
        
        axis_limits_layout.addWidget(QLabel("Y-max:"))
        self.ymax_input = QLineEdit()
        self.ymax_input.setText("28")
        axis_limits_layout.addWidget(self.ymax_input)
        
        right_layout.addWidget(axis_limits_panel)
        
        # --------------------------------------------
        # EXTINCTION PANEL
        # --------------------------------------------
        
        extinction_panel = QWidget()
        extinction_layout = QHBoxLayout()
        extinction_panel.setLayout(extinction_layout)
        
        extinction_layout.addWidget(QLabel("Av:"))
        self.av_input = QLineEdit()
        self.av_input.setText("0.19")
        extinction_layout.addWidget(self.av_input)
        
        extinction_layout.addWidget(QLabel("Av_ (for arrow):"))
        self.av_arrow_input = QLineEdit()
        self.av_arrow_input.setText("3")
        extinction_layout.addWidget(self.av_arrow_input)
        
        extinction_layout.addWidget(QLabel("Av_x:"))
        self.av_x_input = QLineEdit()
        self.av_x_input.setText("4")
        extinction_layout.addWidget(self.av_x_input)
        
        extinction_layout.addWidget(QLabel("Av_y:"))
        self.av_y_input = QLineEdit()
        self.av_y_input.setText("25")
        extinction_layout.addWidget(self.av_y_input)
        
        right_layout.addWidget(extinction_panel)
        
        # --------------------------------------------
        # DISTANCE MODULUS PANEL
        # --------------------------------------------
        
        distance_panel = QWidget()
        distance_layout = QHBoxLayout()
        distance_panel.setLayout(distance_layout)
        
        distance_layout.addWidget(QLabel("Distance Modulus:"))
        self.distance_modulus_input = QLineEdit()
        self.distance_modulus_input.setText("29.81")
        distance_layout.addWidget(self.distance_modulus_input)
        
        distance_layout.addWidget(QLabel("Label Min:"))
        self.label_min_input = QLineEdit()
        self.label_min_input.setText("0")
        distance_layout.addWidget(self.label_min_input)
        
        distance_layout.addWidget(QLabel("Label Max:"))
        self.label_max_input = QLineEdit()
        self.label_max_input.setText("10")
        distance_layout.addWidget(self.label_max_input)
        
        right_layout.addWidget(distance_panel)
        
        # --------------------------------------------
        # METALLICITY PANEL
        # --------------------------------------------
        
        metallicity_panel = QWidget()
        metallicity_layout = QHBoxLayout()
        metallicity_panel.setLayout(metallicity_layout)
        
        metallicity_layout.addWidget(QLabel("Metallicity:"))
        self.metallicity_input = QLineEdit()
        self.metallicity_input.setText("0.02")
        metallicity_layout.addWidget(self.metallicity_input)
        
        right_layout.addWidget(metallicity_panel)
        
        # --------------------------------------------
        # AGES PANEL
        # --------------------------------------------
        
        ages_panel = QWidget()
        ages_layout = QHBoxLayout()
        ages_panel.setLayout(ages_layout)
        
        ages_layout.addWidget(QLabel("Ages (comma-sep):"))
        self.ages_input = QLineEdit()
        self.ages_input.setText("6.7,7,7.7,8")
        ages_layout.addWidget(self.ages_input)
        
        right_layout.addWidget(ages_panel)
        
        # --------------------------------------------
        # PLOT SETTINGS PANEL
        # --------------------------------------------
        
        plot_settings_panel = QWidget()
        plot_settings_layout = QHBoxLayout()
        plot_settings_panel.setLayout(plot_settings_layout)
        
        plot_settings_layout.addWidget(QLabel("Marker Size (s):"))
        self.marker_size_input = QLineEdit()
        self.marker_size_input.setText("3")
        plot_settings_layout.addWidget(self.marker_size_input)
        
        plot_settings_layout.addWidget(QLabel("Alpha:"))
        self.alpha_input = QLineEdit()
        self.alpha_input.setText("1")
        plot_settings_layout.addWidget(self.alpha_input)
        
        plot_settings_layout.addWidget(QLabel("Legend Columns:"))
        self.legend_ncols_input = QLineEdit()
        self.legend_ncols_input.setText("2")
        plot_settings_layout.addWidget(self.legend_ncols_input)
        
        right_layout.addWidget(plot_settings_panel)
        
        # --------------------------------------------
        # ERROR SETTINGS PANEL
        # --------------------------------------------
        
        error_settings_panel = QWidget()
        error_settings_layout = QHBoxLayout()
        error_settings_panel.setLayout(error_settings_layout)
        
        error_settings_layout.addWidget(QLabel("Mag Error Limit:"))
        self.mag_err_lim_input = QLineEdit()
        self.mag_err_lim_input.setText("0.2")
        error_settings_layout.addWidget(self.mag_err_lim_input)
        
        error_settings_layout.addWidget(QLabel("Ref X Position:"))
        self.ref_xpos_input = QLineEdit()
        self.ref_xpos_input.setText("-0.5")
        error_settings_layout.addWidget(self.ref_xpos_input)
        
        right_layout.addWidget(error_settings_panel)
        
        # Add stretch to push button to bottom
        right_layout.addStretch()
        
        # Submit button
        self.submit_cmd_button = QPushButton("Plot CMD")
        self.submit_cmd_button.clicked.connect(self.plot_selected_cmd)
        right_layout.addWidget(self.submit_cmd_button)

        # --------------------------------------------
        # SPLITTER
        # --------------------------------------------

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        splitter.setSizes([900, 500])

        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------
    # MENU
    # ------------------------------------------------------

    def create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        load_image_action = QAction(
            "Load FITS Image",
            self
        )

        load_catalog_action = QAction(
            "Load Catalog",
            self
        )

        load_isochrone_action = QAction(
            "Load Isochrone",
            self
        )

        file_menu.addAction(load_image_action)
        file_menu.addAction(load_catalog_action)
        file_menu.addAction(load_isochrone_action)

        load_image_action.triggered.connect(
            self.load_fits_image
        )

        load_catalog_action.triggered.connect(
            self.load_catalog
        )

        load_isochrone_action.triggered.connect(
            self.load_isochrone
        )

    # ------------------------------------------------------
    # PLACEHOLDERS
    # ------------------------------------------------------

    def load_fits_image(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load FITS Image",
            "",
            "FITS Files (*.fits)"
        )

        if not filename:
            return

        try:
            with fits.open(filename, memmap=False) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                self.ra_cen = header['CRVAL1'] if 'CRVAL1' in header else None
                self.dec_cen = header['CRVAL2'] if 'CRVAL2' in header else None
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                    header = hdul[1].header

                if data is None:
                    raise ValueError("No image data found in FITS file.")

                if data.ndim == 3:
                    data = data[0]

                if data.ndim != 2:
                    raise ValueError("FITS image data must be 2D.")

                data = np.array(data, dtype=np.float32)
                wcs = None
                try:
                    wcs = WCS(header)
                except Exception:
                    wcs = None

                # Auto-compute display range from the data
                finite = data[np.isfinite(data)]
                vmin = max(0.0, float(np.percentile(finite, 1)))
                vmax = float(np.percentile(finite, 99.5))
                self.image_canvas.vmin = vmin
                self.image_canvas.vmax = vmax
                self.data = data
                self.wcs = wcs
                self.title = os.path.basename(filename)
                self.image_canvas.plot_fits(data, wcs=wcs, title=os.path.basename(filename))

            self.statusBar().showMessage(f"Loaded image: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Load FITS Image", str(exc))
            self.statusBar().showMessage("Failed to load FITS image.")

    def load_isochrone(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load PARSEC Isochrone CSV",
            "",
            "CSV Files (*.csv)"
        )

        try:
            self.cmd_canvas.df_iso = pd.read_csv(filename)
            self.statusBar().showMessage(f"Loaded isochrone: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Isochrone", str(exc))
            self.statusBar().showMessage("Failed to load isochrone.")

        if not filename:
            return
    def update_cmd_filters(self, table):
        self.filter1_combo.clear()
        self.filter2_combo.clear()
        self.filter3_combo.clear()
        filter_columns = [col for col in table.keys() if col.startswith("mag_vega_")]

        if not filter_columns:
            placeholder = "No available filters"
            self.filter1_combo.addItem(placeholder)
            self.filter2_combo.addItem(placeholder)
            self.filter3_combo.addItem(placeholder)
            self.filter1_combo.setEnabled(False)
            self.filter2_combo.setEnabled(False)
            self.filter3_combo.setEnabled(False)
            return

        filter_columns = sorted(filter_columns)

        self.filter1_combo.blockSignals(True)
        self.filter2_combo.blockSignals(True)
        self.filter3_combo.blockSignals(True)

        for col in filter_columns:
            label = col.replace("mag_vega_", "")
            value = label.lower()
            self.filter1_combo.addItem(label, value)
            self.filter2_combo.addItem(label, value)
            self.filter3_combo.addItem(label, value)

        last_index = len(filter_columns) - 1
        self.filter2_combo.setCurrentIndex(last_index)
        self.filter3_combo.setCurrentIndex(last_index)

        self.filter1_combo.setEnabled(True)
        self.filter2_combo.setEnabled(True)
        self.filter3_combo.setEnabled(True)

        self.filter1_combo.blockSignals(False)
        self.filter2_combo.blockSignals(False)
        self.filter3_combo.blockSignals(False)

    def plot_selected_cmd(self, *args):
        self.statusBar().showMessage("Plotting CMD...")
        if self.catalog_table is None:
            return

        if not (self.filter1_combo.isEnabled() and self.filter2_combo.isEnabled() and self.filter3_combo.isEnabled()):
            return

        filt1 = self.filter1_combo.currentData()
        filt2 = self.filter2_combo.currentData()
        filt3 = self.filter3_combo.currentData()

        if not filt1 or not filt2 or not filt3:
            return

        if filt1 == filt2:
            QMessageBox.warning(self, "CMD Plot", "Filter 1 and Filter 2 must be different for a valid color.")
            return

        self.ra_cen = 0.5*(self.catalog_table ['ra'].max() + self.catalog_table['ra'].min())  if 'ra' in self.catalog_table.colnames else None
        self.dec_cen =0.5*(self.catalog_table['dec'].max() + self.catalog_table['dec'].min()) if 'dec' in self.catalog_table.colnames else None
        
        # Parse all input fields with error handling
        try:
            xmin = float(self.xmin_input.text())
            xmax = float(self.xmax_input.text())
            ymin = float(self.ymin_input.text())
            ymax = float(self.ymax_input.text())
        except ValueError:
            xmin, xmax = -1, 5.2
            ymin, ymax = 17, 28
        
        try:
            av = float(self.av_input.text())
            av_arrow = float(self.av_arrow_input.text())
            av_x = float(self.av_x_input.text())
            av_y = float(self.av_y_input.text())
        except ValueError:
            av, av_arrow, av_x, av_y = 0.19, 3, 4, 25
        
        try:
            distance_modulus = float(self.distance_modulus_input.text())
        except ValueError:
            distance_modulus = 29.81
        
        try:
            if ',' in self.metallicity_input.text():
                metallicity = [float(x.strip()) for x in self.metallicity_input.text().split(',')]
            else:
                metallicity = [float(self.metallicity_input.text())]
        except ValueError:
            metallicity = [0.02]
        
        try:
            ages = [float(x.strip()) for x in self.ages_input.text().split(',')]
        except ValueError:
            ages = [6.7, 7, 7.7, 8]
        
        try:
            label_min = float(self.label_min_input.text())
            label_max = float(self.label_max_input.text())
        except ValueError:
            label_min, label_max = 0, 10
        
        try:
            marker_size = float(self.marker_size_input.text())
            alpha = float(self.alpha_input.text())
            legend_ncols = int(self.legend_ncols_input.text())
        except ValueError:
            marker_size, alpha, legend_ncols = 3, 1, 2
        
        try:
            mag_err_lim = float(self.mag_err_lim_input.text())
            ref_xpos = float(self.ref_xpos_input.text())
        except ValueError:
            mag_err_lim, ref_xpos = 0.2, -0.5
                
        # Extract current rectangle state only when Plot CMD is clicked
        region_for_params = self._extract_current_region()

        # translate interactive region to gen_CMD expected keys
        if region_for_params is not None and region_for_params.get('spatial_filter') == 'box':
            region_dict = {
                'spatial_filter': 'box',
                'width_in': 0,
                'height_in': 0,
                'width_out': region_for_params['width'],
                'height_out': region_for_params['height'],
                'ang': region_for_params.get('ang', 0),
                'ra_cen': region_for_params.get('ra_cen'),
                'dec_cen': region_for_params.get('dec_cen'),
            }
        else:
            region_dict = None

        # If interactive region provides RA/Dec center, prefer it
        ra_cen_use = self.ra_cen
        dec_cen_use = self.dec_cen
        if region_for_params is not None:
            if region_for_params.get('ra_cen') is not None and region_for_params.get('dec_cen') is not None:
                ra_cen_use = region_for_params.get('ra_cen')
                dec_cen_use = region_for_params.get('dec_cen')

        params = {
            'filters': {'filt1': filt1, 'filt2': filt2, 'filt3': filt3},
            'positions': {
                'ra_col': 'ra',
                'dec_col': 'dec',
                'ra_cen': ra_cen_use,
                'dec_cen': dec_cen_use,
            },
            'region': region_dict if region_dict is not None else {'a1': 0, 'b1': 0, 'a2': 2000, 'b2': 2000, 'ang': 0, 'spatial_filter': 'ellipse'},
            'extinction': {'Av': av, 'Av_x': av_x, 'Av_y': av_y, 'Av_': av_arrow},
            'distance_modulus': distance_modulus,
            'axis_limits': {'xlims': [xmin, xmax], 'ylims': [ymin, ymax]},
            'isochrone_params': {'met': metallicity, 'label_min': label_min, 'label_max': label_max, 'ages': ages},
            'plot_settings': {'s': marker_size, 'legend.ncols': legend_ncols, 'alpha': alpha},
            'error_settings': {'ref_xpos': ref_xpos, 'mag_err_lim': mag_err_lim},
            'kde_contours': {'gen_kde': False, 'gen_contours': False},
            'other_settings': {},
        }

        self._update_region_display(region_for_params)
        try:
            # Run CMD generation in a background thread to keep UI responsive
            self.submit_cmd_button.setEnabled(False)
            self.statusBar().showMessage("Plotting CMD…")

            # prepare axes on main thread (matplotlib requirement)
            self.cmd_canvas.ax = self.cmd_canvas.fig.add_subplot(111)
            while len(self.cmd_canvas.fig.axes) > 1:
                self.cmd_canvas.fig.axes[-1].remove()
            self.cmd_canvas.fig.axes[0].clear()
            self.cmd_canvas.ax = self.cmd_canvas.fig.axes[0]

            df_iso = getattr(self.cmd_canvas, 'df_iso', None)

            self._cmd_thread = QThread()
            self._cmd_worker = CMDWorker(
                self.catalog_table, df_iso, params,
                self.cmd_canvas.fig, self.cmd_canvas.ax
            )
            self._cmd_worker.moveToThread(self._cmd_thread)
            self._cmd_thread.started.connect(self._cmd_worker.run)
            self._cmd_worker.finished.connect(self._on_cmd_done)
            self._cmd_worker.error.connect(self._on_cmd_error)
            self._cmd_worker.finished.connect(self._cmd_thread.quit)
            self._cmd_worker.error.connect(self._cmd_thread.quit)
            self._cmd_thread.start()
        except Exception as exc:
            self.submit_cmd_button.setEnabled(True)
            QMessageBox.warning(self, "CMD Plot", f"Unable to plot CMD: {exc}")

    def _on_cmd_done(self, fig, ax, tab1):
        self.submit_cmd_button.setEnabled(True)
        self.cmd_canvas.draw()
        self.statusBar().showMessage(f"CMD plotted successfully ({len(tab1):,} stars).")

    def _on_cmd_error(self, msg):
        self.submit_cmd_button.setEnabled(True)
        QMessageBox.warning(self, "CMD Plot", f"Unable to plot CMD: {msg}")
        self.statusBar().showMessage("CMD plot failed.")

    def load_catalog(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Catalog",
            "",
            "Catalog Files (*.csv *.txt *.fits)"
        )

        if not filename:
            return

        # Show a non-modal progress indicator while loading in background
        self._catalog_progress = QProgressDialog("Loading catalog…", None, 0, 0, self)
        self._catalog_progress.setWindowTitle("Please wait")
        self._catalog_progress.setWindowModality(Qt.WindowModal)
        self._catalog_progress.setMinimumDuration(0)
        self._catalog_progress.setValue(0)
        self._catalog_progress.show()

        # Disable load/plot controls while loading
        self.submit_cmd_button.setEnabled(False)

        self._cat_thread = QThread()
        self._cat_worker = CatalogWorker(filename)
        self._cat_worker.moveToThread(self._cat_thread)
        self._cat_thread.started.connect(self._cat_worker.run)
        self._cat_worker.finished.connect(self._on_catalog_loaded)
        self._cat_worker.error.connect(self._on_catalog_error)
        self._cat_worker.finished.connect(self._cat_thread.quit)
        self._cat_worker.error.connect(self._cat_thread.quit)
        self._cat_thread.start()

    def _on_catalog_loaded(self, tab):
        self._catalog_progress.close()
        self.submit_cmd_button.setEnabled(True)
        self.catalog_table = tab
        self.update_cmd_filters(tab)
        self.statusBar().showMessage(
            f"Loaded catalog: {len(tab):,} rows"
        )

    def _on_catalog_error(self, msg):
        self._catalog_progress.close()
        self.submit_cmd_button.setEnabled(True)
        QMessageBox.critical(self, "Load Catalog", msg)
        self.statusBar().showMessage("Failed to load catalog.")

    def _update_region_display(self, region):
        self.region_list.clear()
        if region is None:
            self.region_list.addItem("No active region")
            return

        width = region.get('width')
        height = region.get('height')
        angle = region.get('ang', 0)
        ra_cen = region.get('ra_cen')
        dec_cen = region.get('dec_cen')

        self.region_list.addItem(f"Width: {width:.1f} arcsec")
        self.region_list.addItem(f"Height: {height:.1f} arcsec")
        self.region_list.addItem(f"Angle: {angle:.1f}°")
        if ra_cen is not None and dec_cen is not None:
            self.region_list.addItem(f"RA center: {ra_cen:.6f}")
            self.region_list.addItem(f"Dec center: {dec_cen:.6f}")

    # ---------------------------
    # Rectangle selection handlers
    # ---------------------------
    def start_rectangle_mode(self):
        """Enable interactive rectangle drawing on the FITS image."""
        if not hasattr(self, 'image_canvas'):
            return
        ax = self.image_canvas.ax
        
        # Remove any existing rectangle patch and handles
        if hasattr(self, 'rect_patch') and self.rect_patch is not None:
            try:
                self.rect_patch.remove()
            except Exception:
                pass
            self.rect_patch = None
        
        if hasattr(self, 'rect_handles') and self.rect_handles is not None:
            try:
                for h in self.rect_handles:
                    h.remove()
            except Exception:
                pass
            self.rect_handles = []
        
        # ensure any existing selector is removed
        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            try:
                self.rect_selector.set_active(False)
                del self.rect_selector
            except Exception:
                pass

        # Reset rotation slider
        #self.angle_slider.setValue(0)
        self.region_list.clear()
        
        # Create RectangleSelector (avoid using 'drawtype' for compatibility)
        self.rect_selector = RectangleSelector(
            ax,
            self.on_rect_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='data',
            interactive=True,
        )
        self.image_canvas.draw()
        self.statusBar().showMessage('Draw rectangle on image (click-drag).')


    def on_rect_select(self, eclick, erelease):
        """Callback when rectangle is drawn. eclick/erelease are mouse events."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        # compute center and extents in data coordinates
        x0 = min(x1, x2)
        y0 = min(y1, y2)
        width_data = abs(x2 - x1)
        height_data = abs(y2 - y1)

        # convert to arcseconds if WCS present
        if hasattr(self.image_canvas, 'wcs') and self.image_canvas.wcs is not None:
            scales = proj_plane_pixel_scales(self.image_canvas.wcs)  # degrees/pixel
            # assume width_data/height_data are in world units (degrees) when WCS projection used
            # if the axes are in world coords, width_data already in degrees
            # but if the selector returned pixel coords, convert pixels->deg via scales
            # Try to detect: if width_data < 1e-3 treat as degrees already, else assume pixels
            if abs(width_data) < 1e-3:
                width_deg = width_data
                height_deg = height_data
            else:
                # treat as pixel counts
                width_deg = width_data * scales[0]
                height_deg = height_data * scales[1]

            width_arcsec = width_deg * 3600.0
            height_arcsec = height_deg * 3600.0
            # rectangle in data units should be degrees
            patch_width = width_deg
            patch_height = height_deg
        else:
            # No WCS: treat data units as pixels; assume 1 pixel = 1 arcsec
            width_arcsec = width_data
            height_arcsec = height_data
            patch_width = width_data
            patch_height = height_data

        angle = 0.0

        # keep the RectangleSelector active so the rectangle remains editable after release
        self.image_canvas.draw()
        self.region_list.clear()
        self.region_list.addItem("Region defined. Click Plot CMD to show properties.")
        self.statusBar().showMessage('Rectangle added. Drag the corners or edges to edit the selection, then click Plot CMD.')

    # manual rectangle apply removed; drawing/editing is interactive only

    # ---------------------------
    # Interactive rectangle editing
    # ---------------------------
    def _rect_corners(self, x0, y0, w, h, angle_deg):
        """Return rotated rectangle corners given lower-left x0,y0, width, height and angle in degrees."""
        cx = x0 + w / 2.0
        cy = y0 + h / 2.0
        theta = np.deg2rad(angle_deg)
        # corners relative to center (counter-clockwise)
        dx = np.array([-w/2.0, w/2.0, w/2.0, -w/2.0])
        dy = np.array([-h/2.0, -h/2.0, h/2.0, h/2.0])
        xs = cx + dx * np.cos(theta) - dy * np.sin(theta)
        ys = cy + dx * np.sin(theta) + dy * np.cos(theta)
        return list(zip(xs, ys))

    def init_rect_interaction(self):
        """Create handles and connect events for the current rect_patch."""
        if not hasattr(self, 'rect_patch') or self.rect_patch is None:
            return
        ax = self.image_canvas.ax

        # remove old handles
        try:
            for h in getattr(self, 'rect_handles', []):
                h.remove()
        except Exception:
            pass
        self.rect_handles = []

        # get rect properties
        x0, y0 = self.rect_patch.get_xy()
        w = self.rect_patch.get_width()
        h = self.rect_patch.get_height()
        ang = getattr(self.rect_patch, 'angle', 0.0)

        corners = self._rect_corners(x0, y0, w, h, ang)
        # add corner handles
        for (cx, cy) in corners:
            c = mpatches.Circle((cx, cy), radius=max(w, h) * 0.02, facecolor='white', edgecolor='black', zorder=300)
            ax.add_patch(c)
            self.rect_handles.append(c)

        # rotation handle: place above top-center
        top_center = ((corners[2][0] + corners[1][0]) / 2.0, (corners[2][1] + corners[1][1]) / 2.0)
        # offset outward
        vecx = top_center[0] - (x0 + w/2.0)
        vecy = top_center[1] - (y0 + h/2.0)
        norm = np.hypot(vecx, vecy) if (vecx or vecy) else 1.0
        off = 0.1 * max(w, h)
        rx = top_center[0] + (vecx / norm) * off
        ry = top_center[1] + (vecy / norm) * off
        self.rotate_handle = mpatches.Circle((rx, ry), radius=max(w, h) * 0.02, facecolor='yellow', edgecolor='black', zorder=300)
        ax.add_patch(self.rotate_handle)

        # connect events
        canvas = self.image_canvas
        # disconnect previous if any
        for cid in ('_cid_press', '_cid_release', '_cid_motion'):
            if hasattr(self, cid):
                try:
                    canvas.mpl_disconnect(getattr(self, cid))
                except Exception:
                    pass

        self._cid_press = canvas.mpl_connect('button_press_event', self._on_rect_press)
        self._cid_release = canvas.mpl_connect('button_release_event', self._on_rect_release)
        self._cid_motion = canvas.mpl_connect('motion_notify_event', self._on_rect_motion)

        self.interaction_mode = None
        self.interaction_data = {}

    def _distance_display(self, event, x, y):
        """Distance in display coords between event and data point x,y."""
        disp = self.image_canvas.ax.transData.transform((x, y))
        ex, ey = event.x, event.y
        return np.hypot(disp[0] - ex, disp[1] - ey)

    def _on_rect_press(self, event):
        if event.inaxes != self.image_canvas.ax:
            return
        if not hasattr(self, 'rect_patch') or self.rect_patch is None:
            return

        # check rotate handle first
        if self._distance_display(event, *self.rotate_handle.center) < 10:
            self.interaction_mode = 'rotate'
            # store center
            x0, y0 = self.rect_patch.get_xy()
            w = self.rect_patch.get_width(); h = self.rect_patch.get_height()
            cx = x0 + w/2.0; cy = y0 + h/2.0
            self.interaction_data = {'center': (cx, cy), 'start_event': event, 'start_angle': getattr(self.rect_patch, 'angle', 0.0)}
            return

        # check corner handles
        for idx, handle in enumerate(self.rect_handles):
            if self._distance_display(event, *handle.center) < 10:
                self.interaction_mode = 'corner'
                self.interaction_data = {'corner_idx': idx, 'start_event': event}
                return

        # check inside rect polygon
        # build path
        x0, y0 = self.rect_patch.get_xy(); w = self.rect_patch.get_width(); h = self.rect_patch.get_height(); ang = getattr(self.rect_patch, 'angle', 0.0)
        corners = self._rect_corners(x0, y0, w, h, ang)
        from matplotlib.path import Path
        path = Path(corners)
        if path.contains_point((event.xdata, event.ydata)):
            self.interaction_mode = 'move'
            self.interaction_data = {'start_event': event, 'orig_xy': (x0, y0)}
            return

    def _on_rect_motion(self, event):
        if event.inaxes != self.image_canvas.ax:
            return
        if not hasattr(self, 'rect_patch') or self.rect_patch is None:
            return
        if self.interaction_mode is None:
            return

        x0, y0 = self.rect_patch.get_xy(); w = self.rect_patch.get_width(); h = self.rect_patch.get_height(); ang = getattr(self.rect_patch, 'angle', 0.0)

        if self.interaction_mode == 'move':
            se = self.interaction_data['start_event']
            dx = event.xdata - se.xdata
            dy = event.ydata - se.ydata
            new_x = self.interaction_data['orig_xy'][0] + dx
            new_y = self.interaction_data['orig_xy'][1] + dy
            self.rect_patch.set_xy((new_x, new_y))

        elif self.interaction_mode == 'corner':
            idx = self.interaction_data['corner_idx']
            # resize depending on corner idx
            corners = self._rect_corners(x0, y0, w, h, ang)
            # transform event point into rectangle-local coordinates by rotating by -ang around center
            cx = x0 + w/2.0; cy = y0 + h/2.0
            theta = -np.deg2rad(ang)
            ex = event.xdata - cx; ey = event.ydata - cy
            lx = ex * np.cos(theta) - ey * np.sin(theta)
            ly = ex * np.sin(theta) + ey * np.cos(theta)
            # compute new width/height from local coords based on opposite corner
            # map corner index to sign
            signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            sx, sy = signs[idx]
            new_w = abs((lx - (-sx * w/2.0)))
            new_h = abs((ly - (-sy * h/2.0)))
            # ensure min size
            new_w = max(new_w, 1e-6); new_h = max(new_h, 1e-6)
            # compute new lower-left in world coords
            # center stays at cx,cy in this simple resize
            new_x0 = cx - new_w/2.0; new_y0 = cy - new_h/2.0
            self.rect_patch.set_width(new_w); self.rect_patch.set_height(new_h); self.rect_patch.set_xy((new_x0, new_y0))

        elif self.interaction_mode == 'rotate':
            data = self.interaction_data
            cx, cy = data['center']
            # angle between center->start and center->current
            sx = data['start_event'].xdata - cx; sy = data['start_event'].ydata - cy
            ex = event.xdata - cx; ey = event.ydata - cy
            if sx is None or ex is None:
                return
            start_ang = np.arctan2(sy, sx)
            cur_ang = np.arctan2(ey, ex)
            delta = np.rad2deg(cur_ang - start_ang)
            new_ang = (data['start_angle'] + delta) % 360
            self.rect_patch.angle = new_ang

        # update handles positions
        self._update_handles()
        self.image_canvas.draw()

    def _on_rect_release(self, event):
        if self.interaction_mode is None:
            return
        # Just clean up interaction state; don't update region yet
        # Parameters will be extracted when Plot CMD is clicked
        self.interaction_mode = None
        self.interaction_data = {}

    def _extract_current_region(self):
        """Compute region dict from current RectangleSelector state. Returns None if no rectangle."""
        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            try:
                x1, x2, y1, y2 = self.rect_selector.extents
                x0 = min(x1, x2)
                y0 = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                ang = 0.0
            except Exception:
                return None
        elif hasattr(self, 'rect_patch') and self.rect_patch is not None:
            x0, y0 = self.rect_patch.get_xy()
            w = self.rect_patch.get_width()
            h = self.rect_patch.get_height()
            ang = getattr(self.rect_patch, 'angle', 0.0)
        else:
            return None

        # compute center in data coords
        center_data_x = x0 + w / 2.0
        center_data_y = y0 + h / 2.0
        ra_cen = None
        dec_cen = None
        
        if hasattr(self.image_canvas, 'wcs') and self.image_canvas.wcs is not None:
            scales = proj_plane_pixel_scales(self.image_canvas.wcs)
            try:
                scales = [s if s is not None else 1.0 * u.dimensionless_unscaled for s in scales]
            except Exception:
                scales = [1.0 * u.dimensionless_unscaled, 1.0 * u.dimensionless_unscaled]

            if w < 1.0:
                w_deg_q = (w * u.deg)
                h_deg_q = (h * u.deg)
            else:
                try:
                    w_deg_q = (w * scales[0]).to(u.deg)
                except Exception:
                    w_deg_q = (w * getattr(scales[0], 'value', scales[0])) * u.deg
                try:
                    h_deg_q = (h * scales[1]).to(u.deg)
                except Exception:
                    h_deg_q = (h * getattr(scales[1], 'value', scales[1])) * u.deg

            w_arc = float(w_deg_q.to(u.arcsec).value)
            h_arc = float(h_deg_q.to(u.arcsec).value)

            # determine RA/Dec for center
            try:
                if 0.0 <= center_data_x <= 360.0 and -90.0 <= center_data_y <= 90.0:
                    ra_cen = float(center_data_x)
                    dec_cen = float(center_data_y)
                else:
                    try:
                        radec = self.image_canvas.wcs.wcs_pix2world(center_data_x, center_data_y, 0)
                        ra_cen = float(radec[0])
                        dec_cen = float(radec[1])
                    except Exception:
                        radec_arr = self.image_canvas.wcs.all_pix2world([[center_data_x, center_data_y]], 0)
                        ra_cen = float(radec_arr[0][0])
                        dec_cen = float(radec_arr[0][1])
                if not (0.0 <= ra_cen <= 360.0 and -90.0 <= dec_cen <= 90.0):
                    ra_cen = None
                    dec_cen = None
            except Exception:
                ra_cen = None
                dec_cen = None
        else:
            w_arc = w
            h_arc = h

        return {
            'spatial_filter': 'box',
            'width': float(w_arc),
            'height': float(h_arc),
            'ang': float(ang),
            'center_data': (float(center_data_x), float(center_data_y)),
            'ra_cen': ra_cen,
            'dec_cen': dec_cen,
        }

    def _update_handles(self):
        if not hasattr(self, 'rect_patch') or self.rect_patch is None:
            return
        x0, y0 = self.rect_patch.get_xy(); w = self.rect_patch.get_width(); h = self.rect_patch.get_height(); ang = getattr(self.rect_patch, 'angle', 0.0)
        corners = self._rect_corners(x0, y0, w, h, ang)
        for handle, (cx, cy) in zip(self.rect_handles, corners):
            try:
                handle.set_center((cx, cy))
            except Exception:
                handle.center = (cx, cy)
            handle.set_visible(True)
        # update rotate handle
        top_center = ((corners[2][0] + corners[1][0]) / 2.0, (corners[2][1] + corners[1][1]) / 2.0)
        vecx = top_center[0] - (x0 + w/2.0)
        vecy = top_center[1] - (y0 + h/2.0)
        norm = np.hypot(vecx, vecy) if (vecx or vecy) else 1.0
        off = 0.1 * max(w, h)
        rx = top_center[0] + (vecx / norm) * off
        ry = top_center[1] + (vecy / norm) * off
        try:
            self.rotate_handle.set_center((rx, ry))
        except Exception:
            self.rotate_handle.center = (rx, ry)


# ==========================================================
# MAIN
# ==========================================================

def main():

    app = QApplication(sys.argv)

    win = StellarPopulationGUI()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()