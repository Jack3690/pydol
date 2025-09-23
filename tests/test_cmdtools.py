import pytest
from astropy.table import Table
import matplotlib.pyplot as plt
from pydol.photometry.scripts.cmdtools import gen_CMD

@pytest.fixture
def tab():
        return Table.read('tests/test_data/test_data.fits')

@pytest.fixture
def fig_ax():
        fig, ax = plt.subplots(figsize=(12,10))
        yield fig, ax
        plt.close(fig)

@pytest.fixture
def common_params():
        ra_cen = 202.4036116426457
        dec_cen = 47.17577052288809
        filters = {'filt1':'f115w', 'filt2':'f200w', 'filt3':'f200w'}
        positions = {'ra_col': 'ra', 'dec_col' : 'dec', 'ra_cen': ra_cen, 'dec_cen': dec_cen}
        region = {'width_in': 0, 'height_in': 0, 'width_out': 24, 'height_out': 24, 'ang': 131.67459, 'spatial_filter': 'box'}
        extinction = {'Av': 0.095, 'Av_x': 4, 'Av_y': 18, 'Av_': 2}
        axis_limits = {'xlims': [-1, 5], 'ylims': [16.5, 28]}
        error_settings = {'ref_xpos': -0.5, 'mag_err_lim': 0.2}
        plot_settings = {'s':2, 'legend.ncols':2, 'alpha':0.7, 'lw':3}
        return filters, positions, region, extinction, axis_limits, error_settings, plot_settings

def test_gen_CMD_basic(tab, fig_ax, common_params):
        filters, positions, region, extinction, axis_limits, error_settings, plot_settings = common_params
        isochrone_params = {'met': [0.02], 'label_min': 0, 'label_max': 10, 'ages': [7,7.4,7.7,8]}
        fig, ax = fig_ax
        fig, ax, tab1 = gen_CMD(tab, None, filters, positions, region, extinction, 29.67, axis_limits,
                                                        isochrone_params, plot_settings=plot_settings, error_settings=error_settings,
                                                        other_settings={'ab_dist': False, 'skip_data': False}, fig=fig, ax=ax)
        assert isinstance(tab1, Table)
        assert len(tab1) > 0

def test_gen_CMD_skip_data(tab, fig_ax, common_params):
        filters, positions, region, extinction, axis_limits, error_settings, plot_settings = common_params
        isochrone_params = {'met': [0.002], 'label_min': 0, 'label_max': 10, 'ages': [9.0, 9.4, 10.]}
        fig, ax = fig_ax
        fig, ax, tab1 = gen_CMD(tab, None, filters, positions, region, extinction, 29.67, axis_limits,
                                                        isochrone_params, plot_settings=plot_settings, error_settings=error_settings,
                                                        other_settings={'ab_dist': True, 'skip_data': True}, fig=fig, ax=ax)
        assert isinstance(tab1, Table)
        assert len(tab1) == 10305

def test_gen_CMD_different_region(tab, fig_ax, common_params):
        filters, positions, region, extinction, axis_limits, error_settings, plot_settings = common_params
        region['spatial_filter'] = 'circle'
        isochrone_params = {'met': [0.02], 'label_min': 0, 'label_max': 10, 'ages': [8.5, 9.0]}
        fig, ax = fig_ax
        fig, ax, tab1 = gen_CMD(tab, None, filters, positions, region, extinction, 29.67, axis_limits,
                                                        isochrone_params, plot_settings=plot_settings, error_settings=error_settings,
                                                        other_settings={'ab_dist': False, 'skip_data': False}, fig=fig, ax=ax)
        assert isinstance(tab1, Table)
        assert len(tab1) > 0

def test_gen_CMD_extinction_variation(tab, fig_ax, common_params):
        filters, positions, region, extinction, axis_limits, error_settings, plot_settings = common_params
        extinction['Av'] = 0.5
        isochrone_params = {'met': [0.02], 'label_min': 0, 'label_max': 10, 'ages': [7, 8, 9]}
        fig, ax = fig_ax
        fig, ax, tab1 = gen_CMD(tab, None, filters, positions, region, extinction, 29.67, axis_limits,
                                                        isochrone_params, plot_settings=plot_settings, error_settings=error_settings,
                                                        other_settings={'ab_dist': False, 'skip_data': False}, fig=fig, ax=ax)
        assert isinstance(tab1, Table)
        assert len(tab1) > 0

def test_gen_CMD_error_settings(tab, fig_ax, common_params):
        filters, positions, region, extinction, axis_limits, error_settings, plot_settings = common_params
        error_settings['mag_err_lim'] = 0.05
        isochrone_params = {'met': [0.02], 'label_min': 0, 'label_max': 10, 'ages': [7, 8]}
        fig, ax = fig_ax
        fig, ax, tab1 = gen_CMD(tab, None, filters, positions, region, extinction, 29.67, axis_limits,
                                                        isochrone_params, plot_settings=plot_settings, error_settings=error_settings,
                                                        other_settings={'ab_dist': False, 'skip_data': False}, fig=fig, ax=ax)
        assert isinstance(tab1, Table)
        assert len(tab1) > 0