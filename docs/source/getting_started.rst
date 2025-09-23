=================
Quickstart Guide 
=================

The following is a quickstart guide to installing and running PyDOL. Currently PyDOL supports JWST/NIRCam, HST/ACS, and HST/WFC3

In short, PyDOL is a simple Python wrapper for DOLPHOT which does the following steps
1) Generates parameter file based on user list of files.
2) Generates sky image.
3) Runs the masking routine (NIRCAMMASK, ACSMASK, WFC3MASK)
4) Runs DOLPHOT for crowd field simulataneous PSF photometry (Parallel mode available)
5) Generates FITS tables with quallity cuts applied for selecting only good stars.


**1. Install DOLPHOT 2.0**

Before running PyDOL, you need to have DOLPHOT installed. See `Installation <https://pydol.readthedocs.io/en/latest/installation.html>`_ for detailed instructions.

**2. Create a conda environment**
.. code-block:: bash
    $ conda create --name dolphot python=3.11

**3. Install PyDOL from GitHub**

.. code-block:: bash

  (dolphot)$ pip install git+https://github.com/Jack3690/pydol