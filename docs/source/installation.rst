====================
Installation
====================


DOLPHOT 2.0
###########

##Compilation
You can download DOLPHOT 2.0 source code and JWST/NIRCAM, HST/ACS, and HST/WFC3 modules from http://americano.dolphinsim.com/dolphot/nircam.html

For installing particular DOLPHOT modules, you can modify the Makefile to uncomment the following lines 

.. code-block:: bash

  export USENIRCAM=1 
  export USEACS=1 
  export USEWFC3=1

For altering the default maximum number of images and number of stars supported by DOLPHOT you can use can modify the values for 'DMAXNIMG' variable and 'DMAXNSTARS' variable.

.. code-block:: bash

  DMAXNIMG=500
  DMAXNSTARS=5000000

##Filters
If you prefer to use the Vega zero points instead of the new Vega_Sirius zero points for JWST/NIRCam. You can go to dolphot2.0/nircam/data and rename the filters.dat as filters_vega_sirius. dat and rename filters_vega.dat to filters.dat

Next, you can download and extract the necessary filters you would use for photometry from the same DOLPHOT website.

For HST modules, you also need to download the pixel area maps.

##PATH
Finally, add 'dolphot2.0/bin' to PATH. Note that this PATH variable should be accessible by your Python environment.










