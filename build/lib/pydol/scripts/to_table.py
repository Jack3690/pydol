from astropy.table import Table
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DOLPHOT Output to Table')
	parser.add_argument("--f", dest='filename', default='out', type = str, help='Photometry')
	parser.add_argument("--t", dest='format', default='fits', type = str, help="'csv' or 'fits'")
	options = parser.parse_args()
  
	col = ['ext','chip','x','y','chi_fit','obj_SNR','obj_sharpness','obj_roundness',
	'dir_maj_axis','obj_crowd','type','flux','sky','fluxr_n','fluxr_n_err','mag_vega',
	'mag_trans','mag_err','chi','SNR','sharpness','roundness','crowd','flags']
	
	with open(options.filename) as f:
		dat = f.readlines()
	data = np.array([i.split() for i in dat]).astype(float)
		
	df = pd.DataFrame(data,columns=col)
	
	filename = os.path.split(options.filename)[0]
	if options.format == 'csv':
		df.to_csv(f'{filename}/photometry.csv')
	elif options.format == 'fits':
		tab = Table.from_pandas(df)
		tab.write(f'{filename}/photometry.fits', overwrite=True)
