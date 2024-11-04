from astropy.table import Table
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DOLPHOT Output to Table')
	parser.add_argument("--f", dest='filename', default='out', type = str, help='Photometry')
	parser.add_argument("--c", dest='columns', default='out', type = str, help='DOLPHOT columns')
	parser.add_argument("--d", dest='detector', default='NIRCAM', type = str, help='detector')
	parser.add_argument("--t", dest='format', default='fits', type = str, help="'csv' or 'fits'")
	parser.add_argument("--o", dest='out', default='photometry', type = str, help="Output filename")
	options = parser.parse_args()
	out = options.out
	
	with open(options.columns) as f:
		cols = f.readlines()
		for n, i in enumerate(cols):
			if 'Measured' in i:
				n_filt = (n - 11)//13
				break
	
	filts = [cols[11+ i*13].split(f'{options.detector.upper()}_')[-1][:-1] for i in range(n_filt)]
	col_source = ['ext','chip','x','y','chi_fit','obj_SNR','obj_sharpness','obj_roundness','dir_maj_axis','obj_crowd','type',]
	col_filt = ['counts_tot','sky_tot','count_rate','count_rate_err','mag_vega','mag_ubvri','mag_err','chi','SNR','sharpness','roundness','crowd','flags']
	
	cols_inp = ['ext_inp','chip_inp','x_inp','y_inp']

	for j in filts:
	    cols_inp.append(f'mag_vega_{j.upper()}_inp')
	    
	cols_out =  col_source.copy()
	for i in filts:
	    for j in col_filt:
	        cols_out.append(j + '_' + i)
	        
	tot_cols = cols_inp  + cols_out
	
	with open(options.filename) as f:
		dat = f.readlines()

	start = 4 + len(filts)*16
	end = start + len(cols_out)
	
	dat = np.array([i.split()[:end] for i in dat]).astype(float)
	mag_index = np.arange(5,len(filts)*16,16).astype(int)
	data = np.concatenate([dat[:,:4], dat[:,mag_index], dat[:,start:end]], axis=1)
	
	df = pd.DataFrame(data,columns=tot_cols)

	filename = os.path.split(options.filename)[0]
	if options.format == 'csv':
		df.to_csv(f'{filename}/{out}.csv')
	elif options.format == 'fits':
		tab = Table.from_pandas(df)
		tab.write(f'{filename}/{out}.fits', overwrite=True)
