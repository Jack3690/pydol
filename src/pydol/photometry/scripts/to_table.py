from astropy.table import Table
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DOLPHOT Output to Table')
	parser.add_argument("--f", dest='filename', default='out', type = str, help='Photometry')
	parser.add_argument("--n", dest='n', default='0', type = int, help='Photometry')
	parser.add_argument("--t", dest='format', default='fits', type = str, help="'csv' or 'fits'")
	parser.add_argument("--o", dest='out', default='photometry', type = str, help="Output filename")
	options = parser.parse_args()
	n = options.n
	out = options.out
	col = ['ext','chip','x','y','chi_fit','obj_SNR','obj_sharpness','obj_roundness',
		'dir_maj_axis','obj_crowd','type','counts_tot','sky_tot','count_rate','count_rate_err','mag_vega',
		'mag_ubvri','mag_err','chi','SNR','sharpness','roundness','crowd','flags']

	col_t = ['counts_measured', 'sky_measured', 'count_rate','count_rate_err','mag_vega',
			'mag_ubvri','mag_err','chi','SNR','sharpness','roundness','crowd','flags']

	#for i in range(n):
	#	for j in col_t:
	#		col +=  [j + '_' + str(i+1)]
	
	with open(options.filename) as f:
		dat = f.readlines()
	data = np.array([i.split()[:len(col)] for i in dat]).astype(float)
		
	df = pd.DataFrame(data,columns=col)
	
	filename = os.path.split(options.filename)[0]
	if options.format == 'csv':
		df.to_csv(f'{filename}/{out}.csv')
	elif options.format == 'fits':
		tab = Table.from_pandas(df)
		tab.write(f'{filename}/{out}.fits', overwrite=True)
