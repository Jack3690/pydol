import bayestar as bst
import pandas as pd
import os
import numpy as np
data_file = bst.data_dir / 'test_files/data.test'

df = pd.read_fwf(data_file, sep= ' ').drop(columns = '#')
col_dict = {'F435Wmag'     : 'fw1',
            'F435Wmag_err' : 'fw1_error',
            'F555Wmag'     : 'fw2',
            'F555Wmag_err' : 'fw2_error',
            'F814Wmag'     : 'fw3',
            'F814Wmag_err' : 'fw3_error',}

df = df.rename(columns = col_dict)

sfh = bst.SFH(df, parallel=True)

fname = sfh()

if os.path.exists(fname):
    df_out = pd.read_csv(fname)
    if np.all(df_out.keys()==['Z', 'Log_age', 'p10', 'p50', 'p90']):
        print("Tests Completed Successfully!")
    else:
        print(df_out.keys())
else:
    print("Tests Failed")
