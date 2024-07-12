from astroquery.mast import Observations
import os

def test_data_access():

    #Testing MAST server access

    #Example data for M83

    mast_dir = 'mast:jwst/product' # Download from MAST
    data_dir = './data/stage0/'  # save downloaded data
    os.makedirs(data_dir, exist_ok=True)

    ext = 'uncal'
    # JWST images to be analyzed
    image_files = ['jw01783001001_03107_00001_nrca1_uncal.fits', 
                   'jw01783001001_03107_00001_nrca2_uncal.fits' ]


    for image_file in image_files:
        # Download file (if not already downloaded)
        mast_path  = os.path.join(mast_dir, image_file)
        local_path = os.path.join(data_dir, image_file)
        Observations.download_file(mast_path, local_path=local_path)

    a = os.path.exists('./data/stage0/jw01783001001_03107_00001_nrca1_uncal.fits')
    b = os.path.exists('./data/stage0/jw01783001001_03107_00001_nrca2_uncal.fits')

    assert a==b==True
    

