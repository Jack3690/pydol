from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
import jwst.associations

from glob import glob

import os
from crds import client
import jwst
import multiprocessing as mp
from pathlib import Path

crds_dir = Path(__file__).parent.joinpath('CRDS')/'crds_cache'
os.makedirs(crds_dir, exist_ok=True)
os.environ['CRDS_PATH'] = str(crds_dir)
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
client.set_crds_server("https://jwst-crds.stsci.edu")

class jpipe():
    def __init__(self, input_files='.', out_dir='.',
                 crds_context="jwst_1200.pmap"):
        """
            Parameters
            ----------
        """
        if len(input_files)<1:
            raise Exception("Input files list CANNOT be empty!")
        self.input_files = input_files
        self.out_dir = out_dir
        os.makedirs(out_dir + '/data/stage1/', exist_ok=True)
        os.makedirs(out_dir + '/data/stage2/', exist_ok=True)
        os.environ["CRDS_CONTEXT"] = crds_context

    def stage1(self, filename):
        # Instantiate the pipeline
        img1 = Detector1Pipeline()
        # Specify where the output should go
        img1.output_dir = self.out_dir + '/data/stage1/'
        # Save the final resulting _rate.fits files
        img1.save_results = True
        #No of cores
        img1.jump.maximum_cores = f'{mp.cpu_count()-1}'
        # Run the pipeline on an input list of files
        img1(filename)
        
    def stage2(self, filename):
        # Instantiate the pipeline
        img2 = Image2Pipeline()
        # Specify where the output should go
        img2.output_dir = self.out_dir + '/data/stage2/'
        # Save the final resulting _rate.fits files
        img2.save_results = True
        # Run the pipeline on an input list of files
        img2(filename)

    def __call__(self):
        uncal_files = [i for i in self.input_files if 'uncal' in i ]
        for f in uncal_files:
            o = f.replace('stage0','stage1')
            o = o.replace('uncal','rate')
            if not os.path.exists(o):
                self.stage1(f)

        rate_files = glob(self.out_dir + '/data/stage2/*_rate.fits')
        rate_files_ = []
        for f in rate_files:
            o = f.replace('stage1','stage2')
            o = o.replace('rate','cal')
            if not os.path.exists(o):
                rate_files_.append(f)
            
        if len(rate_files_)>0:
            with mp.Pool(mp.cpu_count()-1) as p:
                p.map(self.stage2, rate_files_)


