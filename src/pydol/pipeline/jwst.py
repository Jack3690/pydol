from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
import jwst.associations

from glob import glob

import os
from crds import client
import jwst
import multiprocessing as mp
from pathlib import Path
    
client.set_crds_server("https://jwst-crds.stsci.edu")

class jpipe():
    def __init__(self, input_files=[], out_dir='.', filter='',
                 crds_context="jwst_1241.pmap", crds_dir='.', n_cores=None,
                 **kwargs):
        """
            Parameters
            ----------
            input_files: list,
                         Input list of level 0 '_uncal.fits' files.
                         Recommended: /data/stage0/
            out_dir: str,
                     Output directory.
                     Recommended: The directory that contains /data/stage0/
                     Pipeline will create /data/stage1/ and /data/stage2/

            crds_context: str,
                          Reference context for JWST pipeline from CRDS.
              Returns
              -------
                  None

        """
        self.config = {'1byf_corr' : False,
                       'snowball_corr' : True}
        self.config.update(kwargs)
        self.filter_name=filter
        if n_cores is None or n_cores > mp.cpu_count()-1:
            self.n_cores = mp.cpu_count()-1
        else:
            self.n_cores = n_cores
            
        if os.access(crds_dir,os.W_OK):
            os.makedirs(crds_dir, exist_ok=True)
        else:
            raise Exception(f"{crds_dir} is not WRITABLE")
            
        os.environ['CRDS_PATH'] = str(crds_dir)
        os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
        if len(input_files)<1:
            raise Exception("Input files list CANNOT be empty!")
        self.input_files = input_files
        self.out_dir = out_dir
        if os.access(out_dir,os.W_OK):
            os.makedirs(out_dir + '/stage1/', exist_ok=True)
            os.makedirs(out_dir + '/stage2/', exist_ok=True)
            os.makedirs(out_dir + '/stage3/', exist_ok=True)
        else:
            raise Exception(f"{out_dir} is not WRITABLE")

        os.environ["CRDS_CONTEXT"] = crds_context

    def stage1_pipeline(self, filename):
        """
            Parameters
            ----------
            filename: str,
                      path to the level 0 "_uncal.fits" file
            Returns
            -------
                None
        """
        # Instantiate the pipeline
        img1 = Detector1Pipeline()   
        # Snowball Removal (M82 Group)
        img1.jump.expand_large_events = self.config['snowball_corr']
        # 1/f noise correction
        img1.clean_flicker_noise.skip = self.config['1byf_corr']       
        # Specify where the output should go
        img1.output_dir = self.out_dir + '/stage1/'
        # Save the final resulting _rate.fits files
        img1.save_results = True
        #No of cores
        img1.jump.maximum_cores = f'{self.n_cores}'
        # Run the pipeline on an input list of files
        img1(filename)

    def stage2_pipeline(self, filename):
        """
            Parameters
            ----------
            filename: str,
                      path to the level 1 "_rate.fits" file
            Returns
            -------
                None
        """
        # Instantiate the pipeline
        img2 = Image2Pipeline()
        # Specify where the output should go
        img2.output_dir = self.out_dir + '/stage2/'
        # Save the final resulting _rate.fits files
        img2.save_results = True
        # Run the pipeline on an input list of files
        img2(filename)

        
    def stage3_pipeline(self, filenames):
        """
            Parameters
            ----------
            filename: str,
                      list of paths to the level 2 "_cal.fits" files
                      
                      if a single file is provided only 
                      resample and source_catalog steps will be applied.
            Returns
            -------
                None
        """
        # Instantiate the pipeline
        img3 = Image3Pipeline()
        # Specify where the output should go
        img3.output_dir = self.out_dir + '/stage3/'
        # Save the final resulting _rate.fits files
        img3.save_results = True
        # Output file name
        img3.output_file = self.filter_name
        # Run the pipeline on an input list of files
        img3(filenames)

    def __call__(self):
        """
            Runs the JWST Stage 1, Stage 2, and Stage 3 pipeline for generating
            '_crf.fits' files
        """
        # Stage1
        uncal_files = [i for i in self.input_files if 'uncal' in i ]
        [ self.stage1_pipeline(f) for f in uncal_files if not os.path.exists(f.replace('stage0', 'stage1').replace('uncal', 'rate')) ]

        rate_files = [f.replace('stage0', 'stage1').replace('uncal', 'rate') for f in uncal_files]

        # Stage 2
        rate_files_ = [f for f in rate_files if not os.path.exists(f.replace('stage1', 'stage2').replace('rate', 'cal'))]
        
        if len(rate_files_)>0:
            if self.n_cores>1:
                with mp.Pool(self.n_cores) as p:
                    p.map(self.stage2_pipeline, rate_files_)
            else:
                for f in rate_files_:
                    self.stage2_pipeline(f)              
        # Stage 3
        cal_files = [f.replace('stage1', 'stage2').replace('rate', 'cal') for f in rate_files]
        cal_files_ = [f for f in cal_files if not os.path.exists(f.replace('stage2', 'stage3').replace('cal', 'crf'))]

        if len(cal_files_) > 0:
            self.stage3_pipeline(cal_files)
