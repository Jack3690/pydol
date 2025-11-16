from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
import jwst.associations

from glob import glob
import os
import multiprocessing as mp
from pathlib import Path


class jpipe():
    def __init__(self, input_files=[], out_dir='.', filter='',
                 crds_context="jwst_1241.pmap", crds_dir='.', n_cores=None,
                 **kwargs):

        # Default custom configuration
        self.config = {
            'corr_1byf': False,
            'corr_snowball': True,
            'fit_by_channel': False,
            'background_method': 'median',
        }
        self.config.update(kwargs)
        self.filter_name = filter

        # Core count
        if n_cores is None or n_cores > mp.cpu_count() - 1:
            self.n_cores = mp.cpu_count() - 1
        else:
            self.n_cores = n_cores

        # CRDS path setup
        if os.access(crds_dir, os.W_OK):
            os.makedirs(crds_dir, exist_ok=True)
        else:
            raise Exception(f"{crds_dir} is not WRITABLE")

        os.environ['CRDS_PATH'] = str(crds_dir)
        os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
        os.environ["CRDS_CONTEXT"] = crds_context

        # Input files
        if len(input_files) < 1:
            raise Exception("Input files list CANNOT be empty!")
        self.input_files = input_files

        # Output directories
        self.out_dir = out_dir
        if os.access(out_dir, os.W_OK):
            os.makedirs(out_dir + '/stage1/', exist_ok=True)
            os.makedirs(out_dir + '/stage2/', exist_ok=True)
            os.makedirs(out_dir + '/stage3/', exist_ok=True)
        else:
            raise Exception(f"{out_dir} is not WRITABLE")

    # ---------------- STAGE 1 ---------------- #

    def stage1_pipeline(self, filename):

        self.stage1 = Detector1Pipeline()

        # Snowball removal
        self.stage1.jump.expand_large_events = self.config['corr_snowball']

        # 1/f noise
        self.stage1.clean_flicker_noise.skip = not self.config['corr_1byf']
        self.stage1.clean_flicker_noise.fit_by_channel = self.config['fit_by_channel']
        self.stage1.clean_flicker_noise.background_method = self.config['background_method']

        # Output
        self.stage1.output_dir = self.out_dir + '/stage1/'
        self.stage1.save_results = True

        # Parallel cores
        self.stage1.jump.maximum_cores = f'{self.n_cores}'

        self.stage1.call(filename, save_results=True)

    # ---------------- STAGE 2 ---------------- #

    def stage2_pipeline(self, filename):

        self.stage2 = Image2Pipeline()
        self.stage2.output_dir = self.out_dir + '/stage2/'
        self.stage2.save_results = True
        self.stage2.call(filename, save_results=True)

    # ---------------- STAGE 3 ---------------- #

    def stage3_pipeline(self, filenames):

        self.stage3 = Image3Pipeline()
        self.stage3.output_dir = self.out_dir + '/stage3/'
        self.stage3.save_results = True
        self.stage3.output_file = self.filter_name

        self.stage3.call(filenames, save_results=True)

    # ---------------- MASTER CALL ---------------- #

    def __call__(self):

        # ------- Stage 1 ------- #
        uncal_files = [i for i in self.input_files if 'uncal' in i]

        for f in uncal_files:
            rate_f = f.replace('stage0', 'stage1').replace('uncal', 'rate')
            if not os.path.exists(rate_f):
                self.stage1_pipeline(f)

        rate_files = [f.replace('stage0', 'stage1').replace('uncal', 'rate')
                      for f in uncal_files]

        # ------- Stage 2 ------- #
        rate_files_to_run = [
            f for f in rate_files
            if not os.path.exists(f.replace('stage1', 'stage2').replace('rate', 'cal'))
        ]

        # STScI recommends NOT using multiprocessing for Stage 2
        for f in rate_files_to_run:
            self.stage2_pipeline(f)

        # ------- Stage 3 ------- #
        cal_files = [f.replace('stage1', 'stage2').replace('rate', 'cal')
                     for f in rate_files]

        # JWST 1.20.2: Stage 3 imaging output = *_i2d.fits
        cal_files_to_run = [
            f for f in cal_files
            if not os.path.exists(f.replace('stage2', 'stage3').replace('cal', 'i2d'))
        ]

        if len(cal_files_to_run) > 0:
            self.stage3_pipeline(cal_files)
