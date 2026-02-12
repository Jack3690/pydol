from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
import os
import multiprocessing as mp
from pathlib import Path
from crds import client


class JPipe:

    def __init__(
        self,
        input_files,
        out_dir=".",
        filter_name="",
        crds_context="jwst_1241.pmap",
        crds_dir=".",
        n_cores=None,
        **kwargs,
    ):

        # ---------------- Configuration ---------------- #

        self.config = {
            "corr_1byf": False,
            "corr_snowball": True,
            "fit_by_channel": False,
            "background_method": "median",
        }
        self.config.update(kwargs)

        self.filter_name = filter_name

        # Core handling (STScI recommends controlled usage)
        available_cores = mp.cpu_count()
        if n_cores is None:
            self.n_cores = available_cores
        else:
            self.n_cores = min(n_cores, available_cores)

        # ---------------- CRDS Setup ---------------- #

        crds_dir = Path(crds_dir)
        crds_dir.mkdir(parents=True, exist_ok=True)

        os.environ["CRDS_PATH"] = str(crds_dir)
        os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
        os.environ["CRDS_CONTEXT"] = crds_context

        client.set_crds_server("https://jwst-crds.stsci.edu")

        # ---------------- Input Files ---------------- #

        if not input_files:
            raise ValueError("Input files list CANNOT be empty!")

        self.input_files = input_files

        # ---------------- Output Directories ---------------- #

        self.out_dir = Path(out_dir)
        (self.out_dir / "stage1").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "stage2").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "stage3").mkdir(parents=True, exist_ok=True)

    # ==========================================================
    #                       STAGE 1
    # ==========================================================

    def stage1_pipeline(self, filename):

        # STScI-supported internal parallelization
        steps_stage1 = {
            "jump": {
                "expand_large_events": self.config["corr_snowball"],
                "maximum_cores": "all", 
            },
            "ramp_fit": {
                "maximum_cores": "all",   
            },
            "clean_flicker_noise": {
                "skip": not self.config["corr_1byf"],
                "fit_by_channel": self.config["fit_by_channel"],
                "background_method": self.config["background_method"],
            },
        }

        Detector1Pipeline.call(
            filename,
            output_dir=str(self.out_dir / "stage1"),
            save_results=True,
            steps=steps_stage1,
        )

    # ==========================================================
    #                       STAGE 2
    # ==========================================================

    def stage2_pipeline(self, filename):

        # Sequential execution (STScI recommendation)
        Image2Pipeline.call(
            filename,
            output_dir=str(self.out_dir / "stage2"),
            save_results=True,
        )

    # ==========================================================
    #                       STAGE 3
    # ==========================================================

    def stage3_pipeline(self, filenames):

        # Stage 3 is memory heavy — run once per association
        Image3Pipeline.call(
            filenames,
            output_file=self.filter_name,
            output_dir=str(self.out_dir / "stage3"),
            save_results=True,
        )

    # ==========================================================
    #                       MASTER CALL
    # ==========================================================

    def __call__(self):

        # ---------------- Stage 1 ---------------- #

        uncal_files = [f for f in self.input_files if "uncal" in f]

        rate_files = []

        for f in uncal_files:

            rate_f = f.replace("stage0", "stage1").replace("uncal", "rate")
            rate_files.append(rate_f)

            if not Path(rate_f).exists():
                self.stage1_pipeline(f)

        # ---------------- Stage 2 ---------------- #

        rate_files_to_run = [
            f for f in rate_files
            if not Path(f.replace("stage1", "stage2").replace("rate", "cal")).exists()
        ]

        # STScI recommends NOT using multiprocessing for Stage 2
        for f in rate_files_to_run:
            self.stage2_pipeline(f)

        # ---------------- Stage 3 ---------------- #

        cal_files = [
            f.replace("stage1", "stage2").replace("rate", "cal")
            for f in rate_files
        ]

        if cal_files:
            self.stage3_pipeline(cal_files)
