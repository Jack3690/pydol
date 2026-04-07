from glob import glob

import os
from crds import client
import jwst
import multiprocessing as mp
from pathlib import Path
import subprocess

client.set_crds_server("https://jwst-crds.stsci.edu")

os.environ['CRDS_PATH'] = '/zfs-home/202404072C/JWST/CRDS/'
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
os.environ["CRDS_CONTEXT"] = "jwst_1464.pmap"

input_files = glob(f'/zfs-home/202404072C/JWST/data/m82n/data/stage0/F115W/*')
input_files += glob(f'/zfs-home/202404072C/JWST/data/m82n/data/stage0/F200W/*')

for f in input_files:
        cmd = f"""crds bestrefs --update-bestrefs --sync-reference 1 --files {f} --new-context {os.environ["CRDS_CONTEXT"]}"""
        subprocess.run(cmd, shell=True)
