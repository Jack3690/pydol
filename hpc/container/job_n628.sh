#!/usr/bin/bash
#SBATCH --job-name=j_ngc628
#SBATCH --output=out_ngc628.txt
#SBATCH --error=err_ngc628.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10-00:00:00
#SBATCH --partition=ia

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ulimit -s unlimited
srun --propagate=STACK ./run_container.sh python run_SFH.py
