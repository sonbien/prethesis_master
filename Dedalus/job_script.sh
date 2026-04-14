#! /bin/bash

#SBATCH -p medium
#SBATCH -N 1
#SBATCH -n 95
#SBATCH --ntasks-per-node=95
#SBATCH -t 48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniforge3
module load gcc/11.5.0
module load openmpi/4.1.7
module load fftw
module load hdf5
source activate ~/.conda/envs/dedalus3_test16
mpirun -n 95 python3 ~/code/main.py
