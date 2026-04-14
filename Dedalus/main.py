import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pathlib
import subprocess
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import fncs

# Clean up any old files
import shutil
shutil.rmtree('analysis', ignore_errors=True)

######################## Main ###################################

# Parameters
Lx, Ly = 1, 0
# Lz = [1, 1.5, 1.9, 1.99, 2]
Lz = 1
Nx, Ny = 500, 0
# Nz = [128, 256, 256, 256, 256]
Nz = 500
# F = np.logspace(3, 6, 10)
F = 1e5
Schmidt = 1
dealias = 3/2
max_timestep = 1e-7
stop_sim_time = [500000 * max_timestep]
timestepper = d3.RK222
dtype = np.float64

for i in range(len(stop_sim_time)):
    fncs.simul(Lx, Lz, Nx, Nz, F, dealias, stop_sim_time[i], timestepper, max_timestep, dtype, Ly, Ny, i, Schmidt)  ## don't forget i
