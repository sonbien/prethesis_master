import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pathlib
import subprocess
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def visitor(name, obj):
    visitor.counter += 1
    if visitor.counter == target_index[0]:
        saved_name[0] = name
    elif visitor.counter == target_index[1]:
        saved_name[1] = name
    # elif visitor.counter == target_index[2]:
        # saved_name[2] = name


def read_file(path, target_index, saved_name):
    visitor.counter = -1  # start at -1 so first increment puts it at 0
    
    with h5py.File(path, "r") as f:
        f.visititems(visitor)
    
    with h5py.File(path, mode = 'r') as file:
        E_k = file['tasks']['kinetic energy'][:]
        epsilon_v = file['tasks']['dissipation'][:]
        t = file['scales/sim_time'][:]
        x = file['%s' %(saved_name[0])][:]
        z = file['%s' %(saved_name[1])][:]
        # y = file['%s' %(saved_name[1])][:]
        # z = file['%s' %(saved_name[2])][:]
        u = file["tasks"]["velocity"][:]
        # f = file["tasks"]["force"][:]
        # p = file["tasks"]["pressure"][:]

    return E_k, t, x, z, u

##################################### Main ###############################################
E_ks = []
ts = []
E_k_tavgs = []
epsilon_v_tavgs = []
E_k_tavg_rmss = []
rhss = []
us = []

###  Investigate the time

e = 2 * np.pi
Lx = 1
nu = 1
F = [1e3, 3e3, 5e3, 7e3, 1e4, 2e4, 3e4, 1e5, 2e5, 5e5, 1e6]
# target_index = [7, 8, 9]
target_index = [7,8]
# saved_name = [None, None, None]
saved_name = [None, None]
max_timestep = np.array([1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-8])
multiplier = np.array([20000, 100000, 100000, 100000, 100000, 100000, 1000000, 1500000, 3000000, 3000000, 6500000])
stop_sim_time = multiplier * max_timestep
names = [0, 8.006, 8.517, 8.854, 1, 9.903, 10.309, 2, 12.206, 13.122, 3]
t_stat = 80

spec_paths = ["code/snapshots_timetest1_Fe3_0/snapshots_timetest1_Fe3_0_s1.h5", 
              "code/snapshots_timetest2_Fe4_0/snapshots_timetest2_Fe4_0_s1.h5", 
              "code/snapshots_timetest1_Fe5_0/snapshots_timetest1_Fe5_0_s1.h5", 
              "code/snapshots_timetest2_Fe61_0/snapshots_timetest2_Fe61_0_s1.h5"]

for i in range(len(F)):
    if names[i] == 0 or names[i] == 1 or names[i] == 2 or names[i] == 3:
        path = spec_paths[names[i]]
    elif i > 5:
        path = "code/snapshots_timetest2_%.3f/snapshots_timetest2_%i_s1.h5" %(names[i], names[i])
    else:
        path = "code/snapshots_timetest1_%.3f/snapshots_timetest1_%i_s1.h5" %(names[i], names[i])

    E_k, t, x, z, u = read_file(path, target_index, saved_name)
    us.append(u)
    E_ks.append(E_k)
    ts.append(t)   

### Plot
for i in range(len(F)):
    plt.plot(ts[i], E_ks[i][:, 0, 0], marker = "o", ms = 4, label = "%.0f" %(F[i]), alpha = .3)

plt.grid()
plt.legend()
plt.yscale("log")
plt.xlabel("t")
plt.ylabel(r"$E_K$")
plt.title(r"$E_k$ vs time in different max time")
plt.savefig("Ek_t_dedalus.pdf")