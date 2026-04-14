import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pathlib
import subprocess
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Clean up any old files
import shutil
shutil.rmtree('analysis', ignore_errors=True)


def simul(Lx, Lz, Nx, Nz, F, dealias, stop_sim_time, timestepper, max_timestep, dtype, Ly = 0, Ny = 0, i = 0, Schmidt = 0):
    if Ly != 0:
        V = Lx * Ly * Lz
    else:
        V = Lx * Lz
        D = 1 / Schmidt

    # Bases
    if Ly != 0:
        coords = d3.CartesianCoordinates('x', 'y', 'z')
    else:
        coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype = dtype)
    xbasis = d3.RealFourier(coords['x'], size = Nx, bounds = (0, Lx), dealias = dealias)

    if Ly != 0:
        ybasis = d3.RealFourier(coords['y'], size = Ny, bounds = (0, Ly), dealias = dealias)

    zbasis = d3.RealFourier(coords['z'], size = Nz, bounds = (0, Lz), dealias = dealias)

    # Fields
    if Ly != 0:
        p = dist.Field(name = 'p', bases = (xbasis, ybasis, zbasis))
        u = dist.VectorField(coords, name = 'u', bases = (xbasis, ybasis, zbasis))
        f = dist.VectorField(coords, name = 'f', bases = (xbasis, ybasis, zbasis))
    else:
        p = dist.Field(name = 'p', bases = (xbasis, zbasis))
        u = dist.VectorField(coords, name = 'u', bases = (xbasis, zbasis))
        f = dist.VectorField(coords, name = 'f', bases = (xbasis, zbasis))
        s = dist.Field(name='s', bases=(xbasis,zbasis))

    tau_p = dist.Field(name = 'tau_p')

    # Substitutions

    # kx, ky, kz = dist.local_modes(xbasis, ybasis, zbasis)
    if Ly != 0:
        x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
        ex, ey, ez = coords.unit_vector_fields(dist)
        f['g'][0] = F * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        f['g'][1] = - F * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        f['g'][2] = F * np.sqrt(2) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    else:
        x, z = dist.local_grids(xbasis, zbasis)
        ex, ez = coords.unit_vector_fields(dist)
        f['g'][0] = F * np.sqrt(2) * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z)
        f['g'][1] = - F * np.sqrt(2) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z)


    # Initial conditions
    # Background shear
    # u['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.25)/0.005) - np.tanh((z+0.75)/0.005))
    # Add small vertical velocity perturbations localized to the shear layers
    # u['g'][1] = 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.25)**2/0.0025)
    # u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.75)**2/0.0025)
    # u['g'][2] = 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-0.25)**2/0.0025)
    # u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z+0.75)**2/0.0025)

    # Problem
    if Ly != 0:
        problem = d3.IVP([u, p, tau_p], namespace = locals())
    else:
        problem = d3.IVP([u, s, p, tau_p], namespace = locals())  ## add tracer
    problem.add_equation("dt(u) + grad(p) - lap(u) = - u@grad(u) + f")
    if Ly == 0:
        problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")  ## add tracer
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    # Background shear
    # u['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
    if Ly == 0:
        u['g'][0, :, 0] = 0
        u['g'][0, :, -1] = 1
    else:
        u['g'][0, :, :, 0] = 0
        u['g'][0, :, :, -1] = 1

    if Ly == 0:
        s['g'] = u['g'][0]
    # Add small vertical velocity perturbations localized to the shear layers
    # u['g'][1] = 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
    # u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)
    # u['g'][2] = 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z-0.5)**2/0.01)
    # u['g'][2] += 0.1 * np.sin(2*np.pi*y/Ly) * np.exp(-(z+0.5)**2/0.01)
    ## try another initial condions, talor-green vortex (remember to examine Lx, Ly, Lz again when not equal)
    # if Ly != 0:
    #     u['g'][0] = np.cos(2 * np.pi / Lx * x) * np.sin(2 * np.pi / Ly * y) * np.sin(2 * np.pi / Lz * z)
    #     u['g'][1] = - np.sin(2 * np.pi / Lx * x) * np.cos(2 * np.pi / Ly * y) * np.sin(2 * np.pi / Lz * z)
    #     u['g'][2] = 0
    # else:
    #     u['g'][0] = np.sin(2 * np.pi / Lx * x) * np.cos(2 * np.pi / Lz * z)
    #     # u['g'][0] += 0.01 * np.random.standard_normal(u['g'][0].shape)
    #     s['g'] = u['g'][0]
    #     u['g'][1] = - np.cos(2 * np.pi / Lx * x) * np.sin(2 * np.pi / Lz * z)
    #     # u['g'][1] += 0.01 * np.random.standard_normal(u['g'][1].shape)

    # Analysis
    snapshots = solver.evaluator.add_file_handler('../.project/dir.project/data/snapshots_timetest_Fe5_%i' %(i), sim_dt=max_timestep * 10000, max_writes=3000)
    snapshots.add_task(u, layout = 'g', name = 'velocity')
    snapshots.add_task(p, name = 'pressure')
    snapshots.add_task(d3.Average(f @ u), layout = 'g', name = 'rhs')
    snapshots.add_task(f, layout = 'g', name = 'force')

    if Ly == 0:
        snapshots.add_task(s, name = 'tracer')

    snapshots.add_task(d3.Average(1 / 2 * u @ u), layout = 'g', name = 'kinetic energy')

    if Ly != 0:
        snapshots.add_task(d3.Average(1 / 2 * d3.Curl(u) @ d3.Curl(u)), layout = 'g', name = 'dissipation')
    else:
        snapshots.add_task(d3.Average(1 / 2 * -d3.div(d3.skew(u)) * -d3.div(d3.skew(u))), layout = 'g', name = 'dissipation')

    # CFL
    # CFL = d3.CFL(solver, initial_dt=max_timestep, safety=max_timestep, threshold=max_timestep * 10,
    #              max_change=1.0, min_change=0.5, max_dt=max_timestep)
    # CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property((u@ez)**2, name='w2')

    # Main loop
    try:
        logger.info('Starting main loop')
        dt = max_timestep   ## fixed time step, not using built-in CFL
        while solver.proceed:
            # timestep = CFL.compute_timestep()  ##using built-in CFL
            # solver.step(timestep)  ##using built-in CFL
            solver.step(dt)  ## fixed time step, not using built-in CFL
            if (solver.iteration-1) % 10 == 0:
                max_w = np.sqrt(flow.max('w2'))
                # logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))  ##using built-in CFL
                logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, dt, max_w))   ## fixed time step, not using built-in CFL
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
