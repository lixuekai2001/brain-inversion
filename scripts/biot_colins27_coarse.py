from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from braininversion.BiotSolver import solve_biot
from braininversion.IOHandling import read_mesh_from_h5, write_to_xdmf, xdmf_to_unstructuredGrid, read_xdmf_timeseries
import pyvista as pv
from dolfin.cpp.log import info


set_log_level(13)
parameters['ghost_mode'] = 'shared_facet' 

# Form compiler options
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["optimize"] = True

h5_filename = "../meshes/colin27_coarse_boundaries.h5"
outfile = "../results/biot_colins27_coarse.xdmf"

# time stepping
T = 0.01           # final time
num_steps = 1    # number of time steps
dt = T/ num_steps
times = np.linspace(dt, T, num_steps)

# material parameter
kappa = 1e-17       # permeability 15*(1e-9)**2
visc = 0.8*1e-3     # viscocity 
K = kappa/visc      # hydraulic conductivity
c = 2*1e-8         # storage coefficent
alpha = 1.0         # Biot-Willis coefficient

# Biot material parameters
E = 1500.0          # Young modulus
nu = 0.479         # Poisson ratio

material_parameter = dict()
material_parameter["c"] = c
material_parameter["K"] = K
material_parameter["lmbda"] = nu*E/((1.0-2.0*nu)*(1.0+nu)) 
material_parameter["mu"] = E/(2.0*(1.0+nu))
material_parameter["alpha"] = alpha

mesh, bm = read_mesh_from_h5(h5_filename) #skull: 1, ventricle:2

n = FacetNormal(mesh)

mmHg2Pa = 132.32
freq = 1.0 
A = 2*mmHg2Pa
p_obs_outer = Expression("A*sin(2*pi*f*t)", A=A,f=freq,t=0,degree=2)
p_obs_inner = Constant(0.0)

u_nullspace = True

g = Constant(0.0)
f = Constant([0.0, 0.0, 0.0])

boundary_conditions_p = {2:{"Neumann":Constant(0.0)},
                         1:{"Neumann":Constant(0.0)},
                         }

boundary_conditions_u = {1:{"Neumann":n*p_obs_outer},
                         2:{"Neumann":n*p_obs_inner},
                        } 

solution = solve_biot(mesh, f, g, T, num_steps, material_parameter,
                              bm, boundary_conditions_p,
                              bm, boundary_conditions_u,
                              u_nullspace=u_nullspace, theta=1.0,
                              u_degree=2, p_degree=1,
                              solver_type="LU")
solution = [s.copy() for s in solution]

                        
