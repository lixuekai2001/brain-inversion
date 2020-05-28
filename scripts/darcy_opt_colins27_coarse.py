#!/usr/bin/env python
# coding: utf-8


from fenics import *
import numpy as np
from IOHandling import read_mesh_from_h5, write_to_xdmf
from fenics_adjoint import *
import moola
from DarcySolver import solve_darcy
from dolfin.cpp.log import info
from Optimization import optimize_darcy_source

set_log_level(40)
parameters['ghost_mode'] = 'shared_facet' 

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


h5_filename = "../meshes/colin27_coarse_boundaries.h5"

# time stepping
T = 1.2           # final time
num_steps = 8    # number of time steps
dt = T/ num_steps
times = np.linspace(dt, T, num_steps)

# material parameter
kappa = 1e-17       # permeability 15*(1e-9)**2
visc = 0.8*1e-3     # viscocity 
K = kappa/visc      # hydraulic conductivity
c = 2*1e-4          # storage coefficient
mmHg2Pa = 132.32
material_parameter = {"K":K, "c":c}

mesh, boundary_marker = read_mesh_from_h5(h5_filename) #skull: 1, ventricle:2

f = 1
A = 0.2*mmHg2Pa
p_skull = Expression("A*sin(2*pi*f*t)", A=A,f=f,t=0,degree=2)
p_ventricle = Expression("A*sin(2*pi*f*t)", A=A*0.5,f=f,t=0,degree=2)
#p_ventricle = Constant(0.0)


boundary_conditions = {1:{"Neumann":Constant(0.0)}, 2:{"Neumann":Constant(0.0)}}
alpha = 1e-6
def laplace(p):
    mesh = p.function_space().mesh()
    n = FacetNormal(mesh)
    return alpha*jump(grad(p), n)**2

    
minimization_target = {"ds": {1: lambda x: (x - p_skull)**2,
                              2: lambda x: (x - p_ventricle)**2},
                       "dS":{"everywhere":laplace}
                        }
time_dep_expr = [p_skull, p_ventricle]

# compute constant controls
res = optimize_darcy_source(mesh, material_parameter, times, minimization_target,
                            boundary_marker, boundary_conditions,
                            time_dep_expr=time_dep_expr, opt_solver="scipy",
                            control_args="constant")

opt_ctrls, opt_solution, initial_solution = res

# compute DG0 controls
res = optimize_darcy_source(mesh, material_parameter, times, minimization_target,
                            boundary_marker, boundary_conditions,
                            time_dep_expr=time_dep_expr,
                            control_args=["DG", 0],
                            initial_guess=opt_ctrls)

opt_ctrls, opt_solution, initial_solution = res

write_to_xdmf([opt_ctrls, opt_solution],
              "../results/opt_darcy_colins27_coarse.xdmf", times)

