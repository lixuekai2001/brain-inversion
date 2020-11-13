from dolfin import *
from multiphenics import *
from braininversion.BiotNavierStokesSolver import solve_biot_navier_stokes
from braininversion.SourceExpression import get_source_expression
import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#parameters["mesh_partitioner"] = "ParMETIS" # ParMETIS
#parameters["partitioning_approach"] = "PARTITION"
parameters['ghost_mode'] = 'shared_facet' 

def runFluidPorousBrainSim(config_file_path, mesh_file_path, outfile_path):
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    
    set_log_level(config["log_level"])
    num_threads = config["num_threads"]
    if num_threads!="default":
        pass
    PETScOptions.set("mat_mumps_use_omp_threads", 4)
    #PETScOptions.set("mat_mumps_icntl_13", 0) # set scalapack use
    PETScOptions.set("mat_mumps_icntl_4", 3) # set mumps verbosity (0-4)
    #PETScOptions.set("mat_mumps_icntl_14", 50) # set percentage increase of workspace
    #PETScOptions.set("mat_mumps_icntl_7", 5) # set mumps ordering
    #PETScOptions.set("mat_mumps_icntl_20", 0) # centralized rhsset mumps ordering
    PETScOptions.set("mat_mumps_icntl_22", 1) # use out of core
    PETScOptions.set("mat_mumps_icntl_28", 2) # use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering
    PETScOptions.set("mat_mumps_icntl_29", 2) # parallel ordering 1 = ptscotch, 2 = parmetis
    #PETScOptions.set("mat_mumps_icntl_35", 1) # set use of BLR (Block Low-Rank) feature (0:off, 1:optimal)
    #PETScOptions.set("mat_mumps_cntl_7", 1e-6) # set BLR relaxation

    #PETScOptions.set("snes_lag_jacobian",1) #use -1 to never recompute

    # set parameter
    sim_name = Path(config_file_path).stem
    mesh_stem = Path(mesh_file_path).stem
    mesh_parent = Path(mesh_file_path).parent
    mesh_path = f"{mesh_parent}/{mesh_stem}"
    T = config["T"]
    num_steps = config["num_steps"]
    dt = T/num_steps
    times = np.linspace(dt, T, num_steps)

    sliprate = config["sliprate"]
    material_parameter = config["material_parameter"]
    E = material_parameter["E"]
    nu = material_parameter["nu"]
    material_parameter["mu_s"] = Constant(E/(2.0*(1.0+nu)))
    material_parameter["lmbda"] = Constant(nu*E/((1.0-2.0*nu)*(1.0+nu)))

    # read meshes:
    subdomains_infile = f"{mesh_path}.xdmf"
    boundary_infile = f"{mesh_path}_boundaries.xdmf"
    porous_restriction_file = f"{mesh_path}_porous.rtc.xdmf"
    fluid_restriction_file = f"{mesh_path}_fluid.rtc.xdmf"

    # subdomain ids
    fluid_id = 2
    porous_id = 1

    # boundary ids
    interface_id = 1
    rigid_skull_id = 2
    spinal_outlet_id = 3
    fixed_stem_id = 4

    infile = XDMFFile(subdomains_infile)
    mesh = Mesh()
    infile.read(mesh)
    gdim = mesh.geometric_dimension()
    subdomains = MeshFunction("size_t", mesh, gdim, 0)
    infile.read(subdomains)
    infile.close()

    infile = XDMFFile(boundary_infile)
    boundaries =  MeshFunction("size_t", mesh, gdim - 1, 0)
    infile.read(boundaries)

    initial_pressure = config["initial_pressure"]
    # define boundary conditions
    spinal_pressure = Expression(config["spinal_outlet"]["outlet_expression"],
                                 P0 = initial_pressure, **config["spinal_outlet"]["outlet_params"],
                                 outflow_vol=0.0, degree=2)

    #spinal_pressure = Expression("100*sin(2*pi*t)", t=0.0, degree=2)
    boundary_conditions = [
        {rigid_skull_id: {0:Constant([0.0]*gdim)}},
        {fixed_stem_id: {2:Constant([0.0]*gdim)}},
        ]

    source_conf = config["source_data"]
    g_source = get_source_expression(source_conf, mesh,
                                     subdomains, porous_id,
                                     times)
    
    #g_source = Expression("0.0", degree=2, t=0.0)
    solve_biot_navier_stokes(mesh, T, num_steps,
                            material_parameter, 
                            boundaries,
                            subdomains,
                            boundary_conditions,
                            porous_restriction_file,
                            fluid_restriction_file,
                            sliprate=sliprate,
                            g_source=g_source,
                            outlet_pressure=spinal_pressure,
                            initial_pressure=initial_pressure,
                            filename=outfile_path,
                            elem_type=config["element_type"],
                            linearize=config["linearize"],
                            move_mesh=False,
                            time_dep_expr=[g_source, spinal_pressure])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
        metavar="config.yml",
        help="path to config file",
        type=str,)
    parser.add_argument("-m",
        metavar="mesh.xdml",
        help="path to mesh file",
        type=str,)
    parser.add_argument("-o",
        metavar="result.xdml",
        help="path to outfile",
        type=str,)
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    mesh_file_path = conf_arg["m"]
    outfile_path = conf_arg["o"]
    runFluidPorousBrainSim(config_file_path, mesh_file_path, outfile_path)
