from dolfin import *
from multiphenics import *
from braininversion.BiotNavierStokesSolver import solve_biot_navier_stokes
from braininversion.ArrayExpression import getArrayExpression
import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
parameters['ghost_mode'] = 'shared_facet' 


def runFluidPorousBrainSim(config):
    set_log_level(config["log_level"])
    num_threads = config["num_threads"]
    if num_threads!="default":
        PETScOptions.set("mat_mumps_use_omp_threads", num_threads)
        
    #PETScOptions.set("snes_lag_jacobian",1) #use -1 to never recompute

    # set parameter
    mesh_name = config["mesh_dir_name"]
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
    #
    infile_dir = f"../meshes/{mesh_name}"
    outfile = f"../results/{mesh_name}_{T}_{num_steps}/{mesh_name}_{T}_{num_steps}"
    subdomains_infile = f"{infile_dir}/{mesh_name}.xdmf"
    boundary_infile = f"{infile_dir}/{mesh_name}_boundaries.xdmf"
    porous_restriction_file = f"{infile_dir}/{mesh_name}_porous.rtc.xdmf"
    fluid_restriction_file = f"{infile_dir}/{mesh_name}_fluid.rtc.xdmf"

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
    #mesh.smooth(50)
    subdomains = MeshFunction("size_t", mesh, gdim, 0)
    infile.read(subdomains)
    infile.close()

    infile = XDMFFile(boundary_infile)
    boundaries =  MeshFunction("size_t", mesh, gdim - 1, 0)
    infile.read(boundaries)

    # define boundary conditions

    boundary_conditions = [
        {rigid_skull_id: {0:Constant([0.0]*gdim)}},
        #{fixed_stem_id: {2:Constant([0.0]*gdim)}},
        ]

    source_conf = config["source_data"]
    if "source_expression" in source_conf.keys():
        g_source = Expression(source_conf["source_expression"],
                        t=0.0,degree=2,**source_conf["source_params"])
    elif "source_file" in source_conf.keys():
        #g_source = InterpolatedSource(source_conf["source_file"])
        data = np.loadtxt(source_conf["source_file"], delimiter=",")
        t = data[:,0]
        inflow = data[:,1]
        if "scaling" in source_conf.keys():
            inflow *= source_conf["scaling"]
        if source_conf["scale_by_total_vol"]:
            dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
            tot_parenchyma_vol = assemble(1.0*dx(porous_id))
            fluid_vol = assemble(1.0*dx(fluid_id))
            inflow /= tot_parenchyma_vol
            print(f"total volume of brain tissue: {tot_parenchyma_vol}")
            print(f"total volume of CSF: {fluid_vol}")
    
        values = np.interp(times, t, inflow, period = t[-1])
        plt.figure()
        plt.plot(times, values)
        plt.savefig("source_time_series.png")
        g_source = getArrayExpression(values)
        i = 1
        g_source.i = i
        assert g_source(Point(0, 0, 0)) == values[i]
        i = len(times) - 1
        g_source.i = i
        assert g_source(Point(0, 0, 0)) == values[i]

    results = solve_biot_navier_stokes(mesh, T, num_steps,
                                    material_parameter, 
                                    boundaries,
                                    subdomains,
                                    boundary_conditions,
                                    porous_restriction_file,
                                    fluid_restriction_file,
                                    sliprate=sliprate,
                                    g_source=g_source,
                                    filename=outfile,
                                    elem_type=config["element_type"],
                                    linearize=config["linearize"],
                                    move_mesh=False,
                                    time_dep_expr=[g_source])

    return f"../results/{mesh_name}_{T}_{num_steps}/{mesh_name}_{T}_{num_steps}"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c",
        metavar="config.yml",
        help="path to config file",
        type=str,)
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    target_folder = runFluidPorousBrainSim(config)
    os.popen(f'cp {config_file_path} {target_folder}_config.yml') 
