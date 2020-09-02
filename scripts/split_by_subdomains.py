from fenics import *
from braininversion.IOHandling import (read_mesh_from_h5, write_to_xdmf, 
                                       xdmf_to_unstructuredGrid, read_xdmf_timeseries)
import yaml
import numpy as np

import sys 
config_file_path = sys.argv[1] 

with open(config_file_path) as conf_file:
    config = yaml.load(conf_file, Loader=yaml.FullLoader)

mesh_name = config["mesh_dir_name"]
T = config["T"]
num_steps = config["num_steps"]
# subdomain ids
fluid_id = 2
porous_id = 1

# boundary ids
interface_id = 1
rigid_skull_id = 2
spinal_outlet_id = 3
fixed_stem_id = 4

sim_name = f"{mesh_name}_{T}_{num_steps}"
config_file_path = f"../results/{sim_name}/{sim_name}_config.yml"
sim_file = f"../results/{sim_name}/{sim_name}_checkp.xdmf"
sim_file_fluid = f"../results/{sim_name}/{sim_name}_fluid.xdmf"
sim_file_porous = f"../results/{sim_name}/{sim_name}_por.xdmf"

mesh_file_por = f"../meshes/{mesh_name}/{mesh_name}_porous.xdmf"
mesh_file_fluid = f"../meshes/{mesh_name}/{mesh_name}_fluid.xdmf"
mesh_file = f"../meshes/{mesh_name}/{mesh_name}.xdmf"

mesh_config_path = f"../meshes/{mesh_name}/{mesh_name}_config.yml"
with open(mesh_config_path) as conf_file:
    mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

dt = T/num_steps
times = np.linspace(0, T, num_steps + 1)

infile_mesh = XDMFFile(mesh_file)
mesh = Mesh()
infile_mesh.read(mesh)
gdim = mesh.geometric_dimension()

subdomain_marker = MeshFunction("size_t", mesh, gdim, gdim)
infile_mesh.read(subdomain_marker, "subdomains")

#infile_mesh_por = XDMFFile(mesh_file_por)
fluid_submesh = SubMesh(mesh, subdomain_marker, fluid_id)
por_submesh = SubMesh(mesh, subdomain_marker, porous_id)
#por_submesh = Mesh()
#infile_mesh_por.read(por_submesh)
#infile_mesh_fluid = XDMFFile(mesh_file_fluid)
#fluid_submesh = Mesh()
#infile_mesh_fluid.read(fluid_submesh)
v_order = 2

V = VectorFunctionSpace(mesh, "CG", v_order)
V_por = VectorFunctionSpace(por_submesh, "CG", v_order)
V_fluid = VectorFunctionSpace(fluid_submesh, "CG", v_order)

W = FunctionSpace(mesh, "CG", 1)
W_por = FunctionSpace(por_submesh, "CG", 1)
W_fluid = FunctionSpace(fluid_submesh, "CG", 1)

names = {"pF":W, "pP":W, "phi":W,
         "d":V, "u":V}
domains = {"pF":"fluid", "pP":"porous", "phi":"porous",
         "d":"porous", "u":"fluid"}
infile_results = XDMFFile(sim_file)
results = {n:[] for n in names}
for n, space in names.items():
    for i in range(num_steps + 1):
        f = Function(space)
        #f.set_allow_extrapolation(True)
        infile_results.read_checkpoint(f, n, i)
        results[n].append(f)

        
outfile_por = XDMFFile(sim_file_porous)
outfile_fluid = XDMFFile(sim_file_fluid)
outfile_por.parameters["functions_share_mesh"] = True
outfile_por.parameters["rewrite_function_mesh"] = False
outfile_fluid.parameters["functions_share_mesh"] = True
outfile_fluid.parameters["rewrite_function_mesh"] = False
append_f = False
append_p = False

for name, timesteps in results.items():
    print(f"started splitting {name}...")
    if domains[name] == "fluid":
        for i in range(num_steps + 1):
            if isinstance(timesteps[0].ufl_element(), VectorElement):
                restricted_func = interpolate(timesteps[i], V_fluid)
            else:
                restricted_func = interpolate(timesteps[i], W_fluid)
            restricted_func.rename(name, name)
            outfile_fluid.write_checkpoint(restricted_func, name, i*dt, append=append_f)
            append_f = True
    if domains[name] == "porous":
        for i in range(num_steps + 1):
            if isinstance(timesteps[0].ufl_element(), VectorElement):
                restricted_func = interpolate(timesteps[i], V_por)
            else:
                restricted_func = interpolate(timesteps[i], W_por)
            restricted_func.rename(name, name)
            outfile_por.write_checkpoint(restricted_func, name, i*dt, append=append_p)
            append_p = True
    print(f"finished splitting {name}!")

        
outfile_fluid.close()
outfile_por.close()