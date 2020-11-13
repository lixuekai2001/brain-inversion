from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from braininversion.PostProcessing import (get_source_expression,
                                           scalar_bars,
                                           load_meshes)
import os
import ufl
import sys

mesh_name = sys.argv[1]
sim_name = sys.argv[2]

sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
plot_dir = f"results/{mesh_name}_{sim_name}/plots/"
try:
    os.mkdir(plot_dir)
except FileExistsError:
    pass
porous_id = 1
fluid_id = 2
interface_id = 1
spinal_outlet_id = 3
mmHg2Pa = 132.32
m3tomL = 1e6

with open(mesh_config_file) as conf_file:
    mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)
with open(sim_config_file) as conf_file:
    sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)  

mesh, subdomain_marker, label_marker, boundary_marker, label_boundary_marker = load_meshes(mesh_name)
gdim = mesh.geometric_dimension()
sim_file = f"results/{mesh_name}_{sim_name}/results.xdmf"
source_conf = sim_config["source_data"]
T = sim_config["T"]
num_steps = sim_config["num_steps"]
dt = T/num_steps
times = np.linspace(0, T, num_steps + 1)
source_expr = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)

domains = mesh_config["domains"]
name_to_label = {dom["name"]:dom["id"] for dom in domains}

dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
# compute total volumes:
for dom in domains:
    vol = assemble(Constant(1)*dx( dom["id"] ))
    print(f" volume of {dom['name']} : {vol*m3tomL}")


