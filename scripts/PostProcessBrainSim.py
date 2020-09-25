from fenics import *
from multiphenics import *
import numpy as np
from braininversion.IOHandling import (read_mesh_from_h5, write_to_xdmf, 
                                       xdmf_to_unstructuredGrid, read_xdmf_timeseries)
from braininversion.PlottingHelper import (plot_pressures_and_forces_timeslice, 
                                           plot_pressures_and_forces_cross_section,
                                           extract_cross_section, style_dict)
import matplotlib.pyplot as plt
import yaml
import pyvista as pv
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
probes = mesh_config["probes"]
flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])
domains = mesh_config["domains"]
name_to_label = {dom["name"]:dom["id"] for dom in domains}


class subdomainFilter(UserExpression):
    def __init__(self, subdomain_marker, subdomain_id, **kwargs):
        self.marker = subdomain_marker
        self.domain_id = subdomain_id
        super().__init__(**kwargs)


    def eval_cell(self, values, x, cell):
            if self.marker[cell.index] == self.domain_id:
                values[0] = 1
            else:
                values[0] = np.nan

fluid_filter = subdomainFilter(subdomain_marker, fluid_id, degree=0)
por_filter = subdomainFilter(subdomain_marker, porous_id, degree=0)

DG = FunctionSpace(mesh, "DG", 0)
fluid_filter = interpolate(fluid_filter, DG)
fluid_filter.set_allow_extrapolation(True)
por_filter = interpolate(por_filter, DG)
por_filter.set_allow_extrapolation(True)


V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

names = {"pF":W, "pP":W, "phi":W,"d":V,"u":V}

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
             "d":"porous", "u":"fluid"}

infile = XDMFFile(sim_file)


results = {n:[] for n in names}
for n, space in names.items():
    for i in range(num_steps + 1):
        f = Function(space)
        if variables[n] == "fluid":
            infile.read_checkpoint(f, n, i)
        elif variables[n] == "porous":
            infile.read_checkpoint(f, n, i)
        else:
            print("error!")
        results[n].append(f)

infile.close()

for vec in ["u","d"]:
    results[f"{vec}_tot"] = {}
    for i in range(num_steps + 1):
        v = results[vec][i]
        results[f"{vec}_tot"][i] = project(sqrt(inner(v,v)), W)


def plot_scalar_time_evolution(point_var_list, mesh_config, results, ylabel,times, scale=1):
    plt.figure(figsize=(10,8))
    plotname = f"pressure_" 
    for (var, p_name) in point_var_list:
        data = extract_cross_section(results[var], [Point(flatprobes[p_name])]).flatten()*scale
        plt.plot(times, data,"-*" , label=f"{var} : {p_name}")
        plotname += p_name + "_"
    plt.legend()
    plt.grid()
    plt.xlabel("t [s]")
    plt.ylabel(ylabel)
    plt.savefig(f"{plot_dir}/{plotname[:-1]}.png")

ylabel = "p in mmHg"
point_var_list = [("pF","front_sas"),("phi","front_parenchyma"),("pF","lateral_ventricles")]
plot_scalar_time_evolution(point_var_list, mesh_config, results, ylabel, times, scale=1/mmHg2Pa)

point_var_list = [("pF","back_sas"),("phi","back_parenchyma"),("pF","lateral_ventricles")]
plot_scalar_time_evolution(point_var_list, mesh_config, results, ylabel, times, scale=1/mmHg2Pa)



plotname = "vel_aqueduct"
title = "aqueduct velocity"
point = "aqueduct"  
var = "u"
plt.figure(figsize=(10,8))
data = extract_cross_section(results[var], [Point(flatprobes[point])])*1e3
data = data[:,0,:]
tot = np.linalg.norm(data, axis=1)
for i, comp in enumerate(["x","y","z"]):
    plt.plot(times, data[:,i], "-*", label=f"u_{comp}")
plt.legend()
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("u in mm/s")
plt.title(title)
plt.savefig(f"{plot_dir}/{plotname}.png")

# plot source in inflow
plotname = "source_inflow"
source_inflow = []
for i in range(num_steps + 1):
    source_expr.i = i
    source_inflow.append(source_expr(Point([0,0,0])))
source_inflow = np.array(source_inflow)
dx_par = Measure("dx", domain=mesh, subdomain_data=label_marker, subdomain_id = name_to_label["parenchyma"])
parenchyma_vol = assemble(Constant(1)*dx_par)
plt.figure(figsize=(10,8))
plt.plot(times, source_inflow*parenchyma_vol*m3tomL, "-*")
plt.grid()
plt.xlabel("t in s")
plt.ylabel("inflow [ml/s]")
plt.title("net blood inflow")
plt.savefig(f"{plot_dir}/{plotname}.png")


# compute pressure gradient
gradient_ventr_probe_point = "lateral_ventricles"
gradient_sas_probe_point = "top_sas"
plotname = f"pressure_gradient_{gradient_ventr_probe_point}_{gradient_sas_probe_point}"

ventr_point = flatprobes[gradient_ventr_probe_point]
sas_point = flatprobes[gradient_sas_probe_point]
ventr_data = extract_cross_section(results["pF"], [Point(ventr_point)]).flatten()/mmHg2Pa
sas_data = extract_cross_section(results["pF"], [Point(sas_point)]).flatten()/mmHg2Pa

dist = np.array(ventr_point) - np.array(sas_point)
dist = np.linalg.norm(dist)
diff = ventr_data - sas_data
gradient = diff/dist
plt.figure(figsize=(10,8))
plt.plot(times, gradient , label="gradient")

#plt.legend()
plt.grid()
plt.xlabel("t [s]")
plt.title(f"pressure gradient ({gradient_ventr_probe_point} - {gradient_sas_probe_point})")
plt.ylabel("pressure grad in mmHg/m")
plt.savefig(f"{plot_dir}/{plotname}.png")




# plot cross section through the domain
filter_dict = {"pF":fluid_filter, "phi":por_filter, "pP":por_filter, "d":por_filter, "u":fluid_filter}

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)]
    
def plot_cross_section(p1, p2, n_crossP, variables, results, time_idx, filter_dict, scale=1):
    p1_coords = np.array(flatprobes[p1])
    p2_coords = np.array(flatprobes[p2])
    cross_points = [Point(p) for p in intermediates(p1_coords,p2_coords, n_crossP)]
    dist = np.linspace(0, np.linalg.norm(p2_coords - p1_coords), n_crossP)
    values = []
    for var in variables:
        cs = extract_cross_section(results[var], cross_points, filter_function=filter_dict[var])*scale
        values.append(cs)

    for time_idx in time_indices:
        plt.figure(figsize=(10,8))
        for i, var in enumerate(variables):
            plt.plot(dist, values[i][time_idx,:], ".-", label=var)

        plt.legend()
        plt.grid()
        plt.title(f"{p1} to {p2}, t = {(time_idx)*dt:.3f}")
        plt.xlabel("distance in m")
        plt.ylabel("p in mmHg")
        plt.savefig(f"{plot_dir}/cross_section_{p1}_{p2}_{time_idx}.png")
        
        
n_crossP = 300
time_indices = np.linspace(1, num_steps, 10, dtype=np.int)
p1 = "left_sas"
p2 = "right_sas"
plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                   results, time_indices, filter_dict, scale=1/mmHg2Pa)
p1 = "front_parenchyma"
p2 = "back_parenchyma"
plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                   results, time_indices, filter_dict, scale=1/mmHg2Pa)

p1 = "lateral_ventricles"
p2 = "top_sas"
plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                   results, time_indices, filter_dict, scale=1/mmHg2Pa)


# compute outflow into spinal coord 
ds_outflow = Measure("ds", domain=mesh, subdomain_data=boundary_marker, subdomain_id=spinal_outlet_id)
n = FacetNormal(mesh)

spinal_outflow = np.array([assemble(dot(u,n)*ds_outflow) for u in results["u"]])
plt.figure(figsize=(10,8))
plt.plot(times, spinal_outflow*m3tomL, label="outflow into spinal coord")
plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("flowrate in mL/ s")
plt.title("CSF outflow into spinal coord")
plt.savefig(f"{plot_dir}/spinal_out_CSF.png")


cum_outflow = np.cumsum(spinal_outflow)*dt
plt.figure(figsize=(10,8))
plt.plot(times, cum_outflow*m3tomL, label="cumulative outflow into spinal coord")
plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("V in mL")
plt.title("cumulative CSF outflow into spinal coord")
plt.savefig(f"{plot_dir}/cum_spinal_out_CSF.png")


# In[14]:


def compute_internal_flow(dom1, dom2):
    """ compute flow from dom1 into dom2"""
    dom1_id = name_to_label[dom1]
    dom2_id = name_to_label[dom2]
    intf_id = int(f"{min([dom1_id, dom2_id])}{max([dom1_id, dom2_id])}")
    ds_intf = Measure("dS", domain=mesh, subdomain_data=label_boundary_marker,
                      subdomain_id=intf_id)
    dx = Measure("dx", domain=mesh, subdomain_data=label_marker)
    if dom1_id > dom2_id:
        n = FacetNormal(mesh)("+")
    else:
        n = FacetNormal(mesh)("-")
    flow = np.array([assemble( dot(u, n)*ds_intf + Constant(0.0)*dx) for u in results["u"] ] )
    return flow

flow_pairs = [("lateral_ventricles", "foramina"),
             ("foramina", "third_ventricle"),
             ("third_ventricle", "aqueduct"),
             ("aqueduct", "fourth_ventricle"),
             ("fourth_ventricle", "median_aperture"),
              ("median_aperture", "csf"),
             ]

internal_flows = {}
for fp in flow_pairs:
    flow = compute_internal_flow(fp[0], fp[1])
    internal_flows[f"{fp[0]} -> {fp[1]}"] = flow
    

plt.figure(figsize=(10,8))
for name, flow in internal_flows.items():
    plt.plot(times, flow*m3tomL, label=name)
plt.plot(times, spinal_outflow*m3tomL/50, label="spinal outflow (scaled: 1/50)")

plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("flowrate in mL/ s")
plt.title("ventricular CSF flow")
plt.savefig(f"{plot_dir}/ventr_CSF_flow.png")
# max value according to Baladont: 0.2 mL/s


# In[15]:


plt.figure(figsize=(10,8))
for name, flow in internal_flows.items():
    plt.plot(times, np.cumsum(flow)*dt*m3tomL, label=name)
    
plt.plot(times, cum_outflow*dt*m3tomL/50, label="cum spinal outflow (scaled: 1/50)")

plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("V in mL")
plt.title("cumulative ventricluar CSF flow")
plt.savefig(f"{plot_dir}/cum_CSF_flow.png")


# In[20]:


# compute parenchyma volume change
ds_interf = Measure("dS", domain=mesh, subdomain_data=boundary_marker, subdomain_id=interface_id)
dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
n = FacetNormal(mesh)("-")
par_dV = np.array([assemble(dot(d("-"), n)*ds_interf + Constant(0.0)*dx) for d in results["d"]])
plt.figure(figsize=(10,8))
plt.plot(times, par_dV*m3tomL, label="DV")
plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("dV in ml")
plt.title("parenchyma volume change")
plt.savefig(f"{plot_dir}/par_vol_change.png")


# In[22]:


# compute flow over parenchyma csf interface
ds_interf = Measure("dS", domain=mesh, subdomain_data=boundary_marker, subdomain_id=interface_id)
dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
n = FacetNormal(mesh)("-")
par_csf_flow = np.array( [ assemble(dot(-grad(pP)("-"), n)*ds_interf + Constant(0.0)*dx) for pP in results["pP"] ] )
plt.figure(figsize=(10,8))
plt.plot(times, par_csf_flow*m3tomL, label="par_csf_flow")
plt.legend()
plt.grid()
plt.xlabel("time in s")
plt.ylabel("dV in ml")
plt.title("flow over parenchyma-csf interface")
plt.savefig(f"{plot_dir}/parenchyma-csf_flow.png")


# In[ ]:


def domain_statistic(stat_func, var, domain, time_idx):
    return stat_func(extract_cells(results[var][time_idx], label_marker, name_to_label[domain]))
             
def extract_cells(f, subdomains, subd_id):
    '''f values in subdomains cells marked with subd_id'''
    V = f.function_space()
    dm = V.dofmap()
    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index()) for c in SubsetIterator(subdomains, subd_id)]))
    
    return f.vector().get_local()[subd_dofs]


aquduct_max = [domain_statistic(np.max, "u_tot", "aqueduct", i) for i in range(num_steps + 1)]
plt.figure(figsize=(10, 8))
plt.plot(times, aquduct_max)
plt.grid()


# In[ ]:


parenchyma_max = np.array([domain_statistic(np.max, "phi", "parenchyma", i) for i in range(num_steps + 1)])
parenchyma_min = np.array([domain_statistic(np.min, "phi", "parenchyma", i) for i in range(num_steps + 1)])
parenchyma_std = np.array([domain_statistic(np.std, "phi", "parenchyma", i) for i in range(num_steps + 1)])

plt.figure(figsize=(10, 8))
plt.plot(times, parenchyma_std)
plt.grid()


# In[ ]:




