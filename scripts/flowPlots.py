from fenics import *
import numpy as np
from braininversion.PlottingHelper import extract_cross_section
import matplotlib.pyplot as plt
import yaml
import pyvista as pv
from pathlib import Path
from braininversion.PostProcessing import (get_source_expression,
                                           load_meshes)
import os
import ufl
import sys

porous_id = 1
fluid_id = 2

mmHg2Pa = 132.32
m3tomL = 1e6

interface_id = 1
spinal_outlet_id = 3

names = ["d", "u"]

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
            "d":"porous", "u":"fluid"}

def domain_statistic(stat_func, var, domain, time_idx):
    return stat_func(extract_cells(results[var][time_idx], label_marker, name_to_label[domain]))
             
def extract_cells(f, subdomains, subd_id):
    '''f values in subdomains cells marked with subd_id'''
    V = f.function_space()
    dm = V.dofmap()
    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index()) for c in SubsetIterator(subdomains, subd_id)]))
    
    return f.vector().get_local()[subd_dofs]

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

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    plot_dir = f"results/{mesh_name}_{sim_name}/flow_plots/"

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
    V = VectorFunctionSpace(mesh, "CG", 2)

    infile = XDMFFile(sim_file)


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

    print("inflow plot created")


    results = {n:[] for n in names}
    for n in names:
        for i in range(num_steps + 1):
            f = Function(V)
            infile.read_checkpoint(f, n, i)
            results[n].append(f)
    infile.close()

    

    # aqueduct velocity

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
    plt.plot(times, cum_outflow*m3tomL, "-*", label="cumulative outflow")
    plt.legend()
    plt.grid()
    plt.xlabel("time in s")
    plt.ylabel("V in mL")
    plt.title("cumulative CSF outflow into spinal canal")
    plt.savefig(f"{plot_dir}/cum_spinal_out_CSF.png")

    flow_pairs = [("lateral_ventricles", "foramina"),
                #("foramina", "third_ventricle"),
                ("third_ventricle", "aqueduct"),
                #("aqueduct", "fourth_ventricle"),
                ("fourth_ventricle", "median_aperture"),
                #("median_aperture", "csf"),
                ]

    internal_flows = {}
    for fp in flow_pairs:
        flow = compute_internal_flow(fp[0], fp[1])
        internal_flows[f"{fp[0]} -> {fp[1]}"] = flow
        

    plt.figure(figsize=(10,8))
    for name, flow in internal_flows.items():
        plt.plot(times, flow*m3tomL, "-*", label=name)
    #plt.plot(times, spinal_outflow*m3tomL, "-*", label="spinal outflow")

    plt.legend()
    plt.grid()
    plt.xlabel("time in s")
    plt.ylabel("flowrate in mL/ s")
    plt.title("ventricular CSF flow")
    plt.savefig(f"{plot_dir}/ventr_CSF_flow.png")
    # max value according to Baladont: 0.2 mL/s


    plt.figure(figsize=(10,8))
    for name, flow in internal_flows.items():
        plt.plot(times, np.cumsum(flow)*dt*m3tomL, label=name)
        
    #plt.plot(times, cum_outflow*dt*m3tomL/50, label="cum spinal outflow (scaled: 1/50)")

    plt.legend()
    plt.grid()
    plt.xlabel("time in s")
    plt.ylabel("V in mL")
    plt.title("cumulative ventricular CSF flow")
    plt.savefig(f"{plot_dir}/cum_CSF_flow.png")


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


    # compute flow over parenchyma csf interface
    #ds_interf = Measure("dS", domain=mesh, subdomain_data=boundary_marker, subdomain_id=interface_id)
    #dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    #n = FacetNormal(mesh)("-")
    #par_csf_flow = np.array( [ assemble(dot(-grad(pP)("-"), n)*ds_interf + Constant(0.0)*dx) for pP in results["pP"] ] )
    #plt.figure(figsize=(10,8))
    #plt.plot(times, par_csf_flow*m3tomL, label="par_csf_flow")
    #plt.legend()
    #plt.grid()
    #plt.xlabel("time in s")
    #plt.ylabel("dV in ml")
    #plt.title("flow over parenchyma-csf interface")
    #plt.savefig(f"{plot_dir}/parenchyma-csf_flow.png")