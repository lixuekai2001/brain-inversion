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
from collections import defaultdict
import ufl
import sys
plt.style.use('bmh') 
figsize = (7, 5)
porous_id = 1
fluid_id = 2

mmHg2Pa = 132.32
m3tomL = 1e6

phases = [0.13, .35, 0.56, 0.8]


interface_id = 1
spinal_outlet_id = 3

names = ["d", "u"]

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
            "d":"porous", "u":"fluid"}
style_dict = defaultdict(lambda : "-", 
                        {"lateral_ventricles":":",
                         "top_sas":"-*",
                         "top_parenchyma":"-."})



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
    plot_dir = f"results/{mesh_name}_{sim_name}/flow_plots"
    try:
        os.mkdir(plot_dir)
    except:
        pass
    key_quantities_file = f"results/{mesh_name}_{sim_name}/flow_key_quantities.yml"

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
    source_expr, source_vals = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)
    probes = mesh_config["probes"]
    flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])
    domains = mesh_config["domains"]
    name_to_label = {dom["name"]:dom["id"] for dom in domains}
    V = VectorFunctionSpace(mesh, "CG", 2)
    V_abs = FunctionSpace(mesh, "CG", 2)
    infile = XDMFFile(sim_file)

    key_quantities = {}

    start_idx = np.where(times==1)[0][0] 
    end_idx = np.where(times==T - 1)[0][0] - start_idx

    # plot source in inflow

    plotname = "source_inflow"
    source_inflow = []
    for i in range(num_steps + 1):
        source_expr.i = i
        source_inflow.append(source_expr(Point([0,0,0])))
    source_inflow = np.array(source_inflow)
    dx_par = Measure("dx", domain=mesh, subdomain_data=label_marker, subdomain_id = name_to_label["parenchyma"])
    parenchyma_vol = assemble(Constant(1)*dx_par)
    inflow = source_inflow*parenchyma_vol*m3tomL
    plt.figure(figsize=figsize)
    plt.plot(times,inflow , "-*")
    #plt.grid()
    plt.xlabel("t in s")
    plt.ylabel("inflow ml/s")
    #plt.title("net blood inflow")
    plt.savefig(f"{plot_dir}/{plotname}.pdf")


    plotname = "source_inflow_phases"
    phases_end = np.where(times==1)[0][0] 
    plt.figure(figsize=figsize)
    plt.plot(times[:phases_end +1],inflow[:phases_end +1] , "-*")
    plt.xlabel("t in s")
    plt.ylabel("inflow in ml/s")
    #plt.title("net blood inflow")
    for i,pt in enumerate(phases):
        plt.axvline(pt, c="red", ls=":")
        plt.annotate(f"({i+1})", (pt - 0.05, 9.0))
    plt.savefig(f"{plot_dir}/{plotname}.pdf")



    key_quantities["max_blood_inflow"] = inflow[end_idx:].max()
    key_quantities["min_blood_inflow"] = inflow[end_idx:].min()
    key_quantities["mean_blood_inflow"] = inflow[end_idx:].mean()


    results = {n:[] for n in names}
    times = times[start_idx:]
    #results.update({f"{n}_abs":[] for n in names})

    for n in names:
        for i in range(num_steps + 1 - start_idx):
            f = Function(V)
            infile.read_checkpoint(f, n, i + start_idx)
            results[n].append(f)
            #results[n + "_abs"].append( project(sqrt(inner(f,f)), V_abs ))
    infile.close()

    # aqueduct velocity

    plotname = "vel_aqueduct"
    title = "aqueduct velocity"
    point = "aqueduct"  
    var = "u"
    plt.figure(figsize=figsize)
    data = extract_cross_section(results[var], [Point(flatprobes[point])])*1e3
    data = data[:,0,:]
    tot = np.linalg.norm(data, axis=1)
    for i, comp in enumerate(["x","y","z"]):
        plt.plot(times, data[:,i], "-*", label=f"u_{comp}")
    plt.legend()
    #plt.grid()
    plt.xlabel("time in s")
    plt.ylabel("u in mm/s")
    #plt.title(title)
    plt.savefig(f"{plot_dir}/{plotname}.pdf")

    # compute outflow into spinal canal 
    ds_outflow = Measure("ds", domain=mesh, subdomain_data=boundary_marker, subdomain_id=spinal_outlet_id)
    n = FacetNormal(mesh)

    spinal_outflow = np.array([assemble(dot(u,n)*ds_outflow) for u in results["u"]])
    plt.figure(figsize=figsize)
    plt.plot(times, spinal_outflow*m3tomL, "*-", label="outflow into spinal canal")
    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("flow rate in ml/s")
    #plt.title("CSF outflow into spinal canal")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/spinal_out_CSF.pdf")

    key_quantities["max_spinal_outflow"] = spinal_outflow[end_idx:].max()
    key_quantities["min_spinal_outflow"] = spinal_outflow[end_idx:].min()
    key_quantities["mean_spinal_outflow"] = spinal_outflow[end_idx:].mean()

    # cumulative outflow in spinal canal

    cum_outflow = np.cumsum(spinal_outflow)*dt

    key_quantities["max_cum_spinal_outflow"] = cum_outflow[end_idx:].max()
    key_quantities["min_cum_spinal_outflow"] = cum_outflow[end_idx:].min()
    key_quantities["mean_cum_spinal_outflow"] = cum_outflow[end_idx:].mean()


    plt.figure(figsize=figsize)
    plt.plot(times, cum_outflow*m3tomL, "-*", label="cumulative outflow")
    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("V in ml")
    #plt.title("cumulative CSF outflow into spinal canal")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cum_spinal_out_CSF.pdf")

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
        key_quantities[f"max_flow_{fp[0]} -> {fp[1]}"] = flow[end_idx:].max()
        key_quantities[f"min_flow_{fp[0]} -> {fp[1]}"] = flow[end_idx:].min()
        key_quantities[f"mean_flow_{fp[0]} -> {fp[1]}"] = flow[end_idx:].mean()


    plt.figure(figsize=figsize)
    for name, flow in internal_flows.items():
        plt.plot(times, flow*m3tomL, "-*", label=name)
    #plt.plot(times, spinal_outflow*m3tomL, "-*", label="spinal outflow")

    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("flow rate in ml/s")
    #plt.title("ventricular CSF flow")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/ventr_CSF_flow.pdf")
    # max value according to Baladont: 0.2 mL/s


    plt.figure(figsize=figsize)
    for name, flow in internal_flows.items():
        cum_flow = np.cumsum(flow)*dt
        plt.plot(times, np.cumsum(flow)*dt*m3tomL, "*-", label=name)
        key_quantities[f"max_cum_flow_{name}"] = cum_flow[end_idx:].max()
        key_quantities[f"min_cum_flow_{name}"] = cum_flow[end_idx:].min()
        key_quantities[f"mean_cum_flow_{name}"] = cum_flow[end_idx:].mean()

    #plt.plot(times, cum_outflow*dt*m3tomL/50, label="cum spinal outflow (scaled: 1/50)")

    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("V in ml")
    #plt.title("cumulative ventricular CSF flow")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cum_CSF_flow.pdf")


    # compute parenchyma volume change
    ds_interf = Measure("dS", domain=mesh, subdomain_data=boundary_marker, subdomain_id=interface_id)
    dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    n = FacetNormal(mesh)("-")
    par_dV = np.array([assemble(dot(d("-"), n)*ds_interf + Constant(0.0)*dx) for d in results["d"]])
    plt.figure(figsize=figsize)
    plt.plot(times, par_dV*m3tomL, label="DV")
    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("dV in ml")
    #plt.title("parenchyma volume change")
    plt.savefig(f"{plot_dir}/par_vol_change.pdf")


    # displacement statistics

    #max_disp_over_time = np.array([norm(d.vector, "linf") for d in results["d_abs"]])
    def vecmax(d):
        for i, di in enumerate(d.split(deepcopy=True)):
            if i==0:
                vec_abs = di.vector()[:]**2
            else:
                vec_abs += di.vector()[:]**2
        return np.sqrt(vec_abs).max()

    max_disp_over_time = np.array([ vecmax(d) for d in results["d"]])
    #mean_disp_over_time = np.array( [assemble( d*dx(name_to_label["parenchyma"]) ) for d in results["d_abs"]])/parenchyma_vol
    plt.figure(figsize=figsize)
    plt.plot(times, max_disp_over_time)
    #plt.plot(times, mean_disp_over_time, "mean")
    plt.legend()
    plt.xlabel("time in s")
    plt.ylabel("max displacement [m]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/displ_over_time.pdf")

    key_quantities["max_displacement"] = max_disp_over_time.max()
    #key_quantities["max_mean_displacement"] = mean_disp_over_time.max()

    key_quantities = {k: float(v) for k,v in key_quantities.items()} 

    with open(key_quantities_file, "w") as key_q_file:
        yaml.dump(key_quantities, key_q_file)


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