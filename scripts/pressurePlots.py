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

names = ["pF", "pP", "phi"]

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
            "d":"porous", "u":"fluid"}

def plot_scalar_time_evolution(point_var_list, mesh_config, results, ylabel,times, plot_dir, scale=1):
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

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)]
    
def plot_cross_section(p1, p2, n_crossP, variables, results, time_idx, filter_dict, plot_dir, scale=1):
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


if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    plot_dir = f"results/{mesh_name}_{sim_name}/pressure_plots/"

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


    fluid_filter = subdomainFilter(subdomain_marker, fluid_id, degree=0)
    por_filter = subdomainFilter(subdomain_marker, porous_id, degree=0)

    DG = FunctionSpace(mesh, "DG", 0)
    fluid_filter = interpolate(fluid_filter, DG)
    por_filter = interpolate(por_filter, DG)

    W = FunctionSpace(mesh, "CG", 1)

    infile = XDMFFile(sim_file)

    results = {n:[] for n in names}
    for n in names:
        for i in range(num_steps + 1):
            f = Function(W)
            infile.read_checkpoint(f, n, i)
            results[n].append(f)
    infile.close()

    # create plots:

    ylabel = "p in mmHg"
    point_var_list = [("pF","front_sas"),("phi","front_parenchyma"),
                      ("pF","lateral_ventricles")]
    plot_scalar_time_evolution(point_var_list, mesh_config, results,
                              ylabel, times, plot_dir, scale=1/mmHg2Pa)

    point_var_list = [("pF","back_sas"),("phi","back_parenchyma"),
                      ("pF","lateral_ventricles")]
    plot_scalar_time_evolution(point_var_list, mesh_config, results,
                               ylabel, times, plot_dir, scale=1/mmHg2Pa)

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

    n_crossP = 300
    time_indices = np.linspace(1, num_steps, 10, dtype=np.int)
    p1 = "left_sas"
    p2 = "right_sas"
    plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                    results, time_indices, filter_dict, plot_dir, scale=1/mmHg2Pa)
    p1 = "front_parenchyma"
    p2 = "back_parenchyma"
    plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                    results, time_indices, filter_dict, plot_dir, scale=1/mmHg2Pa)

    p1 = "lateral_ventricles"
    p2 = "top_sas"
    plot_cross_section(p1, p2, n_crossP, ["pF", "phi", "pP"],
                    results, time_indices, filter_dict, plot_dir, scale=1/mmHg2Pa)



