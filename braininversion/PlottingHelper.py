import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from collections.abc import Iterable
from fenics import *


style_dict = {"p_ana":{"c":"firebrick", "marker":"H", "ls":""},
              "p_obs":{"c":"red", "marker":">", "ls":""},
              "p_init":{"c":"navy","lw":2, "ls":"--"},
              "p_opt_robin" : {"c":"g", "ls":"--", "lw":3},
              "f_opt_robin":{"c":"g", "ls":"--", "lw":3},
              "f_opt_dirichlet":{"c":"orange", "ls":"-.", "lw":3},
              "p_opt_dirichlet" :{"c":"orange", "ls":"-.", "lw":3},
              "f_ana":{"c":"firebrick", "marker":"H", "ls":""},}

default = {"ls":"-."}
style_dict = defaultdict(lambda: default, style_dict)


def plot_pressures_and_forces_cross_section(pressures, forces, time_idx,
                                            x_coords):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    plt.subplot(1,2,1)

    for pressure_name, pressure_data in pressures.items():

        plt.plot(x_coords, pressure_data[time_idx,:],
                 **style_dict[pressure_name],
                 label=pressure_name)

    plt.legend(loc="upper center")
    plt.xlabel("x in m")
    plt.ylabel("p in mmHg")
    plt.grid()
    plt.subplot(1,2,2)

    for force_name, forces_data in forces.items():

        plt.plot(x_coords, forces_data[time_idx,:],
                 **style_dict[force_name],
                 label=force_name)

    #plt.ylabel("g in 1/s")
    plt.xlabel("x in m")
    plt.grid()
    plt.legend(loc="upper center")

def plot_pressures_cross_section(pressures, time_idx,
                                            x_coords):
    plt.figure(figsize=(6,6))
    for pressure_name, pressure_data in pressures.items():
        plt.plot(x_coords, pressure_data[time_idx,:],
                 **style_dict[pressure_name],
                 label=pressure_name)

    plt.legend(loc="upper left")
    plt.xlabel("x in m")
    plt.ylabel("p in mmHg")
    plt.grid()

  

def plot_pressures_timeslice(pressures, point_idx, times):
    plt.figure(figsize=(6,6))

    for pressure_name, pressure_data in pressures.items():

        plt.plot(times, pressure_data[:,point_idx],
                 **style_dict[pressure_name],
                 label=pressure_name)

    plt.legend(loc="upper left")
    plt.xlabel("t in s")
    plt.ylabel("p in mmHg")
    plt.grid()

    
def plot_pressures_and_forces_timeslice(pressures, forces, point_idx, times):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    plt.subplot(1,2,1)
    #plt.title(f"Point: ({slice_points[i].x():.3f}, {slice_points[i].y():.3f})")
    for pressure_name, pressure_data in pressures.items():

        plt.plot(times, pressure_data[:,point_idx],
                 **style_dict[pressure_name],
                 label=pressure_name)

    plt.legend(loc="upper left")
    plt.xlabel("t in s")
    plt.ylabel("p in mmHg")
    plt.grid()
    plt.subplot(1,2,2)

    for force_name, forces_data in forces.items():
        plt.plot(times, forces_data[:,point_idx],
                 **style_dict[force_name],
                 label=force_name)

    #plt.ylabel("g in 1/s")
    plt.xlabel("t in s")
    plt.grid()
    plt.legend(loc="upper center")


def extract_cross_section(function, points, times=None):
    if isinstance(function, list):
        return extract_cross_section_from_list(function, points)
    if isinstance(function, Expression):
        if times is None:
            raise ValueError("times must be specified for expression cross section")
        return extract_cross_section_from_expression(function,
                                                     points, times)
 

def extract_cross_section_from_list(functions, points):
    npoints = len(points)
    nt = len(functions)
    value_dim = functions[0].value_dimension(0)
    values = np.ndarray((nt, npoints, value_dim))
    for k, p in enumerate(points):
        for i,f in enumerate(functions):
            try:
                values[i, k, :] = f(p) 
            except RuntimeError:
                values[i, k, :] = np.repeat(np.nan, value_dim)
    return values

def extract_cross_section_from_expression(expression, points, times):
    npoints = len(points)
    nt = len(times)
    try:
        value_dim = expression[0].value_dimension(0)
    except:
        value_dim = 1
    values = np.ndarray((nt, npoints, value_dim))
    for k, p in enumerate(points):
        for i,t in enumerate(times):
                expression.t = t
                values[i, k, :] = expression(p) 
    return values



def compute_order(error, h):
    h = np.array(h)
    err_ratio = np.array(error[:-1]) / np.array(error[1:])
    return np.log(err_ratio)/np.log(h[:-1] / h[1:])