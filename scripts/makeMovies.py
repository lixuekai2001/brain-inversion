import matplotlib
import matplotlib.pyplot as plt
from fenics import *
from multiphenics import *
import numpy as np
from braininversion.IOHandling import (read_mesh_from_h5, write_to_xdmf, 
                                       xdmf_to_unstructuredGrid, read_xdmf_timeseries)
from braininversion.PlottingHelper import (plot_pressures_and_forces_timeslice, 
                                           plot_pressures_and_forces_cross_section,
                                           extract_cross_section, style_dict)
from braininversion.SourceExpression import get_source_expression
from matplotlib.backends.backend_agg import FigureCanvasAgg
import yaml
import pyvista as pv
from pathlib import Path
import imageio
from cmocean import cm
import sys

ventricular_system = ["lateral_ventricles", "foramina", "aqueduct", "third_ventricle", "fourth_ventricle",
                      "median_aperture"]

sargs = dict(title_font_size=20,label_font_size=16,shadow=True,n_labels=3,
             italic=True,font_family="arial", height=0.4, vertical=True, position_y=0.05)
scalar_bars = {"left": dict(position_x=0.05, **sargs),
               "right": dict(position_x=0.95, **sargs)}

try:
    mesh_file = snakemake.input["subdomain_file"]
    sim_file = snakemake.input["sim_results"]
    sim_config_file = snakemake.input["sim_config_file"]
except NameError:
    mesh = "MRIExampleSegmentation_Ncoarse"
    sim_name = "stdBrainSim"
    mesh_file = f"../meshes/{mesh}/{mesh}.xdmf"
    sim_file = f"../results/{mesh}_{sim_name}/results.xdmf"
    sim_config_file = f"../results/{mesh}_{sim_name}/config.yml"

mesh_name = f"{Path(mesh_file).parent}/{Path(mesh_file).stem}"
mesh_config_file = f"{mesh_name}_config.yml"

boundary_file = f"{mesh_name}_boundaries.xdmf"
label_boundary_file = f"{mesh_name}_label_boundaries.xdmf"
label_file = f"{mesh_name}_labels.xdmf"

sim_file_old = f"../results/{mesh}_{sim_name}/results_old.xdmf"
movie_path = f"../results/{mesh}_{sim_name}/movies/"

with open(sim_config_file) as conf_file:
    sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    
with open(mesh_config_file) as conf_file:
    mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

T = sim_config["T"]
num_steps = sim_config["num_steps"]

mmHg2Pa = 132.32
porous_id = 1
dt = T/num_steps
times = np.linspace(0, T, num_steps + 1)
mesh_grid = pv.read(label_file)

infile_mesh = XDMFFile(mesh_file)
mesh = Mesh()
infile_mesh.read(mesh)
gdim = mesh.geometric_dimension()
subdomain_marker = MeshFunction("size_t", mesh, gdim)
infile_mesh.read(subdomain_marker)#, "subdomains"
infile_mesh.close()



def scale_grid(grid, fac):
    for name, pdata in grid.point_arrays.items():
        grid.point_arrays[name] *= fac
        
def sum_grids(grid1, grid2):
    for name, pdata in grid1.point_arrays.items():
        grid1.point_arrays[name] += grid2.point_arrays[name]


def compute_stat(stat, mg, var, parts, idx):
    data = extract_data(mg, var, parts, int(idx))
    return stat(data[var])

def compute_glob_stat(stat, mg, var, parts, indices):
    return stat([compute_stat(stat, mg, var, parts, idx) for idx in indices])

def extract_and_interpolate(mg, var, parts, idx):
    if np.isclose(idx%1, 0.0):
        return extract_data(mg, var, parts, int(idx))
    
    floor =  int(np.floor(idx))
    ceil =  int(np.ceil(idx))
    data1 = extract_data(mg, var, parts, floor)
    data2 = extract_data(mg, var, parts, ceil)
    
    scale_grid(data1, idx - floor)
    scale_grid(data2, ceil - idx)
    sum_grids(data1, data2)
    return data1

def extract_data(mg, var, parts, idx):
    mg = mg.copy()
    grid = xdmf_to_unstructuredGrid(sim_file_old, variables=[var], idx=[idx])
    # add new data to mesh
    for name, data in grid.point_arrays.items():
        print()
        mg.point_arrays[name] = grid.point_arrays[name]
    #filter parts:
    dom_meshes= []
    for dom in mesh_config["domains"]:
        if dom["name"] in parts:
            dom_meshes.append(mg.threshold([dom["id"],dom["id"]],
                                                  scalars="subdomains"))
    merged = dom_meshes[0].merge(dom_meshes[1:])
    return merged


def plot_partial_3D(mg, idx, scenes, cpos, interactive=False):
    if interactive:
        p = pv.PlotterITK()
    else:
        p = pv.Plotter(off_screen=True, notebook=False)
    max_val = -np.inf
    min_val = np.inf
    data_dict = {}
    for scene in scenes:
        var = scene["var"]
        parts = scene["mesh_parts"]
        data = extract_and_interpolate(mg, var, parts, idx)
        if "clip" in scene.keys():
            try:
                data = data.clip(*scene["clip"])
            except:
                data = data.clip(**scene["clip"])
        if "slice" in scene.keys():
            try:
                data = data.slice(*scene["slice"])
            except:
                data = data.slice(**scene["slice"])
        data_dict[var] = data
        if "arrow" in scene.keys():
            continue
        max_val = max(max_val, data[var].max())
        min_val = min(min_val, data[var].min())
    
    for i, scene in enumerate(scenes):
        var = scene["var"]
        parts = scene["mesh_parts"]
        data =  data_dict[var]
        if "warp" in scene.keys():
            data = data.warp_by_vector(var, scene["warp_fac"])
        if interactive:
            options = scene["interactive"]
        else:
            options = scene["static"]
            if "clim" not in options.keys() or options["clim"] is None:
                pass
                #options["clim"] = (min_val, max_val)
        if "arrow" in scene.keys():
            vec_scale = scene["vec_scale"]
            arrows = data.glyph(scale=var, factor=vec_scale, orient=var)
            p.add_mesh(arrows,**options)#, lighting=False) #stitle=f"{var} Magnitude",
        else:
            p.add_mesh(data, scalars=var,**options)
    #camera position, focal point, and view up.
    p.camera_position = cpos
    return p, (min_val, max_val)


def repeat(vals):
    l = len(vals)
    for i in range(1, l):
        s = vals[0:i]
        if np.isclose(s*(l//i) + vals[:l%i],  vals).all():
            return s

def plot_source_rgb_raw(source_series,times, t, size, dpi):
    fig = plt.Figure(figsize=size, dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(times, source_series)
    ax.axvline(t, color="red")
    ax.set_xlabel("t in s")
    ax.set_ylabel("g in 1/s")
    ax.grid()
    fig.tight_layout()
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    data = np.asarray(buf)
    return data[:,:,:]


# create video
def create_movie(path, times, plot_generator, fps=10, interpolate_frames=1):
    frames = []
    for i,t in enumerate(times):
        for k in range(interpolate_frames):
            p,_= plot_generator(i + k/interpolate_frames)
            p.show(interactive=False, auto_close=False)
            img = p.screenshot(transparent_background=True, return_img=True, window_size=None)
            p.close()
            size = (4,3)
            dpi = 70
            miniplot = plot_source_rgb_raw(source_series[:rep_idx + 1], times[:rep_idx + 1],
                                           (t + k/interpolate_frames*dt)%times[rep_idx] , 
                                           size, dpi)
            x,y,z = miniplot.shape
            img[:x,:y,:] = miniplot
            frames.append(img)
            if i==len(times) - 1:
                break

    mwriter = imageio.get_writer(path, fps=fps)
    for frame in frames:
        mwriter.append_data(frame)
    mwriter.close()


def create_all_movies(config):
    for movie in config["movies"]:
        # update and replace configuration
        for sc in movie["scenes"]:
            print(sc.keys())
            if "scalar_bar" in sc["static"].keys():
                sc["static"]["scalar_bar_args"] = scalar_bars[sc["static"].pop("scalar_bar")]
            if sc["mesh_parts"] == "ventricular_system":
                sc["mesh_parts"] = ventricular_system
            if sc["mesh_parts"] == "all_fluid":
                sc["mesh_parts"] = ventricular_system + ["csf"]
            if "clim" in sc["static"].keys():
                if sc["static"]["clim"] =="global":
                    v_max =  compute_glob_stat(max, mesh_grid, sc["var"], sc["mesh_parts"], range(num_steps))
                    v_min =  compute_glob_stat(min, mesh_grid, sc["var"], sc["mesh_parts"], range(num_steps))
                    sc["static"]["clim"] = (v_min, v_max)
        # set image generator
        img_generator = lambda idx: plot_partial_3D(mesh_grid, idx, movie["scenes"], cpos=movie["camera_pos"],
                                                    interactive=False)
        create_movie(f"{movie_path}/{movie['name']}.mp4", times, img_generator, 
                     fps=movie["fps"], interpolate_frames=movie["interpolate_frames"])

if __name__=="__main__":
    source_conf = sim_config["source_data"]
    source_expr = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)
    source_series = []
    for t in times:
        source_expr.t=t
        source_series.append(source_expr(Point([0]*gdim)) )
    sub_series = repeat(source_series)
    rep_idx = len(sub_series) 
    config_file_path = sys.argv[1]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    create_all_movies(config)

