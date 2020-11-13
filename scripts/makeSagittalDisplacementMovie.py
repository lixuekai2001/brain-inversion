from braininversion.PostProcessing import (load_results_and_mesh,
                                           scalar_bars,
                                           compute_glob_stat,
                                           extract_data,
                                           create_movie)
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import sys
import pyvista as pv

name = "SagittalDisplacement"
fps = 5
interpFrames = 1
dist = 0.4
cpos = {"y":[(0, dist, 0), (0, 0, 0), (0, 0, 1)],
        "x":[(dist,0, 0), (0, 0, 0), (0, 0, 1)],
        "z": [(0, 0, dist*1.3), (0, 0, 0), (0, 1, 0)]}
origin = (0,0, 0.03)    
views = ["x","y","z"]
invert_dict = {"x":True, "y":True, "z":True}


sargs = dict(title_font_size=20,label_font_size=16,shadow=True,n_labels=3,
             italic=True,font_family="arial", height=0.4, vertical=True, position_y=0.05)
scalar_bars = {"left": dict(position_x=0.05, **sargs),
               "right": dict(position_x=0.95, **sargs)}

class ImageGenerator(object):
    def __init__(self, mesh_name, sim_name):
        mesh_grid, sim_config, mesh_config, sim_file, source_expr =  load_results_and_mesh(mesh_name, sim_name)
        T = sim_config["T"]
        num_steps = sim_config["num_steps"]
        try:
            os.mkdir(movie_path)
        except FileExistsError:
            pass
        self.source_expr = source_expr
        self.times = np.linspace(0, T, num_steps + 1)
        print("start loading data...")
        csf_filled = [dom["name"] for dom in mesh_config["domains"] if dom["name"] not in ["parenchyma"]]
        self.data_u = extract_data(mesh_grid, ["u"], csf_filled,
                                    list(range(num_steps + 1)), sim_file, mesh_config)
        self.data_d = extract_data(mesh_grid, ["d"], ["parenchyma"],
                                    list(range(num_steps + 1)), sim_file, mesh_config)
        print("finished loading data.")

        self.data_u.clip((0,0,-1), (0,0,-0.1), inplace=True, invert=True)
        self.data_d.clip((0,0,-1), (0,0,-0.1), inplace=True, invert=True)

        
        self.d_range = self.get_range("d", self.data_d)
        self.u_range = self.get_range("u", self.data_u)

        #self.phi_range = self.get_range("phi", self.data_phi)
        #self.pressure_range = [min(self.pF_range[0], self.phi_range[0]),
        #                       max(self.pF_range[1], self.phi_range[1])]

    def get_range(self, var, data):
        max_val = - np.inf
        min_val = + np.inf
        for time_idx in range(len(self.times)):
            rng = data.get_data_range(f"{var}_{time_idx}")
            min_val = min([min_val, rng[0]])
            max_val = max([max_val, rng[1]])
        return [min_val, max_val]
        

    def generate_image(self, time_idx, view="x"):
        d = f"d_{int(time_idx)}"
        u = f"u_{int(time_idx)}"

        clipped_data_u = self.data_u.clip(view, origin=origin, invert=invert_dict[view])
        clipped_data_d = self.data_d.clip(view, origin=origin, invert=invert_dict[view])
        p = pv.Plotter(off_screen=True, notebook=False)
        arrows_d = clipped_data_d.glyph(scale=d, factor=60, orient=d)
        p.add_mesh(arrows_d, scalars='GlyphScale', cmap ="speed", clim=self.d_range,
                   stitle="disp Magnitude [m]", scalar_bar_args = scalar_bars["left"])
        p.add_mesh(clipped_data_d, color="red", opacity=0.5) 

        arrows_u = clipped_data_u.glyph(scale=u, factor=0.5, orient=u)
        p.add_mesh(arrows_u, scalars='GlyphScale', cmap ="amp", clim=self.u_range,
                   stitle="vel Magnitude [m/s]", scalar_bar_args = scalar_bars["right"])
        p.add_mesh(clipped_data_u, color="white", opacity=0.2) 
        p.camera_position = cpos[view] #camera position, focal point, and view up.
        return p, ()


def create_array_plot(path, time_indices, source_expr, img_gen_func, times):
    pv.set_plot_theme("document")

    nind = len(time_indices)
    size = 8

    fig, axes = plt.subplots(3, nind, figsize=(nind*size, size*2))
    for j, idx in enumerate(time_indices):
        for i,view in enumerate(views):
            p, _ = img_gen_func(idx, view=view)
            img = p.screenshot(transparent_background=True, return_img=True, window_size=None)
            axes[i,j].imshow(img)
            axes[i,j].set_title(f"t = {times[idx]:.4f} s")
    plt.tight_layout()
    plt.savefig(path + "_array_plot.pdf")

if __name__=="__main__":
    print("start movie creation...")
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2] 
    movie_path = f"results/{mesh_name}_{sim_name}/movies/{name}"

    img_gen = ImageGenerator(mesh_name, sim_name)
    img_gen_func = lambda time_idx: img_gen.generate_image(time_idx)

    #create_movie(f"{movie_path}/{name}", img_gen.times, img_gen.source_expr, img_gen_func, 
    #                    fps=fps, interpolate_frames=interpFrames)

    img_gen_func = lambda time_idx, view: img_gen.generate_image(time_idx, view=view)

    phase_times = [2.1, 2.3, 2.6, 3.0]
    array_plot_times = []
    for pt in phase_times:
        array_plot_times.append( np.abs(pt-img_gen.times).argmin() )    
    create_array_plot(f"{movie_path}/{name}", array_plot_times, img_gen.source_expr, img_gen_func, img_gen.times)
