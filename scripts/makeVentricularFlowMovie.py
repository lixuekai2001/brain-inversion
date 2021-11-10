from braininversion.PostProcessing import (load_results_and_mesh,
                                           scalar_bars,
                                           compute_glob_stat,
                                           extract_data,
                                           create_movie)
import os
import numpy as np
import yaml
import sys
import pyvista as pv

name = "VentricularFlow"
fps = 10
interpFrames = 1
cpos = [ [0.2, 0.15, -0.01], [0, 0, 0], [0.0, 0.0, 1.0] ]

class VentricularFlowImageGenerator(object):
    def __init__(self, mesh_name, sim_name):
        mesh_grid, sim_config, mesh_config, sim_file, source_expr =  load_results_and_mesh(mesh_name, sim_name)
        T = sim_config["T"]
        num_steps = sim_config["num_steps"]
        try:
            os.mkdir(movie_path)
        except FileExistsError:
            pass
        dt = T/num_steps
        self.source_expr = source_expr
        self.times = np.linspace(0, T, num_steps + 1)
        ventricular_system = [dom["name"] for dom in mesh_config["domains"] if dom["name"] not in ["csf","parenchyma"]]
        self.data = extract_data(mesh_grid, ["u", "pF"], ventricular_system,
                                list(range(num_steps + 1)), sim_file, mesh_config)
        self.data.clip((0,0,-1), (0,0,-0.05), inplace=True, invert=True)
        self.pF_range = self.get_range("pF")
        self.u_range = self.get_range("u")

    def get_range(self, var):
        max_val = - np.inf
        min_val = + np.inf
        for time_idx in range(len(self.times)):
            rng = self.data.get_data_range(f"{var}_{time_idx}")
            min_val = min([max_val, rng[0]])
            max_val = max([max_val, rng[1]])
        return [min_val, max_val]
        

    def generate_image(self, time_idx):
        p = pv.Plotter(off_screen=True, notebook=False)
        u = f"u_{int(time_idx)}"
        pF = f"pF_{int(time_idx)}"
        arrows = self.data.glyph(scale=u, factor=1, orient=u)
        p.add_mesh(arrows, scalars=pF, cmap="balance", clim=self.pF_range) #stitle=f"{var} Magnitude",
        #p.add_mesh(self.data, color="white", opacity=0.2) #stitle=f"{var} Magnitude",
        p.camera_position = cpos #camera position, focal point, and view up.
        return p, self.pF_range

def create_array_plot(path, time_indices, source_expr, img_gen_func, times):
    pv.set_plot_theme("document")

    nind = len(time_indices)
    size = 8

    fig, axes = plt.subplots(nind, 3, figsize=(size*3.3, nind*size))
    for j, idx in enumerate(time_indices):
        for i,view in enumerate(views):
            p, _ = img_gen_func(j, view=view)
            img = p.screenshot(transparent_background=True, return_img=True,
                               window_size=None,
                               filename=path + f"_{view}_{times[idx]:.3f}.png")
            p.clear()
            #plt.figure(i*j)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig(path + f"{view}_{times[idx]:.4f}.pdf")
            #plt.tight_layout()
            axes[j,i].imshow(img)
            axes[j,i].set_title(f"t = {times[idx]:.3f} s", fontsize=26)
            axes[j,i].axis('off')
    
    fig.tight_layout()
    fig.savefig(path + "_array_plot.pdf")



if __name__=="__main__":
    print("start movie creation...")
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2] 
    movie_path = f"results/{mesh_name}_{sim_name}/movies/{name}"

    img_gen = VentricularFlowImageGenerator(mesh_name, sim_name)
    img_gen_func = lambda time_idx: img_gen.generate_image(time_idx)

    create_movie(f"{movie_path}/{name}", img_gen.times, img_gen.source_expr, img_gen_func, 
                        fps=fps, interpolate_frames=interpFrames)

    phase_times = [0.13, .35, 0.56, 0.8] #[2.2, 2.4, 2.6, 2.8, 3.0]
    img_gen = VentricularFlowImageGenerator(mesh_name, sim_name, times=phase_times)

    img_gen_func = lambda time_idx, view: img_gen.generate_image(time_idx, view=view)
    create_array_plot(f"{movie_path}/{name}", img_gen.time_indices, img_gen.source_expr, img_gen_func, img_gen.times)
