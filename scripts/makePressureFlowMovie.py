from braininversion.PostProcessing import (load_results_and_mesh,
                                           scalar_bars,
                                           compute_glob_stat,
                                           extract_data,
                                           create_movie,
                                           scale_grid)
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import sys
import pyvista as pv
pv.set_plot_theme("document")


name = "PressureFlow"
fps = 5
interpFrames = 1
dist = 0.35
cpos = {"y":[(0, dist, 0), (0, 0, 0), (0, 0, 1)],
        "x":[(dist,0, 0), (0, 0, 0), (0, 0, 1)],
        "z": [(0, 0, dist*1.3), (0, 0, 0), (0, 1, 0)]}
origin = (0,0, 0.03)    
views = ["x","y","z"]
invert_dict = {"x":True, "y":True, "z":True}


sargs = dict(title_font_size=40,label_font_size=30,shadow=True,n_labels=3,
             italic=True,font_family="arial", height=0.3, vertical=True, position_y=0.02)
scalar_bars = {"left": dict(position_x=0.05, **sargs),
               "right": dict(position_x=0.8, **sargs)}
def symlog(x):
        """ Returns the symmetric log10 value """
        return np.sign(x) * np.log10(np.abs(x))

class ImageGenerator(object):
    def __init__(self, mesh_name, sim_name, times=None):
        mesh_grid, sim_config, mesh_config, sim_file, source_expr =  load_results_and_mesh(mesh_name, sim_name)
        T = sim_config["T"]
        num_steps = sim_config["num_steps"]
        try:
            os.mkdir(movie_path)
        except FileExistsError:
            pass
        self.source_expr = source_expr
        self.times = np.linspace(0, T, num_steps + 1)
        if times is None:
            self.time_indices = list(range(num_steps + 1))
        else:
            self.time_indices =  []
            for pt in times:
                self.time_indices.append( np.abs(pt-self.times).argmin() ) 
  
        print("start loading data...")
        csf_filled = [dom["name"] for dom in mesh_config["domains"] if dom["name"] not in ["parenchyma"]]
        self.data_u = extract_data(mesh_grid, ["u"], csf_filled,
                                    self.time_indices, sim_file, mesh_config)
        self.data_pF = extract_data(mesh_grid, ["pF"], csf_filled,
                                    self.time_indices, sim_file, mesh_config)
        print("finished loading data.")

        self.data_u.clip((0,0,-1), (0,0,-0.1), inplace=True, invert=True)
        self.data_pF.clip((0,0,-1), (0,0,-0.1), inplace=True, invert=True)
        
        scale_grid(self.data_pF, 1/132.32)
        scale_grid(self.data_u, 1000)


        self.u_range = self.get_range("u", self.data_u)
        self.pF_range = self.get_range("pF", self.data_pF)
        self.static_colorbar = False

        #self.phi_range = self.get_range("phi", self.data_phi)
        #self.pressure_range = [min(self.pF_range[0], self.phi_range[0]),
        #                       max(self.pF_range[1], self.phi_range[1])]

    def get_range(self, var, data):
        max_val = - np.inf
        min_val = + np.inf
        for time_idx in range(len(self.time_indices)):
            if var in ["u","d"]:
                rngmin = np.linalg.norm(data[f"{var}_{time_idx}"], axis=1).min()
                rngmax = np.linalg.norm(data[f"{var}_{time_idx}"], axis=1).max()
            else:
                rngmin, rngmax = data.get_data_range(f"{var}_{time_idx}")
            min_val = min([min_val, rngmin])
            max_val = max([max_val, rngmax])
        return [min_val, max_val]
        


    def generate_image(self, time_idx, view="x"):
        pF = f"pF_{int(time_idx)}"
        u = f"u_{int(time_idx)}"

        clipped_data_u = self.data_u.clip(view, origin=origin, invert=invert_dict[view])
        clipped_data_pF = self.data_pF.clip(view, origin=origin, invert=invert_dict[view])
        p = pv.Plotter(off_screen=True, notebook=False)
        clipped_data_u["logu"] = np.log10(np.linalg.norm(clipped_data_u[u], axis=1) + 1)
        arrows_u = clipped_data_u.glyph(scale="logu", factor=2e-2, orient=u, tolerance=0.01, absolute=False
                                        )
        if self.static_colorbar:
            clim = self.pF_range
        else:
            clim = None
        #p.add_mesh(arrows_u, scalars='GlyphScale', color ="white"), #clim=self.u_range,
                   #stitle="vel Magnitude [m/s]", scalar_bar_args = scalar_bars["right"])
        p.add_mesh(arrows_u, cmap ="tempo", scalars = arrows_u["logu"]**10 - 1, clim=(0 , 20), #clim=self.u_range,  scalars=u',
                   stitle="u [mm/s]", scalar_bar_args = scalar_bars["left"])
        p.add_mesh(clipped_data_pF, scalars=pF, cmap="balance", opacity=1.0, clim=clim,
                   scalar_bar_args = scalar_bars["right"], stitle="ICP [mmHg]") 

        p.camera_position = cpos[view] #camera position, focal point, and view up.
        return p, ()


def create_array_plot(path, time_indices, source_expr, img_gen_func, times):

    nind = len(time_indices)
    size = 8
    print("start array plot generation")
    fig, axes = plt.subplots(nind, 3, figsize=(size*3.3, nind*size))
    for j, idx in enumerate(time_indices):
        for i,view in enumerate(views):
            p, _ = img_gen_func(j, view=view)
            img = p.screenshot(transparent_background=True, return_img=True,
                               window_size=None,
                               filename=path + f"_{view}_{times[idx]:.3f}.png")
            p.clear()
            #fig2,ax2 = plt.subplots()
            #ax2.imshow(img)
            #ax2.axis('off')
            #fig2.tight_layout()
            #fig2.savefig(path + f"{view}_{times[idx]:.4f}.pdf")
            axes[j,i].imshow(img)
            axes[j,i].set_title(f"t = {times[idx]:.3f} s",fontsize=26)
            axes[j,i].axis('off')
    
    fig.tight_layout()
    fig.savefig(path + "_array_plot.pdf")

if __name__=="__main__":
    print("start movie creation...")
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2] 
    movie_path = f"results/{mesh_name}_{sim_name}/movies/{name}"

    #img_gen_func = lambda time_idx: img_gen.generate_image(time_idx)

    #create_movie(f"{movie_path}/{name}", img_gen.times, img_gen.source_expr, img_gen_func, 
    #                    fps=fps, interpolate_frames=interpFrames)


    phase_times = [0.13, .35, 0.56, 0.8] #[2.2, 2.4, 2.6, 2.8, 3.0]
    img_gen = ImageGenerator(mesh_name, sim_name, times=phase_times)

    img_gen_func = lambda time_idx, view: img_gen.generate_image(time_idx, view=view)
    create_array_plot(f"{movie_path}/{name}", img_gen.time_indices,
                      img_gen.source_expr, img_gen_func, img_gen.times)

    img_gen = ImageGenerator(mesh_name, sim_name)
    img_gen.static_colorbar = True
    img_gen_func = lambda time_idx: img_gen.generate_image(time_idx)

    create_movie(f"{movie_path}/{name}", img_gen.times, img_gen.source_expr, img_gen_func, 
                fps=fps, interpolate_frames=interpFrames)

