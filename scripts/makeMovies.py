from braininversion.PostProcessing import (load_results_and_mesh,
                                           scalar_bars,
                                           compute_glob_stat,
                                           plot_partial_3D,
                                           create_movie)
import os
import numpy as np
import yaml


ventricular_system = ["lateral_ventricles", "foramina", "aqueduct", "third_ventricle", "fourth_ventricle",
                      "median_aperture"]

def run(config, mesh_name, sim_name):
    print("load data ...") 
    mesh_grid, sim_config, mesh_config, sim_file, source_expr =  load_results_and_mesh(mesh_name, sim_name)
    T = sim_config["T"]
    num_steps = sim_config["num_steps"]
    movie_path = f"results/{mesh_name}_{sim_name}/movies"
    try:
        os.mkdir(movie_path)
    except FileExistsError:
        pass
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    print("configure...")

    for sc in config["scenes"]:
        print(sc.keys())
        if "scalar_bar" in sc["static"].keys():
            sc["static"]["scalar_bar_args"] = scalar_bars[sc["static"].pop("scalar_bar")]
        if sc["mesh_parts"] == "ventricular_system":
            sc["mesh_parts"] = ventricular_system
        if sc["mesh_parts"] == "all_fluid":
            sc["mesh_parts"] = ventricular_system + ["csf"]
        if "clim" in sc["static"].keys():
            if sc["static"]["clim"] =="global":
                v_max =  compute_glob_stat(max, mesh_grid, sc["var"], sc["mesh_parts"], range(num_steps), sim_file, mesh_config)
                v_min =  compute_glob_stat(min, mesh_grid, sc["var"], sc["mesh_parts"], range(num_steps), sim_file, mesh_config)
                sc["static"]["clim"] = (v_min, v_max)

    # set image generator
    img_generator = lambda idx: plot_partial_3D(mesh_grid, idx, config["scenes"], 
                                                sim_file, mesh_config, cpos=config["camera_pos"], interactive=False)
    create_movie(f"{movie_path}/{config['name']}", times, source_expr, img_generator, 
                    fps=config["fps"], interpolate_frames=config["interpolate_frames"])

if __name__=="__main__":
    print("start movie creation...")
    config_file_path = sys.argv[1]
    mesh_name = sys.argv[2]
    sim_name = sys.argv[3]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    
    run(config, mesh_name, sim_name)

