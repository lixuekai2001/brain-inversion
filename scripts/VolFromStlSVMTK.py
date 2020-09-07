import SVMTK as svm
import meshio
import numpy as np
import os
import sys
from itertools import combinations
import yaml
from pathlib import Path


def mesh_from_surfaces(config, outfile):
   
    scaling = config["scaling"]
    cwd = os.getcwd()
    try:
        os.mkdir(Path(outfile).parent)
    except FileExistsError:
        pass
    file_path = config["directory"] + "/{}.stl"
    print(file_path)
    csf_file = file_path.format("csf")
    parenchyma_file = file_path.format("parenchyma")

    components = config["ventricle_components"]

    tmp_file = '/tmp/brain.mesh'

    # --- Options ---
    # Load input file
    print("loading csf and parenchyma...")
    csf  = svm.Surface(csf_file)
    parenchyma = svm.Surface(parenchyma_file)
    csf.difference(parenchyma)
    parenchyma.smooth_taubin(100)
    csf.smooth_taubin(100)

    clip_plane = [0,0,1]
    z_val = -110.0
    parenchyma.clip(0,0,-1, -110, True)
    csf.clip(0,0,-1, -110, True)

    surfaces = [parenchyma, csf]
    comp_surfaces = []
    smap = svm.SubdomainMap()

    num_regions = 2 + len(components)
    smap.add(f'{10**(num_regions - 1):{"0"}{num_regions}}', 1)
    smap.add(f'{10**(num_regions - 2):{"0"}{num_regions}}', 2)
    region_count = 0

    for comp_name, comp_config in components.items():
        print(f"loading {comp_name}...")

        c = svm.Surface() 
        c = svm.Surface(file_path.format(comp_name))
        c.collapse_edges(0.8)
        c.smooth_taubin(comp_config["smooth"])
        c.adjust_boundary(comp_config["grow"])
        components[comp_name]["surface"] = c

        #c.smooth_laplace(config["smooth"][1])

        parenchyma.difference(c)
        csf.difference(c)
        comp_surfaces.append(c)
        smap.add(f'{10**region_count:{"0"}{num_regions}}', comp_config["id"])
        region_count+=1

    for s1,s2 in combinations(comp_surfaces, 2):
        s1.difference(s2)

    surfaces += comp_surfaces

    smap.print()
    # Create the volume mesh
    maker = svm.Domain(surfaces, smap)

    maker.create_mesh(config["edge_size"], config["cell_size"], config["facet_size"],
                      config["facet_angle"], config["facet_distance"],
                      config["cell_radius_edge_ratio"])

    if config["odt"]:
        print("start ODT mesh improvement...")
        maker.odt()
    if config["exude"]:
        print("start exude mesh improvement...")
        maker.exude()

    # Write output file 
    maker.save(tmp_file)
    mesh = meshio.read(tmp_file)

    new_mesh = meshio.Mesh(mesh.points*scaling, cells=[("tetra", mesh.get_cells_type("tetra"))],
                        cell_data={"subdomains":[mesh.cell_data_dict["medit:ref"]["tetra"]]})
    print(new_mesh)
    new_mesh.write(outfile)
    os.remove(tmp_file)

if __name__=="__main__":
    config_file_path = sys.argv[1]
    outfile = sys.argv[2]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    mesh_from_surfaces(config, outfile)