import SVMTK as svm
import meshio
import numpy as np
import os
import sys
from itertools import combinations
import yaml
from pathlib import Path

base_components = ["csf", "parenchyma"]
tmp_file = '/tmp/brain.mesh'


def mesh_from_surfaces(config, outfile):
   
    scaling = config["scaling"]
    cwd = os.getcwd()
    try:
        os.mkdir(Path(outfile).parent)
    except FileExistsError:
        pass
    file_path = config["directory"] + "/{}.stl"
    domains = config["domains"]
    name_to_label = {dom["name"]:dom["id"] for dom in domains}

    csf_file = file_path.format("csf")
    parenchyma_file = file_path.format("parenchyma")

    print("loading csf and parenchyma...")
    csf  = svm.Surface(csf_file)
    parenchyma = svm.Surface(parenchyma_file)

    parenchyma.adjust_boundary(config["surface_processing"]["parenchyma"]["grow"])
    parenchyma.smooth_taubin(config["surface_processing"]["parenchyma"]["smooth_taubin"])
    parenchyma.clip(*config["surface_processing"]["parenchyma"]["clip"], True)

    csf.adjust_boundary(config["surface_processing"]["csf"]["grow"])
    csf.smooth_taubin(config["surface_processing"]["csf"]["smooth_taubin"])
    csf.clip(*config["surface_processing"]["csf"]["clip"], True)

    csf.difference(parenchyma)
    
    surfaces = [parenchyma, csf]
    comp_surfaces = []
    smap = svm.SubdomainMap()
    num_regions = len(domains)
    smap.add(f'{10**(num_regions - 1):{"0"}{num_regions}}',
             name_to_label["parenchyma"])
    smap.add(f'{10**(num_regions - 2):{"0"}{num_regions}}',
             name_to_label["csf"])

    region_count = num_regions -3 
    for dom in domains:
        comp_name = dom["name"]
        if comp_name in base_components:
            continue
        surface_processing = config["surface_processing"][comp_name]

        print(f"loading {comp_name}...")
        c = svm.Surface() 
        c = svm.Surface(file_path.format(comp_name))
        c.adjust_boundary(surface_processing["grow"])
        if "smooth_taubin" in surface_processing.keys():
             c.smooth_taubin(surface_processing["smooth_taubin"])
        if "smooth_shape" in surface_processing.keys():
            c.smooth_shape(*surface_processing["smooth_shape"])
        parenchyma.difference(c)
        c.difference(csf)
        comp_surfaces.append(c)
        smap.add(f'{10**region_count:{"0"}{num_regions}}', dom["id"])
        region_count-=1

    print("start difference")
    for s1,s2 in combinations(comp_surfaces, 2):
        s1.difference(s2)
        
    print("finshed difference")

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