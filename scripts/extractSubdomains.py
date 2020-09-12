import meshio
import sys
from fenics import *
import numpy as np
from braininversion.meshes import (generate_subdomain_restriction, 
                                   extract_internal_interface,set_external_boundaries)
from pathlib import Path
import yaml
import re

# subdomain ids
fluid_id = 2
porous_id = 1

# boundary ids
interface_id = 1
rigid_skull_id = 2
spinal_outlet_id = 3
fix_stem_id = 4

domain_ids = {"fluid":fluid_id, "porous":porous_id}

#label_refine = [{"label":8, "iterations":1, "target_size": 0.002}]

# read back in and extract boundaries and subdomains
def extract_subdomains_from_physiological_labels(mesh_file):

    mesh_dir = Path(mesh_file).parent
    mesh_name = Path(mesh_file).stem
    mesh_path = f"{mesh_dir}/{mesh_name}"
    try:
        config_file_path = f"config_files/{mesh_name}.yml"
        with open(config_file_path) as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        config_file_path = f"config_files/{re.sub('_N[0-9]*.', '', mesh_name)}.yml"
        with open(config_file_path) as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

    label_to_subdomain = {dom["id"]:dom["type"] for dom in config["domains"]}
    name_to_label = {dom["name"]:dom["id"] for dom in config["domains"]}
    if "refinement" in config.keys():
        refinement = config["refinement"]
    else:
        refinement = []

    

    file = XDMFFile(f"{mesh_path}_labels.xdmf")
    mesh = Mesh()
    file.read(mesh)
    gdim = mesh.geometric_dimension()

    labels = MeshFunction("size_t", mesh, gdim, 0)
    file.read(labels, "subdomains")
    file.close()

    #print(labels.array()[:].sum())
    #mesh = refine(mesh)
    #labels = adapt(labels, mesh)
    #print(labels.array()[:].sum())


    for refine_config in refinement:
        refine_iter = refine_config["iterations"]
        label = name_to_label[refine_config["name"]]
        for i in range(refine_iter):
            refine_marker = MeshFunction("bool", mesh, gdim, False)
            diam = CellDiameter(mesh)
            DG0 = FunctionSpace(mesh, "DG", 0)
            diam_func = project(diam, DG0)
            label_crit = labels.array()[:]==label
            size_crit = diam_func.vector()[:] > refine_config["target_size"]
            print(f"refining {(label_crit & size_crit).sum()} of {label_crit.sum()} cells in {refine_config['name']}...")
            refine_marker.array()[label_crit & size_crit] = True
            mesh = refine(mesh, refine_marker)
            labels = adapt(labels, mesh)
    print("finished refinement...")
    file = XDMFFile(f"{mesh_path}_labels.xdmf")
    labels.rename("subdomains", "subdomains")
    file.write(labels)
    file.close()

    subdomains = MeshFunction("size_t", mesh, gdim, 0)

    for label, domain in label_to_subdomain.items():
        subdomains.array()[labels.array()==label] = domain_ids[domain]


    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)
    boundaries = MeshFunction("size_t", mesh, gdim -1, 0)
    label_boundaries = MeshFunction("size_t", mesh, gdim -1, 0)

    z_bottom = mesh.coordinates()[:,2].min()

    # set rigid skull boundary
    rigid_skull = CompiledSubDomain("on_boundary")
    rigid_skull.mark(boundaries, rigid_skull_id)

    # set internal interface
    extract_internal_interface(mesh,subdomains, boundaries, interface_id=interface_id)
    extract_internal_interface(mesh, labels, label_boundaries)


    # add further external boundaries
    set_external_boundaries(mesh, subdomains, boundaries, porous_id, fix_stem_id, lambda x: True)
    set_external_boundaries(mesh, subdomains, boundaries, fluid_id, spinal_outlet_id, lambda x: np.isclose(x[2], z_bottom, rtol=8e-3) )

    boundaries_outfile = XDMFFile(mesh_path + "_boundaries.xdmf")
    boundaries_outfile.write(boundaries)
    boundaries_outfile.close()

    label_boundaries_outfile = XDMFFile(mesh_path + "_label_boundaries.xdmf")
    label_boundaries_outfile.write(label_boundaries)
    label_boundaries_outfile.close()
    
    subdomains_outfile = XDMFFile(mesh_path + ".xdmf")
    subdomains_outfile.write(subdomains)
    subdomains_outfile.close()

    fluid_restr = generate_subdomain_restriction(mesh, subdomains, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, subdomains, porous_id)
    fluid_restr._write(mesh_path + "_fluid.rtc.xdmf")
    porous_restr._write(mesh_path + "_porous.rtc.xdmf")

if __name__=="__main__":
    mesh_file = sys.argv[1]
    extract_subdomains_from_physiological_labels(mesh_file)