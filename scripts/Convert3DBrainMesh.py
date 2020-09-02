import numpy as np
import pyvista as pv
import numpy as np
import meshio
from fenics import *
from braininversion.meshes import (jmesh_to_vtk, generate_subdomain_restriction, 
                                   extract_internal_interface,set_external_boundaries)
from pathlib import Path
import yaml
import argparse
import os

def write_mesh_per_subdomain(path, subdomains, mesh):
        f = XDMFFile( path + "_fluid.xdmf")
        f.write(SubMesh(mesh, subdomains, fluid_id))
        f.close()
        f = XDMFFile( path + "_porous.xdmf")
        f.write(SubMesh(mesh, subdomains, porous_id))
        f.close()


def convert_mesh(config):
    # set parameter
    path = config["path"]
    scaling = config["scaling"]
    extract_subd = config["extract_subd"]
    add_spinal_outlet = config["add_spinal_outlet"]
    if add_spinal_outlet:
        spinal_outlet_radius =  config["spinal_outlet_radius"]
        spinal_outlet_midpoint = config["spinal_outlet_midpoint"]
        def spinal_outlet(point):
            crit = (((spinal_outlet_midpoint - point)**2).sum() 
                    < spinal_outlet_radius**2)
            return crit

    # default configuration
    infile_dict_mat = {"CSF":2, "WM":3, "GM":4, "skull":1, "scalp":0, "other":10}  
    infile_dict_jmsh = {"CSF":3, "WM":4, "GM":5, "skull":2, "scalp":1, "other":0}
    #
    fluid_id = 2
    porous_id = 1
    outfile_dict = {"fluid_id":fluid_id, "porous_id":porous_id}

    # boundary ids
    interface_id = 1
    rigid_skull_id = 2
    spinal_outlet_id = 3
    fix_stem_id = 4

    # setup path and filename:

    p = Path(path)
    stem = p.stem
    new_dir = Path(f"../meshes/{stem}/")
    new_dir.mkdir(exist_ok=True)
    outfile = f"../meshes/{stem}/{stem}"
    if p.suffix==".jmsh":
        infile_dict = infile_dict_jmsh
    elif  p.suffix==".mat":
        infile_dict = infile_dict_mat

    # extract subdomains and write to vtk and xdmf
    jmesh_to_vtk(path, outfile + ".vtk", infile_dict, outfile_dict, reduce=extract_subd, scaling=scaling)
    mesh = meshio.read(outfile + ".vtk")
    meshio.write(outfile + ".xdmf", mesh)

    # read back in and extract boundaries and subdomains
    file = XDMFFile(outfile + ".xdmf")
    mesh = Mesh()
    file.read(mesh)

    subdomains = MeshFunction("size_t", mesh, 3, 0)
    file.read(subdomains, "subdomains")

    write_mesh_per_subdomain(outfile, subdomains, mesh)

    boundaries = MeshFunction("size_t", mesh, 2, 0)
    z_bottom = mesh.coordinates()[:,2].min()


    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)

    # set rigid skull boundary
    rigid_skull = CompiledSubDomain("on_boundary")
    rigid_skull.mark(boundaries, rigid_skull_id)


    # set internal interface
    extract_internal_interface(mesh,subdomains, boundaries, interface_id)

    # add further external boundaries
    if add_spinal_outlet:
        set_external_boundaries(mesh, subdomains, boundaries, fluid_id, spinal_outlet_id, spinal_outlet)
    set_external_boundaries(mesh, subdomains, boundaries, porous_id, fix_stem_id, lambda x: True)
            
    boundaries_outfile = XDMFFile(outfile + "_boundaries.xdmf")

    boundaries_outfile.write(boundaries)
    boundaries_outfile.close()


    # create restrictions and write to file
    fluid_restr = generate_subdomain_restriction(mesh, subdomains, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, subdomains, porous_id)
    fluid_restr._write(outfile + "_fluid.rtc.xdmf")
    porous_restr._write(outfile + "_porous.rtc.xdmf")
    return outfile


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c",
        metavar="config.yml",
        help="path to config file",
        type=str,)
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    target_folder = convert_mesh(config)
    os.popen(f'cp {config_file_path} {target_folder}_config.yml') 


    