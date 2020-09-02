import meshio
import sys
from fenics import *
import numpy as np
from braininversion.meshes import (generate_subdomain_restriction, 
                                   extract_internal_interface,set_external_boundaries)


# subdomain ids
fluid_id = 2
porous_id = 1

# boundary ids
interface_id = 1
rigid_skull_id = 2
spinal_outlet_id = 3
fix_stem_id = 4

label_to_subdomain = {1:"fluid", 2:"fluid", 3:"fluid", 4:"fluid", 5:"fluid", 6:"fluid", 7:"fluid", 8:"porous"}
domain_ids = {"fluid":fluid_id, "porous":porous_id}

# read back in and extract boundaries and subdomains
def extract_subdomains_from_physiological_labels(mesh_file, label_to_subdomain):
    outfile_stem = mesh_file[:-12]
    file = XDMFFile(mesh_file)
    mesh = Mesh()
    file.read(mesh)
    subdomains = MeshFunction("size_t", mesh, 3, 0)
    labels = MeshFunction("size_t", mesh, 3, 0)

    file.read(labels, "subdomains")

    for label, domain in label_to_subdomain.items():
        subdomains.array()[labels.array()==label] = domain_ids[domain]

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
    set_external_boundaries(mesh, subdomains, boundaries, porous_id, fix_stem_id, lambda x: True)
    set_external_boundaries(mesh, subdomains, boundaries, fluid_id, spinal_outlet_id, lambda x: np.isclose(x[2], z_bottom, rtol=8e-3) )

    boundaries_outfile = XDMFFile(outfile_stem + "_boundaries.xdmf")
    boundaries_outfile.write(boundaries)
    boundaries_outfile.close()
    subdomains_outfile = XDMFFile(outfile_stem + ".xdmf")
    subdomains_outfile.write(subdomains)
    subdomains_outfile.close()

    fluid_restr = generate_subdomain_restriction(mesh, subdomains, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, subdomains, porous_id)
    fluid_restr._write(outfile_stem + "_fluid.rtc.xdmf")
    porous_restr._write(outfile_stem + "_porous.rtc.xdmf")

if __name__=="__main__":
    mesh_file = sys.argv[1]
    extract_subdomains_from_physiological_labels(mesh_file, label_to_subdomain)