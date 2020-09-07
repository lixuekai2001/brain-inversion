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

label_to_subdomain = {1:"porous", 2:"fluid", 3:"fluid", 4:"fluid",
                      5:"fluid", 6:"fluid", 7:"fluid", 8:"fluid"}
domain_ids = {"fluid":fluid_id, "porous":porous_id}

label_refine = [{"label":8, "iterations":1, "target_size": 0.002}]

# read back in and extract boundaries and subdomains
def extract_subdomains_from_physiological_labels(mesh_file, label_to_subdomain):
    outfile_stem = mesh_file[:-12]
    file = XDMFFile(mesh_file)
    mesh = Mesh()
    file.read(mesh)
    labels = MeshFunction("size_t", mesh, 3, 0)
    file.read(labels, "subdomains")
    

    #print(labels.array()[:].sum())
    #mesh = refine(mesh)
    #labels = adapt(labels, mesh)
    #print(labels.array()[:].sum())



    for refine_config in label_refine:
        refine_iter = refine_config["iterations"]
        label = refine_config["label"]
        for i in range(refine_iter):
            refine_marker = MeshFunction("bool", mesh, 3, False)
            diam = CellDiameter(mesh)
            DG0 = FunctionSpace(mesh, "DG", 0)
            diam_func = project(diam, DG0)
            label_crit = labels.array()[:]==label
            size_crit = diam_func.vector()[:] > refine_config["target_size"]
            refine_marker.array()[label_crit & size_crit] = True
            print(f"refining {refine_marker.array()[:].sum()} cells in region {label}...")
            mesh = refine(mesh, refine_marker)
            labels = adapt(labels, mesh)
    print("finished refinement...")

    subdomains = MeshFunction("size_t", mesh, 3, 0)


    for label, domain in label_to_subdomain.items():
        subdomains.array()[labels.array()==label] = domain_ids[domain]


    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)
    boundaries = MeshFunction("size_t", mesh, 2, 0)
    z_bottom = mesh.coordinates()[:,2].min()

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