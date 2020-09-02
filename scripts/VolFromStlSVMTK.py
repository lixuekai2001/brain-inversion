import SVMTK as svm
import meshio
import numpy as np
import os


mesh_resolution = 15.0
scaling = 1e-3

name = "slicerMRIExampleSegmentation"
path = f"../brainMeshBaseFiles/{name}/"
out_path = f"../meshes/{name}_N{mesh_resolution}/"
file_path = path + "{0}.stl"

try:
    os.mkdir(out_path)
except FileExistsError:
    pass
csf_file = file_path.format("csf")
parenchyma_file = file_path.format("parenchyma")

components = ["lateral_ventricles","median_aperture", "aqueduct", "third_ventricle", "fourth_ventricle", "foramina",]
components = ["median_aperture","third_ventricle", "aqueduct", "foramina", "lateral_ventricles", "fourth_ventricle"]
tmp_file = out_path + 'brain.mesh'

# --- Options ---
# Load input file
print("loading csf and parenchyma...")
csf  = svm.Surface(csf_file)
parenchyma = svm.Surface(parenchyma_file)
csf.difference(parenchyma)


clip_plane = [0,0,1]
z_val = -110.0
parenchyma.clip(0,0,-1, -110, True)
csf.clip(0,0,-1, -110, True)

surfaces = [parenchyma, csf]
for comp in components:
    print(f"loading {comp}...")
    c = svm.Surface(file_path.format(comp))
    target_edge_length = 0.3
    nb_iter = 3
    protect_border = True

    #c.isotropic_remeshing(target_edge_length,nb_iter,protect_border)
    #c.fill_holes()
    c.smooth_laplacian(0.5, 5)
    #c.adjust_boundary(0.1)
    surfaces.append(c)
    parenchyma.difference(c)
    csf.difference(c)



smap = svm.SubdomainMap()
num_regions = len(surfaces)

for i in range(num_regions):
    smap.add(f'{10**i:{"0"}{num_regions}}', i + 1)

smap.print()

# Create the volume mesh
maker = svm.Domain(surfaces, smap)
#maker.create_mesh(mesh_resolution)


cell_size = 10.0
edge_size = cell_size
facet_size = cell_size/5
facet_angle = 30.0
facet_distance = cell_size/10.0
cell_radius_edge_ratio = 3.0

maker.create_mesh(edge_size, cell_size, facet_size, facet_angle,
                  facet_distance, cell_radius_edge_ratio)

# Write output file 
maker.save(tmp_file)
mesh = meshio.read(tmp_file)

new_mesh = meshio.Mesh(mesh.points*scaling, cells=[("tetra", mesh.get_cells_type("tetra"))],
                       cell_data={"subdomains":[mesh.cell_data_dict["medit:ref"]["tetra"]]})
print(new_mesh)
new_mesh.write(out_path + f"{name}_N{mesh_resolution}_labels.xdmf")

#boundary_mesh = meshio.Mesh(mesh.points, cells=[("triangle", mesh.get_cells_type("triangle"))],
#                       cell_data={"boundaries":[ mesh.cell_data_dict["medit:ref"]["triangle"]]})

#boundary_mesh.write(out_path + f"{name}_boundaries.xdmf")

#for i in range(num_regions + 1):
#    boundary = maker.get_boundary(i)
#    boundary.save(f"subdomain_boundary_{i}.off")

