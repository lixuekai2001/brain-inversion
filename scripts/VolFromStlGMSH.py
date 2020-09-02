import gmsh
import math
import os
import meshio

gmsh.initialize()

gmsh.option.setNumber("General.Terminal", 1)


components = ["csf", "parenchyma", "lateral_ventricles", "aqueduct","third_ventricle", "fourth_ventricle", "foramina", "median_aperture"]
for c in components:
    gmsh.merge(f"/home/marius/Documents/brain_meshes/slicer_stl_files/{c}.stl")

#gmsh.merge("/home/marius/Downloads/124422_csf.stl")
#gmsh.merge("/home/marius/Downloads/101309_ventricles.stl")


n = gmsh.model.getDimension()
s = gmsh.model.getEntities(n)
#l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))]) # add all tags of .stl surfaces!
#gmsh.model.geo.addVolume([l])

surface_tags = []
for i, comp in enumerate(components):
    surface_tag = gmsh.model.geo.addSurfaceLoop([s[i][1]])
    surface_tags.append(surface_tag)

    #vol_tag = gmsh.model.geo.addVolume([surface_tag])
for i, tag in enumerate(surface_tags):
    vol_tag = gmsh.model.geo.addVolume([tag])
    group_tag = gmsh.model.addPhysicalGroup(dim=3, tags=[vol_tag])


print("Volume added")
gmsh.model.geo.synchronize()


#group_tag = gmsh.model.addPhysicalGroup(dim=3, tags=[vent_vol])
#group_tag = gmsh.model.addPhysicalGroup(dim=3, tags=[csf_vol])

#gmsh.model.setPhysicalName(dim=3, tag=group_tag, name="Whole domain")

gmsh.model.mesh.generate(3)

gmsh.write('t13.msh')
gmsh.finalize()

mesh = meshio.read("t13.msh")
meshio.write("test_mesh.xdmf", mesh)
print("mesh generation finished!")