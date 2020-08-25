import meshio
from fenics import *
from braininversion.meshes import extract_internal_interface, generate_subdomain_restriction
import argparse
import yaml
import numpy as np
import os

# subdomain ids
fluid_id = 2
porous_id = 1

# boundary ids
interface_id = 1
rigid_skull_id = 2
spinal_outlet_id = 3

def generate3DIdealizedBrainMesh(config):

    import pygmsh

    N = config["N"] #
    brain_radius = config["brain_radius"] 
    sas_radius = config["sas_radius"] 
    ventricle_radius =  config["ventricle_radius"] 
    aqueduct_width = config["aqueduct_width"] 
    canal_width =  config["canal_width"] 
    canal_length = config["canal_length"] 
    
    path = f"../meshes/ideal_brain_3D_N{N}/ideal_brain_3D_N{N}"
    os.popen(f'mkdir -p {path}') 
    h = 1.0/N
    geom = pygmsh.opencascade.Geometry(
            characteristic_length_min=0,
            characteristic_length_max=h,
            )

    brain = geom.add_ball((0,0,0), brain_radius)
    ventricle = geom.add_ball((0,0,0), ventricle_radius)
    aqueduct = geom.add_cylinder((0,0,0), (0,0,-brain_radius), aqueduct_width)
    #aqueduct_smoothing = geom.add_ball((0,0,-brain_radius), aqueduct_width*2)
    brain = geom.boolean_difference([brain], [ventricle, aqueduct,])# aqueduct_smoothing])
    spinal_canal = geom.add_cylinder((0,0,0), (0,0,-canal_length), canal_width)
    fluid = geom.add_ball((0,0,0), sas_radius)
    fluid = geom.boolean_union([fluid, spinal_canal])
    tot = geom.boolean_fragments([fluid], [brain])
    geom.add_physical(fluid, fluid_id)
    geom.add_physical(brain, porous_id)
    mesh = pygmsh.generate_mesh(geom, ) #extra_gmsh_arguments=["-string", "Mesh.Smoothing=20;"])
    meshio.write(path + ".xdmf",
             meshio.Mesh(points=mesh.points, 
             cells={"tetra": mesh.cells_dict["tetra"]},
             cell_data={"subdomains":mesh.cell_data["gmsh:physical"]}))

    generate_boundaries(path, canal_length)
    return path


def generate_boundaries(path, canal_length):
    infile = XDMFFile( path + ".xdmf")
    mesh = Mesh()
    infile.read(mesh)
    gdim = mesh.geometric_dimension()
    subdomains = MeshFunction("size_t", mesh, gdim, 0)
    infile.read(subdomains, "subdomains")
    boundaries = MeshFunction("size_t", mesh, gdim - 1, 0)
            
    # set rigid skull boundary
    rigid_skull = CompiledSubDomain("on_boundary")

    rigid_skull.mark(boundaries, rigid_skull_id)

    # set spinal outlet 
    spinal_outlet = CompiledSubDomain(f"near(x[{gdim - 1}], - canal_length)", canal_length=canal_length)

    spinal_outlet.mark(boundaries, spinal_outlet_id)
    extract_internal_interface(mesh, subdomains, boundaries, interface_id)

    boundaries_outfile = XDMFFile(path + "_boundaries.xdmf")
    boundaries_outfile.write(boundaries)
    boundaries_outfile.close()

    # create restrictions and write to file
    fluid_restr = generate_subdomain_restriction(mesh, subdomains, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, subdomains, porous_id)
    fluid_restr._write(path + "_fluid.rtc.xdmf")
    porous_restr._write(path + "_porous.rtc.xdmf")


def generate2DIdealizedBrainMesh(config):
    from mshr import Circle, Rectangle, generate_mesh

    N = config["N"] #
    brain_radius = config["brain_radius"] # 0.1
    sas_radius = config["sas_radius"] # brain_radius*1.2
    ventricle_radius =  config["ventricle_radius"] # 0.03
    aqueduct_width = config["aqueduct_width"] # 0.005
    canal_width =  config["canal_width"] # 0.02
    canal_length = config["canal_length"] # 0.2
    
    path = f"../meshes/ideal_brain_2D_N{N}/ideal_brain_2D_N{N}"
    os.popen(f'mkdir -p {path}') 

    brain = Circle(Point(0,0), brain_radius)
    ventricle = Circle(Point(0,0), ventricle_radius)
    aqueduct = Rectangle(Point(-aqueduct_width/2, -brain_radius), Point(aqueduct_width/2, 0))
    brain = brain - ventricle - aqueduct

    spinal_canal = Rectangle(Point(-canal_width/2, -canal_length), Point(canal_width/2, 0))
    fluid = Circle(Point(0,0), sas_radius) + spinal_canal
    domain = fluid + brain
    domain.set_subdomain(fluid_id, fluid)
    domain.set_subdomain(porous_id, brain)

    mesh = generate_mesh(domain, N)
    subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
    subdomains.rename("subdomains", "subdomains")

    subdomains_outfile = XDMFFile(path + ".xdmf")
    subdomains_outfile.write(subdomains)
    subdomains_outfile.close()
    generate_boundaries(path, canal_length)

    # create restrictions and write to file
    fluid_restr = generate_subdomain_restriction(mesh, subdomains, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, subdomains, porous_id)
    fluid_restr._write(path + "_fluid.rtc.xdmf")
    porous_restr._write(path + "_porous.rtc.xdmf")
    return path

def radius_from_sphere_volume(v):
    return (v*3/(4*np.pi))**(1/3)

def compute_geometry(config):
    radii = ["brain_radius", "ventricle_radius", "sas_radius"]
    if all(r in config.keys() for r in radii):
        return config
    elif "from_volumes" in config.keys():
        vols = config["from_volumes"]
        config["ventricle_radius"] = radius_from_sphere_volume(vols["ventricle_volume"])
        config["brain_radius"] = radius_from_sphere_volume(vols["ventricle_volume"] + vols["parenchyma_volume"])
        config["sas_radius"] = radius_from_sphere_volume(vols["ventricle_volume"] +
                                                         vols["parenchyma_volume"] +
                                                         vols["sas_volume"])
    else:
        raise KeyError("specify either volumes or radii!")
    return config
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c",
        metavar="config.yml",
        help="path to config file",
        type=str,)
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.UnsafeLoader)
    config = compute_geometry(config)
    if config["dim"] == 3:
        target_folder = generate3DIdealizedBrainMesh(config)
    elif config["dim"] == 2:
        target_folder = generate2DIdealizedBrainMesh(config)
    with open(f"{target_folder}_config.yml", 'w') as conf_outfile:
        yaml.dump(config, conf_outfile, default_flow_style=False)
