import meshio
from fenics import *
import argparse
import yaml
import numpy as np
import os

interface_id : 1
rigid_skull_id : 2
spinal_outlet_id : 3

def generate3DIdealizedBrainMesh(config):

    import pygmsh
    ids = {dom["name"]:dom["id"] for dom in  config["domains"]}
    N = config["N"] #
    brain_radius = config["brain_radius"] 
    sas_radius = config["sas_radius"] 
    ventricle_radius =  config["ventricle_radius"] 
    aqueduct_width = config["aqueduct_width"] 
    canal_width =  config["canal_width"] 
    canal_length = config["canal_length"] 
    
    path = f"meshes/ideal_brain_3D_N{N}/ideal_brain_3D_N{N}"
    os.popen(f'mkdir -p {path}') 
    h = 1.0/N
    geom = pygmsh.opencascade.Geometry(
            characteristic_length_min=0,
            characteristic_length_max=h,
            )

    brain = geom.add_ball((0,0,0), brain_radius)
    ventricle = geom.add_ball((0,0,0), ventricle_radius)
    aqueduct = geom.add_cylinder((0,0,0), (0,0,-brain_radius), aqueduct_width)
    brain = geom.boolean_difference([brain], [ventricle, aqueduct,], delete_other=False)
    aqueduct = geom.boolean_difference([aqueduct], [ventricle,], delete_other=False)

    spinal_canal = geom.add_cylinder((0,0,0), (0,0,-canal_length), canal_width)
    fluid = geom.add_ball((0,0,0), sas_radius)
    fluid = geom.boolean_union([fluid, spinal_canal])
    geom.boolean_fragments([fluid,brain, aqueduct, ventricle], [fluid,brain, aqueduct, ventricle])
    geom.add_physical(fluid, ids["csf"])
    geom.add_physical(brain, ids["parenchyma"])
    geom.add_physical(aqueduct, ids["aqueduct"])
    geom.add_physical(ventricle, ids["ventricle"])

    mesh = pygmsh.generate_mesh(geom, extra_gmsh_arguments=["-string", "Mesh.Smoothing=50;"])
    meshio.write(path + "_labels.xdmf",
             meshio.Mesh(points=mesh.points, 
             cells={"tetra": mesh.cells_dict["tetra"]},
             cell_data={"subdomains":mesh.cell_data["gmsh:physical"]}))

    #generate_boundaries(path, canal_length)
    return path


def generate2DIdealizedBrainMesh(config):
    from mshr import Circle, Rectangle, generate_mesh
    ids = {dom["name"]:dom["id"] for dom in  config["domains"]}
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
    domain.set_subdomain(ids["csf"], fluid)
    domain.set_subdomain(ids["parenchyma"], brain)
    domain.set_subdomain(ids["aqueduct"], aqueduct)
    domain.set_subdomain(ids["ventricle"], ventricle)

    mesh = generate_mesh(domain, N)
    subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
    subdomains.rename("subdomains", "subdomains")

    subdomains_outfile = XDMFFile(path + "_labels.xdmf")
    subdomains_outfile.write(subdomains)
    subdomains_outfile.close()
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
    gdim = config["dim"]
    sas_r = config["sas_radius"]
    brain_r = config["brain_radius"]
    ventr_r = config["ventricle_radius"]
    config["probes"] = {"ventricular_system":{"ventricle" : [0]*gdim},
                        "sas":{"sas": [0.5*(brain_r + sas_r)] + [0]*(gdim -1)},
                        "parenchyma":{"parenchyma": [0.5*(brain_r + ventr_r)] + [0]*(gdim -1) }}
    return config
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
        metavar="config.yml",
        help="path to config file",
        type=str,)
    parser.add_argument("-N",help="resolution",type=int,)
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    N = conf_arg["N"]

    with open(config_file_path) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.UnsafeLoader)
    config["N"] = N
    config = compute_geometry(config)
    if config["dim"] == 3:
        target_folder = generate3DIdealizedBrainMesh(config)
    elif config["dim"] == 2:
        target_folder = generate2DIdealizedBrainMesh(config)
    with open(f"{target_folder}_config.yml", 'w') as conf_outfile:
        yaml.dump(config, conf_outfile, default_flow_style=None)

