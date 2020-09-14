from fenics import * 
from multiphenics import *
#from mshr import *
import jdata as jd
import numpy as np
import scipy.io
import pyvista as pv
import numpy as np
import vtk

def generate_doughnut_mesh(brain_radius, ventricle_radius, N):
    brain = Circle(Point(0,0), brain_radius)
    ventricle = Circle(Point(0,0), ventricle_radius)
    #aqueduct = Rectangle(Point(-aqueduct_width/2, -brain_radius),
                         #Point(aqueduct_width/2, 0))
    brain = brain - ventricle #- aqueduct
    mesh = Mesh(generate_mesh(brain, N))
    ventricle = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]<r*r*0.9",
                                  r=brain_radius)
    skull = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]>r*r*0.9", 
                                r=brain_radius)
    boundarymarker = MeshFunction("size_t",mesh, 1)
    ventricle.mark(boundarymarker, 1)
    skull.mark(boundarymarker, 2)
    return mesh, boundarymarker

def generate_flower_mesh(brain_radius, n_petals, N):
    r = brain_radius*0.75
    brain = Circle(Point(0,0), r)
    for n in range(n_petals):
        x = Point(r*sin(n*2*pi/n_petals), r*cos(n*2*pi/n_petals))
        brain += Ellipse(x, r/2, r/2)
    for x in [-r/5, r/5]:
        brain -= Ellipse(Point(x, 0), r/8, r/3)
    mesh = Mesh(generate_mesh(brain, N))
    ventricle = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]<r*r*0.9", r=r)
    skull = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]>r*r*0.9", r=r)
    boundarymarker = MeshFunction("size_t",mesh, 1)
    ventricle.mark(boundarymarker, 1)
    skull.mark(boundarymarker, 2)
    return mesh, boundarymarker

def jmesh_to_vtk(infile, outfile, infile_dict, outfile_dict, reduce=False, scaling=1.0):
    if infile.endswith(".jmsh"):
        jmesh = jd.load(infile)
        meshElem = jmesh["MeshElem"]
        points = jmesh["MeshVertex3"]*scaling
    elif infile.endswith(".mat"):
        mat_contents = scipy.io.loadmat(infile)
        points = mat_contents["node"]*scaling
        meshElem = mat_contents["elem"]
    subdomains = meshElem[:,4]
    if reduce:
        subdomains[subdomains==infile_dict["other"]] = 10
        subdomains[subdomains==infile_dict["scalp"]] = 10
        subdomains[subdomains==infile_dict["skull"]] = 10
        subdomains[subdomains==infile_dict["CSF"]] = outfile_dict["fluid_id"]
        subdomains[subdomains==infile_dict["WM"]] = outfile_dict["porous_id"]
        subdomains[subdomains==infile_dict["GM"]] = outfile_dict["porous_id"]
        
    n = meshElem.shape[0]
    p = 4 # number of points per cell for TETRA

    c = np.insert(meshElem[:,:4] - 1, 0, p, axis=1)
    cell_type = np.repeat(vtk.VTK_TETRA, n)
    offset = np.arange(start=0, stop=n*(p+1), step=p+1)
    grid = pv.UnstructuredGrid(offset, c, cell_type, points)
    grid.cell_arrays["subdomains"] = subdomains.flatten()
    if reduce:
        grid = grid.threshold(5, scalars="subdomains", invert=True)
    pv.save_meshio(outfile, grid)


def generate_subdomain_restriction(mesh, subdomains, subdomain_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] == subdomain_id:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

def extract_internal_interface(mesh,subdomains, boundaries, interface_id=None):
    # set internal interface
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(subdomains[c])

        domains = list(set(domains))
        if len(domains) == 2 and interface_id is not None:
            boundaries[f] = interface_id
        elif len(domains) ==2:
            boundaries[f] = int(f"{min(domains)}{max(domains)}")


def set_external_boundaries(mesh, subdomains, boundaries, subdomain_id, boundary_id, criterion=None):
    if criterion is None:
        criterion = lambda x: True
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(subdomains[c])
        domains = list(set(domains))
        if f.exterior() and len(domains) == 1:
            if domains[0] == subdomain_id and criterion(f.midpoint().array()):
                boundaries[f] = boundary_id
