
from fenics import *
#from fenics_adjoint import *
import pyvista as pv
import meshio
import numpy as np
import vtk

def read_mesh_from_h5(filename, scaling=1e-3):
    file = HDF5File(MPI.comm_world, filename, "r")
    mesh = Mesh()
    file.read(mesh, "/mesh", False)
    mesh = Mesh(mesh)
    mesh.scale(scaling)
    boundarymarker = MeshFunction("size_t", mesh, 2)
    file.read(boundarymarker, "/boundaries")
    file.close()
    return mesh, boundarymarker


def write_to_xdmf(functions, filename, times=None):
    
    xf = XDMFFile(MPI.comm_world, filename)
    xf.parameters["functions_share_mesh"] = True
    xf.parameters["rewrite_function_mesh"] = False
    for f in functions:
        for i,t in enumerate(times):
            xf.write(f[i], t)
    xf.close()

def read_xdmf_timeseries(filename, idx=None, variables=None):
    times = []
    point_dataset = []
    cell_data_set = []
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        if idx is None:
            idx = range(reader.num_steps)
        for k in idx:
            t, point_data, cell_data = reader.read_data(k)
            times.append(t)
            data_dict = dict()
            for name, data in point_data.items():
                if variables is not None:
                    if name in variables:
                        data_dict[name] = data
                else:
                        data_dict[name] = data
            point_dataset.append(data_dict)
            data_dict = dict()
            for name, data in cell_data.items():
                if variables is not None:
                    if name in variables:
                        data_dict[name] = data
                else:
                        data_dict[name] = data
            cell_data_set.append(data_dict)
            
            #point_dataset.append(point_data)
            #cell_data_set.append(cell_data)
    return times, points, cells, point_dataset, cell_data_set


def resample_uniform_grid(infile, dimensions=(100,100,100), radius=2):
    data = pv.read(infile)
    uniform = pv.create_grid(data,dimensions).interpolate(data, radius)
    return uniform

def xdmf_to_unstructuredGrid(filename, idx=None, variables=None):
    times, points, cells, point_dataset, cell_dataset = read_xdmf_timeseries(filename, idx=idx,
                                                                             variables=variables)
    n = cells[0].data.shape[0] # number of cells
    p = cells[0].data.shape[1] # number of points per cell
    c = cells[0].data
    
    cell_type = vtk.VTK_TETRA

    c = np.insert(c, 0, p, axis=1) # insert number of points per cell at begin of cell row
    #print(f"{n} cells with {p} points per cell detected")
    offset = np.arange(start=0, stop=n*(p+1), step=p+1)
    grid = pv.UnstructuredGrid(offset, c, np.repeat(cell_type, n), points)
    for i, p_data in enumerate(point_dataset):   
        if len(point_dataset)==1:
            array_name = "{name}"  
        else:
            array_name = "{name}_{idx}"  
        for name, data in p_data.items():
            grid.point_arrays[array_name.format(name=name, idx=i)] = data
    for i, c_data in enumerate(cell_dataset):
        for name, data in c_data.items():
            grid.cell_arrays[array_name.format(name=name, idx=i)] = data
    return grid