from fenics import *
import numpy as np
from fenics_adjoint import *

def solve_darcy_on_doughnut_brain(mesh, f, T, num_steps,
                                  probe_points = [], slice_points = [],
                                  dirichlet_boundary_skull=None,
                                  dirichlet_boundary_ventricle=None,
                                  p_obs=None,
                                  K = 1e-3,
                                  alpha=1e-12,
                                  theta=0.5,
                                  file_name=None):
    
    time = 0.0  

    dt = T / num_steps # time step size
    dx = Measure("dx", domain=mesh)

    V = FunctionSpace(mesh, "CG", 1)
    
    if type(f)==list:
        ctrls = f
        f = Function(V, name="force")
    else:
        ctrls = None

    p = Function(V, name="pressure")
    K = Constant(K)
    v = TestFunction(V)
    p_n = interpolate(Constant(0), V)
    dt_ = Constant(dt)
    dtdp = (p - p_n)/dt_
    p_theta = theta*p + (1.0 - theta)*p_n

    #F = p*v*dx + dt_*inner( K*grad(p), grad(v) )*dx - (p_n + dt_*f)*v*dx
    F = dtdp*v*dx + inner( K*grad(p_theta), grad(v) )*dx - f*v*dx

    R = mesh.coordinates().max()
    ventricle = CompiledSubDomain("on_boundary && (x[0]*x[0] + x[1]*x[1] < R*R*0.95)", R = R)
    skull = CompiledSubDomain("on_boundary && (x[0]*x[0] + x[1]*x[1] >= R*R*0.95 )", R = R)
    
    bcs = []
    if dirichlet_boundary_skull:
        bc_skull = DirichletBC(V, dirichlet_boundary_skull, skull, check_midpoint=False)
        bcs.append(bc_skull)
        
    if dirichlet_boundary_ventricle:
        bc_ventricle = DirichletBC(V, dirichlet_boundary_ventricle, ventricle, check_midpoint=False)
        bcs.append(bc_ventricle)
        
    if file_name:
        xdmf_file = XDMFFile(file_name)
        xdmf_file.parameters["flush_output"] = True
        xdmf_file.parameters["functions_share_mesh"] = True
        xdmf_file.parameters["rewrite_function_mesh"] = False
    
    p_data = np.ndarray((num_steps, len(probe_points)))
    p_obs_data = np.ndarray((num_steps, len(probe_points)))
    f_data = np.ndarray((num_steps, len(probe_points)))

    p_slice_data = np.ndarray((num_steps, len(slice_points)))
    f_slice_data = np.ndarray((num_steps, len(slice_points)))
    target_slice_data = np.ndarray((num_steps, len(slice_points)))

    J = 0
    for n in range(num_steps):
        if ctrls:
            f.assign(ctrls[n])

        # Update current time
        time += dt
        try:
            dirichlet_boundary_skull.t = time
        except:
            pass
                
        try: 
            dirichlet_boundary_ventricle.t = time
        except:
            pass
        try:
            f.t = time
        except:
            pass
            
                
        
        # Compute solution
        solve(F==0, p, bcs)

        # Update previous solution
        p_n.assign(p)

        # write to file
        if file_name:
            xdmf_file.write(p, time)
            
        for i, point in enumerate(slice_points):
            p_slice_data[n,i] = p(point)
            f_slice_data[n,i] = f(point)

        for i, point in enumerate(probe_points):

            p_data[n,i] = p(point)
            f_data[n,i] = f(point)

           
        if p_obs:
            J += assemble( inner(p - p_obs, p - p_obs)*dx + alpha/2 * f**2 * dx )
            for i, point in enumerate(probe_points):
                p_obs_data[n,i] = p_obs(point)

            for i, point in enumerate(slice_points):
                target_slice_data[n,i] = p_obs(point)
            
    
    results = {"p":p}
    results["slice"] = p_slice_data
    results["f_slice"] = f_slice_data
    results["target_slice"] = target_slice_data
    if probe_points:
        results["probe_point_data"] = p_data
        results["probe_point_f_data"] = f_data
    if p_obs:
        results["J"] = J
    if probe_points and p_obs:
        results["probe_point_obs_data"] = p_obs_data
    return results