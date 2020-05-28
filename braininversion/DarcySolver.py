from fenics import *
import numpy as np
from fenics_adjoint import *
from dolfin.cpp.log import info
parameters['ghost_mode'] = 'shared_facet' 



def solve_darcy(mesh, f, T, num_steps, K,
                boundary_marker, boundary_conditions,
                p_initial=Constant(0.0), c=Constant(1.0),
                theta=0.5, degree=1):

    time = 0.0  
    dt = T / num_steps # time step size
    try:
        p_initial.t=0
    except:
        pass
    dirichlet_bcs = []
    time_dep_expr_full = []
    time_dep_expr_theta = []

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)
    V = FunctionSpace(mesh, "CG", degree)

    if type(f)==list:
        ctrls = f
        f = Function(V, name="force")
    else:
        ctrls = None
        time_dep_expr_theta.append(f)
    if K<1e-9:
        sc = Constant(1e-9/K)
        K = Constant(K*sc)
    else:
        sc = Constant(1.0)
        K = Constant(K)

    p = TrialFunction(V)
    v = TestFunction(V)
    p_n = interpolate(p_initial, V)
    dt_ = Constant(dt)
    dtdp = (p - p_n)/dt_
    p_theta = theta*p + (1.0 - theta)*p_n

    F = sc*c*dtdp*v*dx + inner( K*grad(p_theta), grad(v) )*dx - sc*f*v*dx

    
    for marker_id, bc in boundary_conditions.items():
        for bc_type, bc_val in bc.items():
            if bc_type=="Dirichlet":
                bc_d = DirichletBC(V, bc_val, boundary_marker, marker_id)
                dirichlet_bcs.append(bc_d)
                time_dep_expr_full.append(bc_val)
                #print(f"Dirichlet boundary added (ID: {marker_id})")
                break
            if bc_type=="Neumann":
                F += sc*K*v*bc_val*ds(marker_id)
                time_dep_expr_theta.append(bc_val)
                #print(f"Neumann boundary added (ID: {marker_id})")
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F += sc*(-beta*p + r)*K*v*ds(marker_id)
                time_dep_expr_theta.append(r)
                break
            print(f"Warning! bc_type {bc_type} not supported!")

    (a, L) = system(F)
    p = Function(V)
    info(  "Start system assembly")
    A = assemble(a)
    info(  "Finished system assembly")


    for bc in dirichlet_bcs:
        bc.apply(A)

    info(  "Initialize LU solver")

    solver = LUSolver(A, "mumps")
    solver.parameters["symmetric"] = True
    solver.parameters["verbose"] = True

    info( "Finished LU solver")


    for n in range(num_steps):
            
            # Update current time, force and bcs
            update_expression_time(time_dep_expr_full, time + dt)
            update_expression_time(time_dep_expr_theta, time + dt*theta)

            if ctrls:
                f.assign(ctrls[n])
                
            time += dt

            b = assemble(L)
            for bc in dirichlet_bcs:
                bc.apply(b)

            # Compute solution
            info( "Start system solve")
            solver.solve( p.vector(), b)
            #solve(A, p.vector(), b, "cg", "hypre_amg",)
            info( "Finish system solve")

            p_n.assign(p)
            yield p_n

def update_expression_time(list_of_expressions, time):

    for expr in list_of_expressions:
        try:
            expr.t = time
        except:
            pass
        try:
            for op in expr.ufl_operands:
                try:
                    op.t =  time
                except:
                    pass
        except:
            pass


def solve_darcy_on_doughnut_brain(mesh, f, T, num_steps, K,
                                  probe_points = [], slice_points = [],
                                  dirichlet_boundary_skull=None,
                                  dirichlet_boundary_ventricle=None,
                                  p_obs=None,
                                  alpha=0.0,
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

    p = TrialFunction(V)
    K = Constant(K)
    v = TestFunction(V)
    p_n = interpolate(Constant(0), V)
    p_ = interpolate(Constant(0), V)
    dt_ = Constant(dt)
    dtdp = (p - p_n)/dt_
    p_theta = theta*p + (1.0 - theta)*p_n

    #F = p*v*dx + dt_*inner( K*grad(p), grad(v) )*dx - (p_n + dt_*f)*v*dx
    F = dtdp*v*dx + inner( K*grad(p_theta), grad(v) )*dx - f*v*dx
    (a, L) = system(F)
    
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
    
    A = assemble(a)
    for bc in bcs:
        bc.apply(A)
    solver = LUSolver(A)

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
                
        b = assemble(L)
        
        for bc in bcs:
            bc.apply(b)
        
        # Compute solution
        solver.solve(p_.vector(), b)
        # Update previous solution
        p_n.assign(p_)

        # write to file
        if file_name:
            xdmf_file.write(p_, time)
            
        for i, point in enumerate(slice_points):
            p_slice_data[n,i] = p_(point)
            f_slice_data[n,i] = f(point)

        for i, point in enumerate(probe_points):

            p_data[n,i] = p_(point)
            f_data[n,i] = f(point)

           
        if p_obs:
            J += errornorm(p_obs, p_, "L2", mesh=mesh)
            for i, point in enumerate(probe_points):
                p_obs_data[n,i] = p_obs(point)

            for i, point in enumerate(slice_points):
                target_slice_data[n,i] = p_obs(point)
            
    
    results = {"p":p_}
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