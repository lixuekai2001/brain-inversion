from fenics import *
import numpy as np
from fenics_adjoint import *


def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def solve_biot(mesh, f, g, T, num_steps, material_parameter,
              boundary_marker_p, boundary_conditions_p,
              boundary_marker_u, boundary_conditions_u,
              p_initial=Constant(0.0),u_initial=Constant((0.0, 0.0)),
              theta=0.5, u_degree=2, p_degree=1):
              
    try:
        p_initial.t=0
    except:
        pass

    gdim = mesh.geometric_dimension()
    c = material_parameter["c"]
    K = material_parameter["K"]
    lmbda = material_parameter["lmbda"]
    mu = material_parameter["mu"]
    alpha = material_parameter["alpha"]


    time = 0.0
    dt = T / num_steps

    time_dep_expr_full = []
    time_dep_expr_theta = []

    V = VectorElement("CG", mesh.ufl_cell(), u_degree)
    W = FiniteElement("CG", mesh.ufl_cell(), p_degree)
    M = MixedElement([V, W, W])
    VW = FunctionSpace(mesh, M)
    up = TrialFunction(VW)
    
    u, p_T, p = split(up)

    up_n = Function(VW)
    u_, p_T_, p_ = split(up_n)

    #u_.assign(interpolate(u_initial, VW.sub(0).collapse()))
    #p_.assign(interpolate(p_initial, VW.sub(2).collapse()))
    #p_T_.assign(project(lmbda*div(u_) - p_, VW.sub(1).collapse())  )

    vq = TestFunctions(VW)
    v, q_T, q = vq[0], vq[1], vq[2]

    dtdp =  (p - p_)/dt
    dtdp_T = (p_T - p_T_)/dt
    
    p_theta = theta*p + (1.0 - theta)*p_
    p_T_theta = theta*p_T + (1.0 - theta)*p_T_
    u_theta = theta*u + (1.0 - theta)*u_
    
    if type(g)==list:
        ctrls_g = g
        g = Function(VW.sub(2).collapse())
    else:
        ctrls_g = None
        time_dep_expr_theta.append(g)

    if type(f)==list:
        ctrls_f = f
        f = Function(VW.sub(0).collapse())
    else:
        ctrls_f = None
        time_dep_expr_theta.append(f)

    F1 = 2*mu*inner(eps(u_theta), eps(v))*dx + p_T_theta*div(v)*dx - inner(f,v)*dx

    F2 = (c* dtdp + alpha/lmbda *(dtdp_T + dtdp) )*q *dx + \
        K *inner(grad(p_theta), grad(q))*dx - g*q*dx  

    F3 = (lmbda*div(u_theta) - p_theta - p_T_theta) *q_T *dx

    dirichlet_bcs = []

    ds_p = Measure("ds", domain=mesh, subdomain_data=boundary_marker_p)
    ds_u = Measure("ds", domain=mesh, subdomain_data=boundary_marker_u)

    # extract pressure boundary conditions
    for marker_id, bc in boundary_conditions_p.items():
        for bc_type, bc_val in bc.items():
            if bc_type=="Dirichlet":
                bc_d = DirichletBC(VW.sub(2), bc_val,
                                   boundary_marker_p, marker_id)
                dirichlet_bcs.append(bc_d)
                time_dep_expr_full.append(bc_val)
                break
            if bc_type=="Neumann":
                F2 += K*q*bc_val*ds_p(marker_id)
                time_dep_expr_theta.append(bc_val)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F2 += (-beta*p + r)*K*q*ds_p(marker_id)
                time_dep_expr_theta.append(r)
                break
            print(f"Warning! bc_type {bc_type} not supported!")

    # extract displacement boundary conditions
    for marker_id, bc in boundary_conditions_u.items():
        for bc_type, bc_val in bc.items():
            if bc_type=="Dirichlet":
                bc_d = DirichletBC(VW.sub(0), bc_val,
                                   boundary_marker_u, marker_id)
                dirichlet_bcs.append(bc_d)
                time_dep_expr_full.append(bc_val)
                break
            if bc_type=="Neumann":
                F1 += K*inner(v, bc_val)*ds_u(marker_id)
                time_dep_expr_theta.append(bc_val)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F1 += K*inner(-beta*p + r,v)*ds_u(marker_id)
                time_dep_expr_theta.append(r)
                break
            print(f"Warning! bc_type {bc_type} not supported!")
    
    F = F1 + F2 + F3
    
    (a, L) = system(F)
    A = assemble(a)

    for bc in dirichlet_bcs:
        bc.apply(A)

    solver = LUSolver(A, "mumps")
    solver.parameters["symmetric"] = True

    up = Function(VW)


    for n in range(num_steps):

        # Update current time
        if ctrls_g:
            g.assign(ctrls_g[n])
        if ctrls_f:
            f.assign(ctrls_f[n])

        update_expression_time(time_dep_expr_theta, time+ dt*theta)
        update_expression_time(time_dep_expr_full, time + dt)

        time += dt
        b = assemble(L)
        for bc in dirichlet_bcs:
            bc.apply(b)
        # Compute solution
        solver.solve( up.vector(), b)
        
        up_n.assign(up)
        yield up_n
        
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