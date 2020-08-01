from block import *
from block.iterative import *
from block.algebraic.petsc import *
from block.dolfin_util import *
from fenics import *
import numpy as np 

def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def solve_static_biot(mesh, f, g, material_parameter,
              boundary_marker_p, boundary_conditions_p,
              boundary_marker_u, boundary_conditions_u,
              u_degree=2, p_degree=1, u_nullspace=False,
              solver_params=None, solve_iterative=True):
              

    dirichlet_bcs = []
    dim = mesh.topology().dim()
    K = Constant(material_parameter["K"])
    lmbda = Constant(material_parameter["lmbda"])
    alpha = Constant(material_parameter["alpha"])

    V = VectorElement("CG", mesh.ufl_cell(), u_degree)
    W = FiniteElement("CG", mesh.ufl_cell(), p_degree)

    M = MixedElement([V, W, W])
    VW = FunctionSpace(mesh, M)
    V = FunctionSpace(mesh, V)
    W = FunctionSpace(mesh, W)

    up = TrialFunction(VW)
    u, pT, pF = split(up)
    vq = TestFunctions(VW)
    v, qT, qF = vq[0], vq[1], vq[2]

    F1 = inner(eps(u), eps(v))*dx - div(v)*pT*dx - inner(f,v)*dx

    F2 = - div(u)*qT*dx - (1.0/lmbda)*pT*qT*dx + (alpha/lmbda)*pF*qT*dx

    F3 = (alpha/lmbda)*pT*qF*dx - (2.0*(alpha**2)/lmbda) *pF *qF*dx \
         - inner(K*grad(pF), grad(qF))*dx - inner(g, qF)*dx

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
                break
            if bc_type=="Neumann":
                F3 += K*qF*bc_val*ds_p(marker_id)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F3 += (-beta*pF + r)*K*qF*ds_p(marker_id)
                break
            print(f"Warning! bc_type {bc_type} not supported!")


    # extract displacement boundary conditions
    for marker_id, bc in boundary_conditions_u.items():
        for bc_type, bc_val in bc.items():
            if bc_type=="Dirichlet":
                bc_d = DirichletBC(VW.sub(0), bc_val,
                                   boundary_marker_u, marker_id)
                dirichlet_bcs.append(bc_d)
                break
            if bc_type=="Neumann":
                F1 += inner(v, bc_val)*ds_u(marker_id)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F1 += inner(-beta*p + r,v)*ds_u(marker_id)
                break
            print(f"Warning! bc_type {bc_type} not supported!")

    F = F1 + F2 + F3
    (a, L) = system(F)

    if solve_iterative:
        A, b = block_assemble(a, L, bcs=dirichlet_bcs)
        # define preconditioner
        a11 = inner(grad(u), grad(v))*dx
        a22 = pT*qT*dx
        a33 = (alpha**2)/lmbda*pF*qF*dx + inner(K*grad(pF), grad(qF))*dx

        PA = a11 + a22 + a33
        B, _ = block_assemble(PA, L, bcs=dirichlet_bcs)

        null_space = False
        if null_space:
            ns = rigid_body_modes(V)
        else:
            ns= None
        B11 = AMG(B[0,0], pdes=dim, nullspace=ns)
        B22 = Jacobi(B[1,1])
        B33 = AMG(B[2,2], pdes=1)

        B = block_mat([[B11,0, 0],
                       [0,  B22, 0],
                       [0,  0,  B33]])


        default_solver_params = { "maxiter":100, "relativeconv":False, "tolerance":1e-6 }
        if solver_params:
            default_solver_params.update(solver_params)
        AAinv = MinRes(A, precond=B, show=2, name='AA^',**default_solver_params)
        u, pT, pF = AAinv * b

        u, T, pF  = map(Function,[V,W,W], [u, pT, pF])
    else:
        up = Function(VW)
        solve(a==L,up, bcs=dirichlet_bcs, solver_parameters={"linear_solver":"mumps"})
        u, pT, pF = up.split()
    return u, pT, pF