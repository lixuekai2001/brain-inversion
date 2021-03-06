from dolfin import *
from multiphenics import *
import os
import ufl
from ufl import Max
import numpy as np
from pathlib import Path
from multiphenics.la import BlockDefaultFactory

#from tqdm import tqdm

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ftree-vectorize"

snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}



def eps(u):
    return sym(grad(u))
    #return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def solve_biot_navier_stokes(mesh, T, num_steps,
                             material_parameter, 
                             boundary_marker,
                             subdomain_marker,
                             boundary_conditions,
                             porous_restriction_file,
                             fluid_restriction_file,
                             time_dep_expr=(),
                             elem_type="mini",
                             sliprate=0.0,
                             initial_pF=Constant(0.0),
                             initial_pP=Constant(0.0),
                             initial_phi=Constant(0.0),
                             initial_d=None,
                             initial_u=None,
                             g_source=Constant(0.0),
                             f_fluid=None,
                             f_porous=None,
                             outlet_pressure=Constant(0.0),
                             move_mesh=False,
                             filename=None,
                             linearize=True,
                             interf_id=1,
                             u_degree=2, p_degree=1):
    

    gdim = mesh.geometric_dimension()
    fluid_id = 2
    porous_id = 1
    spinal_outlet_id = 3

    for expr in time_dep_expr:
            expr.t = 0
            expr.i = 0

    # porous parameter
    c = material_parameter["c"]
    kappa = material_parameter["kappa"]
    lmbda = material_parameter["lmbda"]
    mu_s = material_parameter["mu_s"]
    rho_s = material_parameter["rho_s"]
    alpha = material_parameter["alpha"]

    # fluid parameter
    rho_f = material_parameter["rho_f"]
    mu_f = material_parameter["mu_f"]

    time = 0.0
    dt = Constant(T / num_steps)

    gamma = Constant(sliprate)
    gravity = Constant([0.0]*gdim)
    f = Constant([0.0]*gdim)

    n = FacetNormal(mesh)("+")
    #n = as_vector([1,0])

    # compute tangential vectors by hausdorfer formula
    #h1 = Max(n[0] - 1, n[0] +1)
    #h2 = n[1]
    #h3 = n[2]
    #h = as_vector([h1,h2,h3])
    #ha = sqrt(inner(h,h))
    #tau1 = as_vector([-2*h1*h2/ha**2, 1 - 2*h2**2/ha**2, -2*h2*h3/ha**2])
    #tau2 = as_vector([-2*h1*h3/ha**2, - 2*h2*h3/ha**2, 1-2*h3**2/ha**2])


    dxF = Measure("dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=fluid_id)
    dxP = Measure("dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=porous_id)
    dxD = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_marker)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)

    ds_Sig = dS(interf_id)
    
    fluidrestriction = MeshRestriction(mesh, fluid_restriction_file)
    porousrestriction = MeshRestriction(mesh, porous_restriction_file)
    if elem_type=="TH":
        V = VectorFunctionSpace(mesh, "CG", u_degree)
    elif elem_type=="mini":
        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        B = FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)
        V = VectorElement(NodalEnrichedElement(P1, B))
        V = FunctionSpace(mesh, V)

    W = FunctionSpace(mesh, "CG", p_degree)

    H = BlockFunctionSpace([V, W, V, W, W],
                            restrict=[fluidrestriction, fluidrestriction,
                                      porousrestriction, porousrestriction,
                                      porousrestriction])
    if linearize:
        trial = BlockTrialFunction(H)
    else:
        trial = BlockFunction(H)
        dtrial = BlockTrialFunction(H)
    u, pF, d, pP, phi = block_split(trial)
    test = BlockTestFunction(H)
    v, qF, w, qP, psi = block_split(test)
    previous = BlockFunction(H)
    u_n, pF_n, d_n, pP_n, phi_n = block_split(previous)
    
    if initial_d is None:
        initial_d = Constant([0.0]*gdim)
    
    if initial_u is None:
        initial_u = Constant([0.0]*gdim)

    if f_fluid is None:
        f_fluid = Constant([0.0]*gdim)
    
    if f_porous is None:
        f_porous = Constant([0.0]*gdim)

    u_n.assign(interpolate(initial_u, H.sub(0)))
    pF_n.assign(interpolate(initial_pF, H.sub(1)))
    d_n.assign(interpolate(initial_d, H.sub(2)))
    pP_n.assign(interpolate(initial_pP, H.sub(3)))
    phi_n.assign(interpolate(initial_phi, H.sub(4)))

    previous.apply("from subfunctions")
    outlet_pressure_func = Function(W)

    # extract Dirichlet boundary conditions
    bcs = []
    for bc in boundary_conditions:
        for marker_id, subsp_bc_val in bc.items():
            for subspace_id, bc_val in subsp_bc_val.items():
                bc_d = DirichletBC(H.sub(subspace_id), bc_val,
                                    boundary_marker, marker_id)
                bcs.append(bc_d)

    bcs = BlockDirichletBC(bcs)

    def proj_t(vec):
        return vec-dot(vec,n)*n


    def tang_interf(u,v):
        #return (dot(u("+"), tau1) + dot(u("+"), tau2))*(dot(v("+"), tau1) + dot(v("+"), tau2))*ds_Sig 
        #return dot(u("+"), tau1)*dot(v("+"), tau1)*ds_Sig + dot(u("+"), tau2)*dot(v("+"), tau2)*ds_Sig + inner(u,v)*Constant(0.0)*dxD
        #tau = as_vector([0,1])
        #return dot(u("+"), tau)* dot(v("+"),tau)*ds_Sig #+ inner(u,v)*Constant(0.0)*dxD
        #return inner(proj_t(u("+")), proj_t(v("+")))*ds_Sig + inner(u,v)*Constant(0.0)*dxD
        return dot(cross(u("+"),n), cross(v("+"),n))*ds_Sig + inner(u,v)*Constant(0.0)*dxD
    # define forms

    if linearize=="steadyStokes":

        def a_F(u,v):
            return 2*mu_f*inner(eps(u), eps(v))*dxF \
                    + (gamma*mu_f/sqrt(kappa))*tang_interf(u,v)

    else:
        def a_F(u,v):
            return rho_f*dot(u/dt, v)*dxF \
                    + 2*mu_f*inner(eps(u), eps(v))*dxF \
                    + (gamma*mu_f/sqrt(kappa))*tang_interf(u,v)
    

    if linearize:
        def c_F(u,v):
            return 0*dxF

    elif linearize=="Oseen":
        def c_F(u,v):
            return rho_f * inner(dot(u_n, grad(u)), v) *dxF

    def b_1_F(v, qF):
        return  -qF*div(v)*dxF

    def b_2_Sig(v, qP):
        return qP("+")*inner(v("+"), n)*ds_Sig + div(v)*qP*Constant(0.0)*dxD

    def b_3_Sig(v, d):
        return - ((gamma*mu_f/sqrt(kappa))*tang_interf(v, d))

    def b_4_Sig(w,qP):
        return -qP("+") * dot(w("+"),n)*ds_Sig + div(w)*qP*Constant(0.0)*dxD

    def a_1_P(d, w):
        return 2.0*mu_s*inner(eps(d), eps(w))*dxP \
                + (gamma*mu_f/sqrt(kappa))*tang_interf(d/dt,w)


    def b_1_P(w, psi):
        return - psi*div(w)*dxP

    def a_2_P(pP,qP):
        return (kappa/mu_f) *inner(grad(pP), grad(qP))*dxP \
                + (c + alpha**2/lmbda)*(pP/dt)*qP*dxP

    def b_2_P(psi, qP):
        return (alpha/lmbda)*psi*qP*dxP

    def a_3_P(phi, psi):
        return (1.0/lmbda)*phi*psi*dxP

    def F_F(v):
        return rho_f*dot(f_fluid, v)*dxF + dot(outlet_pressure_func*FacetNormal(mesh), v)*ds(spinal_outlet_id)

    def F_P(w):
        return rho_s*inner(f_porous, w)*dxP

    def G(qP):
        return rho_f*inner(gravity, grad(qP))*dxP \
                - rho_f*inner(gravity, n)*qP("+")*ds_Sig  + qP*Constant(0.0)*dxD \
                + g_source*qP*dxP

    if linearize=="steadyStokes":
        def F_F_n(v):
            return F_F(v) + b_3_Sig(v, d_n/dt)

    else:
        def F_F_n(v):
            return F_F(v) + rho_f*inner(u_n/dt, v)*dxF + b_3_Sig(v, d_n/dt)

    def F_P_n(w):
        return F_P(w) + (gamma*mu_f/sqrt(kappa))*tang_interf(d_n/dt, w)

    def G_n(qP):
        return G(qP) + (c + (alpha**2)/lmbda)*pP_n/dt*qP*dxP \
            - b_4_Sig(d_n/dt, qP) - b_2_P(phi_n/dt, qP)


    # define system:
    # order trial: u, pF, d, pP, phi
    # order test: v, qF, w, qP, psi

    lhs = [[a_F(u,v) + c_F(u,v), b_1_F(v, pF), b_3_Sig(v, d/dt) , b_2_Sig(v, pP), 0                 ],
        [ b_1_F(u, qF)       ,  0          , 0                , 0             , 0                 ],
        [ b_3_Sig(u, w)      ,  0          , a_1_P(d,w)       , b_4_Sig(w, pP), b_1_P(w, phi)     ],
        [ b_2_Sig(u, qP)    ,  0          , b_4_Sig(d/dt, qP), -a_2_P(pP, qP) , b_2_P(phi/dt, qP)], # *-1
        [ 0                  ,  0          , b_1_P(d, psi)    , b_2_P(psi, pP), -a_3_P(phi, psi) ]]


    b = [[2*mu_f*inner(eps(u),eps(v))*dxF+ gamma*mu_f/sqrt(kappa)*tang_interf(u,v), 0, 0, 0, 0],
        [0, 1/mu_f*pF*qF*dxF, 0, 0, 0],
        [0, 0, 2*mu_s*inner(eps(d), eps(w))*dxP + gamma*mu_f/sqrt(kappa)*tang_interf(d,w), 0, 0],
        [0, 0, 0, (c + alpha**2/lmbda)*pP*qP*dxP+kappa/mu_f*inner(grad(pP),grad(qP))*dxP, 0],
        [0, 0, 0, 0, 1/mu_s*phi*psi*dxP]]


    if linearize==True or linearize=="steadyStokes":
        AA = block_assemble(lhs, keep_diagonal=True)
        bcs.apply(AA)
        iterative_solver = False
        if iterative_solver:
            B = block_assemble(b, keep_diagonal=True)
            bcs.apply(B)
            print("starting iterative solver...")
            solver = PETScKrylovSolver("minres", "amg")
            PETScOptions.set('ksp_monitor_true_residual')   
            #PETScOptions.set("ksp_monitor_singular_value")
            #PETScOptions.set('ksp_type', "minres")   
            #PETScOptions.set('pc_type', "lu")   

            solver.set_from_options()

            solver.set_operators(AA, B)

            #solver.parameters['monitor_convergence'] = True
            solver.parameters['report'] = True
            #solver.parameters['absolute_tolerance'] = 0.0
            #solver.parameters['relative_tolerance'] = 1e-6
            solver.parameters['maximum_iterations'] = 5000
            solver.parameters['nonzero_initial_guess'] = True

        else:
            solver = PETScLUSolver(AA, "mumps")

        
    if filename:
        output_checkp = XDMFFile(filename)
        output_legacy = XDMFFile(f"{Path(filename).parent}/{Path(filename).stem}_old.xdmf" )

        output_checkp.parameters["functions_share_mesh"] = True
        output_checkp.parameters["rewrite_function_mesh"] = False
        output_checkp.parameters["flush_output"] = True
        output_legacy.parameters["functions_share_mesh"] = True
        output_legacy.parameters["rewrite_function_mesh"] = False
        output_legacy.parameters["flush_output"] = True

    names = ["velocity u", "fluid pressure pF", "displacement d",
            "fluid pressure in porous domain pP", "total pressure phi"]
    short_names = ["u", "pF", "d", "pP", "phi"]
    names = short_names

    def solve_linearized():
        rhs = [F_F_n(v), 0, F_P_n(w), -G_n(qP), 0]
        FF = block_assemble(rhs)
        bcs.apply(FF)
        if iterative_solver:
            num_its = solver.solve(previous.block_vector(), FF)
            print(f"number of iterations: {num_its}")
        else:
            solver.solve(previous.block_vector(), FF)
        previous.block_vector().block_function().apply("to subfunctions")
        #solver.solve(sol.block_vector(), FF)
        #sol.block_vector().block_function().apply("to subfunctions")
        #block_assign(previous, sol)

    def solve_oseen():
        AA = block_assemble(lhs, keep_diagonal=True)
        bcs.apply(AA)
        solver = PETScLUSolver(AA, "mumps")
        solver.parameters["symmetric"] = True
        rhs = [F_F_n(v), 0, F_P_n(w), G_n(qP), 0]
        FF = block_assemble(rhs)
        bcs.apply(FF)
        solver.solve(previous.block_vector(), FF)
        previous.block_vector().block_function().apply("to subfunctions")


    class BiotNSBlockNonLinearProblem(BlockNonlinearProblem):
        def __init__(self, block_solution):
            NonlinearProblem.__init__(self)
            rhs = [F_F_n(v), 0, F_P_n(w), G_n(qP), 0]
            F = np.array(lhs).sum(axis=1) - np.array(rhs)
            J = block_derivative(F, trial, dtrial)
            self.residual_block_form = F
            self.jacobian_block_form = J
            self.block_solution = block_solution
            self.bcs = bcs
            # Create block backend for wrapping
            self.block_backend = BlockDefaultFactory()
            self.block_dof_map = self.block_solution.block_function_space().block_dofmap()
            # precompute linear part of jacobian
            # self.precomp_jacobian = block_assemble(J, keep_diagonal=self.bcs is not None)

        def J(self, fenics_jacobian, _):
            # No need to update block solution subfunctions, this has already been done in the residual
            # Wrap FEniCS jacobian into a block jacobian
            block_jacobian = self.block_backend.wrap_matrix(fenics_jacobian)
            # Assemble the block jacobian
            block_assemble(self.jacobian_block_form, block_tensor=block_jacobian, keep_diagonal=self.bcs is not None)
            # Apply boundary conditions
            if self.bcs is not None:
                self.bcs.apply(block_jacobian)


    def solve_nonlinear():
        #rhs = [F_F_n(v), 0, F_P_n(w), G_n(qP), 0]
        #F = np.array(lhs).sum(axis=1) - np.array(rhs)
        #J = block_derivative(F, trial, dtrial)
        #problem = BlockNonLinearProblem(F, trial, bcs, J)
        problem = BiotNSBlockNonLinearProblem(trial)
        solver = BlockPETScSNESSolver(problem)
        solver.parameters.update(snes_solver_parameters["snes_solver"])
        solver.solve()
        block_assign(previous, trial)

    results = block_split(previous)
    write_to_file(results, time, names, output_checkp, output_legacy, elem_type)
    tot_outflow = 0.0
    outflow = 0.0
    outlet_pressure_func.assign(interpolate(outlet_pressure, W))

    for i in range(num_steps):
        time = (i + 1)*dt.values()[0]
        for expr in time_dep_expr:
            expr.t = time
            expr.i = i + 1
            expr.outflow_vol = tot_outflow
            expr.outflow = outflow
            expr.p = results[1]
        outlet_pressure_func.assign(interpolate(outlet_pressure, W))
                
        if linearize==True or linearize=="steadyStokes":
            solve_linearized()
        elif linearize=="oseen":
            solve_oseen()
        else:
            solve_nonlinear()

        results = block_split(previous)

        if move_mesh:
            d =  results[2]
            #d_hat = harmonic_extension(mesh, d, boundary_marker,
            #                           fluidrestriction, dxF, interf_id)
            
            ALE.move(mesh, d)
            #ALE.move(mesh, d_hat)
        
        write_to_file(results, time, names, output_checkp,
                      output_legacy, elem_type)
        
        u = results[0]
        outflow = assemble(inner(u,FacetNormal(mesh))*ds(spinal_outlet_id))
        tot_outflow += outflow*dt
    output_checkp.close()
    output_legacy.close()
    return results


def write_to_file(results, time, names, output_checkp, output_legacy, elem_type):
    for k,r in enumerate(results):
        if np.isclose(time, 0.0) and k==0:
            append = False
        else:
            append = True
        r.rename(names[k], names[k])

        output_legacy.write(r, time)
            
        if isinstance(r.ufl_element(), VectorElement) and elem_type=="mini":
            
            V = VectorFunctionSpace(mesh, "CG", 1)
            output_checkp.write_checkpoint(interpolate(r, V), r.name(), time, append=append)
        else:
            output_checkp.write_checkpoint(r, r.name(), time, append=append)


def harmonic_extension(mesh, d, boundary_marker, fluidrestriction, dxF, interface_id):
    V = VectorFunctionSpace(mesh, "CG", 2)
    H_move = BlockFunctionSpace([V], restrict=[fluidrestriction])
    d_hat = block_split(BlockTrialFunction(H_move))[0]
    v_hat = block_split(BlockTestFunction(H_move))[0]
    D = Constant(1e-2)
    rigid_skull_id = 2

    a = inner(D*nabla_grad(d_hat), nabla_grad(v_hat))*dxF

    bc_move_outer = DirichletBC(H_move.sub(0), Constant([0.0,0.0]),
                            boundary_marker, rigid_skull_id)
    bc_move_interface = DirichletBC(H_move.sub(0), d,
                            boundary_marker, interface_id)

    bc_move = BlockDirichletBC([bc_move_outer, bc_move_interface])
    sol_move = BlockFunction(H_move)

    A_move = block_assemble([[a]])
    b_move = block_assemble([inner(Constant([0.0,0.0]), v_hat)*dxF])
    bc_move.apply(A_move, b_move)
    block_solve(A_move, sol_move.block_vector(), b_move , "mumps")
    return sol_move[0]



    
