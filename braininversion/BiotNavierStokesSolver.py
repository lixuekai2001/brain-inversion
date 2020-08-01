from dolfin import *
from multiphenics import *
import os
import ufl
import numpy as np
from pathlib import Path

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
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)


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
                             g_source=Constant(0.0),
                             move_mesh=False,
                             filename=None,
                             linearize=False,
                             interf_id=1,
                             u_degree=2, p_degree=1):
    

    gdim = mesh.geometric_dimension()
    fluid_id = 2
    porous_id = 1

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
    g = Constant([0.0]*gdim)
    f = Constant([0.0]*gdim)

    n = FacetNormal(mesh)("+")

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


    # extract Dirichlet boundary conditions
    bcs = []
    for bc in boundary_conditions:
        for marker_id, subsp_vc_val in bc.items():
            for subspace_id, bc_val in subsp_vc_val.items():
                bc_d = DirichletBC(H.sub(subspace_id), bc_val,
                                    boundary_marker, marker_id)
                bcs.append(bc_d)

    bcs = BlockDirichletBC(bcs)

    def proj_t(vec):
        return vec-dot(vec,n)*n

    # define forms
    def a_F(u,v):
        return rho_f*dot(u/dt, v)*dxF \
                + 2*mu_f*inner(eps(u), eps(v))*dxF \
                + (gamma*mu_f/sqrt(kappa))*inner(proj_t(u("+")), v("+"))*ds_Sig

    if linearize:
        def c_F(u,v):
            return rho_f * dot(dot(u_n, nabla_grad(u)), v)*dxF
    else:
        def c_F(u,v):
            return rho_f * dot(dot(u, nabla_grad(u)), v)*dxF

    def b_1_F(v, qF):
        return  -qF*div(v)*dxF

    def b_2_Sig(v, qP):
        return qP("+")*inner(v("+"), n)*ds_Sig + div(v)*qP*Constant(0.0)*dxD

    def b_3_Sig(v, d):
        return - ((gamma*mu_f/sqrt(kappa))*inner(proj_t(v("+")), d("+"))*ds_Sig)


    def b_4_Sig(w,qP):
        return -qP("+") * dot(w("+"),n)*ds_Sig + div(w)*qP*Constant(0.0)*dxD

    def a_1_P(d, w):
        return 2.0*mu_s*inner(eps(d), eps(w))*dxP \
                + (gamma*mu_f/sqrt(kappa))*inner(proj_t(d("+")/dt), w("+"))*ds_Sig


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
        return rho_f *dot(g, v)*dxF

    def F_P(w):
        return rho_s*inner(f, w)*dxP

    def G(qP):
        return rho_f*inner(g, grad(qP))*dxP \
                - rho_f*inner(g, n)*qP("+")*ds_Sig   + qP*Constant(0.0)*dxD \
                + g_source*qP*dxP

    def F_F_n(v):
        return F_F(v) + rho_f*inner(u_n/dt, v)*dxF + b_3_Sig(v, d_n/dt)

    def F_P_n(w):
        return F_P(w) + (gamma*mu_f/sqrt(kappa))*inner(proj_t(d_n("+")/dt), w("+"))*ds_Sig

    def G_n(qP):
        return G(qP) + (c + (alpha**2)/lmbda)*pP_n/dt*qP*dxP \
            + b_4_Sig(d_n/dt, qP) - b_2_P(phi_n/dt, qP)


    # define system:
    # order trial: u, pF, d, pP, phi
    # order test: v, qF, w, qP, psi


    lhs = [[ a_F(u,v)+c_F(u,v), b_1_F(v, pF), b_3_Sig(v, d/dt) , b_2_Sig(v, pP), 0                 ],
        [ b_1_F(u, qF)     ,  0          , 0                , 0             , 0                 ],
        [ b_3_Sig(u, w)    ,  0          , a_1_P(d,w)       , b_4_Sig(w, pP), b_1_P(w, phi)     ],
        [ b_2_Sig(u, qP)   ,  0          , b_4_Sig(d/dt, qP), a_2_P(pP, qP) , -b_2_P(phi/dt, qP)], # sign of b_2_Sig(u, qP)?
        [ 0                ,  0          , b_1_P(d, psi)    , b_2_P(psi, pP), -a_3_P(phi, psi) ]]

    if linearize:
        sol = BlockFunction(H)
        AA = block_assemble(lhs, keep_diagonal=False)
        bcs.apply(AA)
        solver = PETScLUSolver(AA, "mumps")
        
    if filename:
        output = XDMFFile(filename + ".xdmf")
        output_checkp = XDMFFile(filename + "_checkp.xdmf")

        output.parameters["rewrite_function_mesh"] = False
        output.parameters["functions_share_mesh"] = True   
        output.parameters["flush_output"] = True    
 
        output_checkp.parameters["functions_share_mesh"] = True
        output_checkp.parameters["rewrite_function_mesh"] = False

    names = ["velocity u", "fluid pressure pF", "displacement d",
            "fluid pressure in porous domain pP", "total pressure phi"]
    short_names = ["u", "pF", "d", "pP", "phi"]
    names = short_names

    def solve_linearized():
        rhs = [F_F_n(v), 0, F_P_n(w), G_n(qP), 0]
        FF = block_assemble(rhs)
        bcs.apply(FF)
        solver.solve(sol.block_vector(), FF)
        sol.block_vector().block_function().apply("to subfunctions")
        block_assign(previous, sol)


    def solve_nonlinear():
        rhs = [F_F_n(v), 0, F_P_n(w), G_n(qP), 0]
        F = np.array(lhs).sum(axis=1) - np.array(rhs)
        J = block_derivative(F, trial, dtrial)
        problem = BlockNonlinearProblem(F, trial, bcs, J)
        solver = BlockPETScSNESSolver(problem)
        solver.parameters.update(snes_solver_parameters["snes_solver"])
        solver.solve()
        block_assign(previous, trial)

    for i in range(num_steps):
        time = (i + 1)*dt.values()[0]
        for expr in time_dep_expr:
            expr.t = time
            expr.i = i
        if linearize:
            solve_linearized()
        else:
            solve_nonlinear()
        
        results = block_split(previous)
        #[r.rename(names[i], names[i]) for i,r in enumerate(results)]

        if move_mesh:
            d =  results[2]
            d_hat = harmonic_extension(mesh, d, boundary_marker,
                                       fluidrestriction, dxF, interf_id)
            
            ALE.move(mesh, d)
            ALE.move(mesh, d_hat)

        if filename:
            fluid_mesh = SubMesh(mesh, subdomain_marker, fluid_id)
            por_mesh = SubMesh(mesh, subdomain_marker, porous_id)


            for k,r in enumerate(results):
                r.rename(names[k], names[k])
                if k==4:
                    Vsub = FunctionSpace(por_mesh,"CG", 1)

                    output.write(interpolate(r, Vsub), time)
                
                if isinstance(r.ufl_element(), VectorElement) and elem_type=="mini":
                    
                    V = VectorFunctionSpace(mesh, "CG", 1)
                    output_checkp.write_checkpoint(interpolate(r, V), r.name(), time, append=True)
                else:
                    output_checkp.write_checkpoint(r, r.name(), time, append=True)


    if filename:
        output.close()
        output_checkp.close()
        path = Path(filename + ".xdmf")
        text = path.read_text()
        text = text.replace(",", ".")
        path.write_text(text)
        path = Path(filename + "_checkp.xdmf")
        text = path.read_text()
        text = text.replace(",", ".")
        path.write_text(text)

    return results


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



    