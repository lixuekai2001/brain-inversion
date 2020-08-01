


def solve_timedep_biot_iterative(mesh, f, g, T, num_steps, material_parameter,
              boundary_marker_p, boundary_conditions_p,
              boundary_marker_u, boundary_conditions_u,
              p_initial=Constant(0.0),u_initial=Constant((0.0, 0.0)),
              u_degree=2, p_degree=1, u_nullspace=False,
              solver_params=None):

    time = 0.0
    dt = T / num_steps
    dirichlet_bcs = []
    time_dep_expr = [f,g]

    gdim = mesh.geometric_dimension()
    dim = mesh.topology().dim()
    #c = Constant(material_parameter["c"])
    K = Constant(material_parameter["K"])
    lmbda = Constant(material_parameter["lmbda"])
    mu = Constant(material_parameter["mu"])
    alpha = Constant(material_parameter["alpha"])
    c = alpha**2/lmbda

    V = VectorElement("CG", mesh.ufl_cell(), u_degree)
    W = FiniteElement("CG", mesh.ufl_cell(), p_degree)

    M = MixedElement([V, W, W])
    VW = FunctionSpace(mesh, M)
    V = FunctionSpace(mesh, V)
    W = FunctionSpace(mesh, W)

    up = TrialFunction(VW)
    u, pT, pF = split(up)
    up_n = Function(VW)
    u_, pT_, pF_ = split(up_n)
    vq = TestFunctions(VW)
    v, qT, qF = vq[0], vq[1], vq[2]
    
    g_tilde =  -dt*g - c*pF_ - alpha*div(u_)

    F1 = inner(eps(u), eps(v))*dx - nabla_div(v)*pT*dx - inner(f,v)*dx

    F2 = - nabla_div(u)*qT*dx - (1.0/lmbda)*pT*qT*dx + (alpha/lmbda)*pF*qT*dx

    F3 = (alpha/lmbda)*pT*qF*dx - 2.0*(alpha**2)/lmbda *pF *qF*dx \
         - inner(K*dt*grad(pF), grad(qF))*dx - inner(g_tilde, qF)*dx

    dirichlet_bcs = []

    ds_p = Measure("ds", domain=mesh, subdomain_data=boundary_marker_p)
    ds_u = Measure("ds", domain=mesh, subdomain_data=boundary_marker_u)

    # extract pressure boundary conditions
    for marker_id, bc in boundary_conditions_p.items():
        for bc_type, bc_val in bc.items():
            time_dep_expr.append(bc_val)
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
            time_dep_expr.append(bc_val)
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

    update_expression_time(time_dep_expr, time)

    F = F1 + F2 + F3
    
    (a, L) = system(F)
    #A, b = assemble_system(a, L, dirichlet_bcs)
    #print(f"A symmetric: {as_backend_type(A).mat().isSymmetric(1e-12)}")

    A, b = block_assemble(a, L, bcs=dirichlet_bcs)


    # define preconditioner
    a11 = inner(grad(u), grad(v))*dx
    a22 = pT*qT*dx
    a33 = (alpha**2)/lmbda*pF*qF*dx + inner(K*dt*grad(pF), grad(qF))*dx

    PA = a11 + a22 + a33
    B, _ = block_assemble(PA, L, bcs=dirichlet_bcs)

    null_space = False
    if null_space:
        ns = rigid_body_modes(V)
    else:
        ns= None
    Ap = AMG(B[0,0], pdes=dim, nullspace=ns)
    Bp = Jacobi(B[1,1])
    Cp = AMG(B[2,2], pdes=1)

    P = block_mat([[Ap, 0, 0],
                   [0,  Bp, 0],
                   [0,  0,  Cp]])

    #solver.set_operators(A, AAp)
    #solver.set_operator(A)

    default_solver_params = { "maxiter":100, "relativeconv":False, "tolerance":1e-6 }
    if solver_params:
        default_solver_params.update(solver_params)


    up = Function(VW)
    for n in range(num_steps):
        time += dt
        update_expression_time(time_dep_expr, time)


        #A, b = block_assemble(a, L, bcs=dirichlet_bcs)
        #B, _ = block_assemble(PA, L, bcs=dirichlet_bcs)
        #Ap = AMG(B[0,0], pdes=dim, nullspace=ns)
        #Bp = Jacobi(B[1,1])
        #Cp = AMG(B[2,2], pdes=1)

        #P = block_mat([[Ap, 0, 0],
        #               [0,  Bp, 0],
        #               [0,  0,  Cp]])


        #AAinv = MinRes(A, precond=P, show=2, name='AA^',initial_guess=None ,**default_solver_params)
        #u, pT, pF = AAinv * b

        #u, pT, pF  = map(Function,[V,W,W], [u, pT, pF])
        solve(a==L,up, bcs=dirichlet_bcs)
        up_n.assign(up)
        u, pT, pF = up.copy(deepcopy=True).split()
        yield u, pT, pF


def update_expression_time(list_of_expressions, time):

    for expr in list_of_expressions:
        if isinstance(expr, ufl.tensors.ComponentTensor):
            for dimexpr in expr.ufl_operands:
                for op in dimexpr.ufl_operands:
                    try:
                        op.t = time
                    except:
                        pass
        else:
            update_operator(expr, time)

def update_operator(expr, time):
    if isinstance(expr, ufl.algebra.Operator):
        for op in expr.ufl_operands:
            update_operator(op, time)
    elif isinstance(expr, ufl.Coefficient):
        expr.t = time

