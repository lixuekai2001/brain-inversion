from fenics import *
import numpy as np
#from fenics_adjoint import *
import ufl


def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def solve_biot(mesh, f, g, T, num_steps, material_parameter,
              boundary_marker_p, boundary_conditions_p,
              boundary_marker_u, boundary_conditions_u,
              p_initial=Constant(0.0),u_initial=Constant((0.0, 0.0)),
              theta=0.5, u_degree=2, p_degree=1, solver_type="LU", u_nullspace=False):
              
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

    if u_nullspace:
        print("using Lagrange multipliers for RM removal")

        # Nullspace of 2D rigid motions 
        # two translations + one rotation
        rm = rigid_motions(mesh)

        R = VectorElement("R", mesh.ufl_cell(), 0)
        RM =  MixedElement([R]*len(rm))
        M = MixedElement([V, W, W, RM])
        VW = FunctionSpace(mesh, M)
        up = TrialFunction(VW)
        u, pT, pF, rs = split(up)
        up_n = Function(VW)
        u_, pT_, pF_, rs_ = split(up_n)
        vq = TestFunctions(VW)
        v, qT, qF = vq[0], vq[1], vq[2]
        ss = vq[3]

    else:
        print("no rigid motion")
        M = MixedElement([V, W, W])
        VW = FunctionSpace(mesh, M)
        up = TrialFunction(VW)
        u, pT, pF = split(up)
        up_n = Function(VW)
        u_, pT_, pF_ = split(up_n)
        vq = TestFunctions(VW)
        v, qT, qF = vq[0], vq[1], vq[2]

    #u_.assign(interpolate(u_initial, VW.sub(0).collapse()))
    #p_.assign(interpolate(p_initial, VW.sub(2).collapse()))
    #p_T_.assign(project(lmbda*div(u_) - p_, VW.sub(1).collapse())  )


    dtdpF =  (pF - pF_)/dt
    dtdpT = (pT - pT_)/dt
    
    pF_theta = theta*pF + (1.0 - theta)*pF_
    pT_theta = theta*pT + (1.0 - theta)*pT_
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

    #F1 = 2*mu*inner(eps(u_theta), eps(v))*dx + p_T_theta*div(v)*dx - inner(f,v)*dx
    F1 = inner(2*mu*sym(grad(u_theta)), sym(grad(v)))*dx + pT_theta*div(v)*dx - inner(f,v)*dx

    if u_nullspace:
        # Lagrange multipliers contrib to a
        for i , e in enumerate ( rm ):
            r = rs[i]
            s = ss[i]
            F1 += r* inner (v , e )*dx + s* inner (u , e )*dx

    #F2 = (c* dtdp + alpha/lmbda *(dtdp_T + dtdp) )*q *dx + \
    #    K *inner(grad(p_theta), grad(q))*dx - g*q*dx

    #F2 = qT * div(u_theta) *dx #- alpha/lmbda*(pF_theta + pT_theta) *qT *dx
    F2 = qT * div(u_theta) *dx - alpha/lmbda*(pF_theta + pT_theta) *qT *dx


    #F3 = (c* dtdpF + (1/lmbda) *(dtdpT + alpha**2*dtdpF) )*qF *dx + K *inner(grad(pF_theta), grad(qF))*dx - g*qF*dx   
    F3 = (c* dtdpF + (alpha/lmbda) *(dtdpT + alpha*dtdpF) )*qF *dx + K *inner(grad(pF_theta), grad(qF))*dx - g*qF*dx   

    #F3 = (lmbda*div(u_theta) - p_theta - p_T_theta) *q_T *dx

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
                F2 += K*qF*bc_val*ds_p(marker_id)
                time_dep_expr_full.append(bc_val)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F2 += (-beta*pF + r)*K*qF*ds_p(marker_id)
                time_dep_expr_full.append(r)
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
                F1 += inner(v, bc_val)*ds_u(marker_id)
                time_dep_expr_full.append(bc_val)
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F1 += inner(-beta*p + r,v)*ds_u(marker_id)
                time_dep_expr_full.append(r)
                break
            print(f"Warning! bc_type {bc_type} not supported!")
    
    F = F1 + F2 + F3
    
    (a, L) = system(F)
    #A, b = assemble_system(a, L, dirichlet_bcs)
    A = assemble(a)
    for bc in dirichlet_bcs:
        bc.apply(A)
    #sym_tol = 1e-10
    #print(f" Matrix A is symmetric: {as_backend_type(A).mat().isSymmetric(sym_tol)} with tolerance {sym_tol}")

    #eigensolver = SLEPcEigenSolver(as_backend_type(A))
    #eigensolver.parameters['problem_type'] = 'hermitian'
    #eigensolver.parameters['spectrum'] = 'smallest magnitude'
    #eigensolver.solve(5)

    #assert eigensolver.get_number_converged() > 3
    #print('3. Eigenvalues:')
    #for i in range(5):
        #w = eigensolver.get_eigenvalue(i)
        #print(w)
   
    if solver_type=="LU":
        solver = LUSolver(A, "mumps")
        #solver.parameters["symmetric"] = True

    elif solver_type=="krylov":
        solver = PETScKrylovSolver("gmres","hypre_amg")
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["maximum_iterations"] = 100000
        solver.parameters["monitor_convergence"] = True

        pu = mu * inner(grad(u), grad(v))*dx
        #pp = sum(alpha[i]*alpha[i]/lmbda*p[i+1]*w[i+1]*dx() + dt*theta*K[i]*inner(grad(p[i+1]), grad(w[i+1]))*dx() \
        #            + (c[i] + sum([dt*theta*S[i][j] for j in list(As[:i])+list(As[i+1:])]))*p[i+1]*w[i+1]*dx() for i in As)
        pp = alpha*alpha/lmbda*pF*qF*dx
        ppt = pT*qT *dx
        prec = pu + pp + ppt
        prec, bdummy = system(prec)
        P, btmp = assemble_system(prec, L, dirichlet_bcs)

        solver.set_operators(A, P)
        #solver.set_operator(A)

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
        #(a, L) = system(F)
        b = assemble(L)
        #A, b = assemble_system(a, L, dirichlet_bcs)
        for bc in dirichlet_bcs:
            bc.apply(b)
        # Compute solution
        #solver.solve(up.vector(), b)
        solve(a==L, up ,bcs=dirichlet_bcs)
        up_n.assign(up)
        yield up_n
        
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


class OneExpression(UserExpression):
    def eval(self, value, x):
        value[0] = 1.0
    def value_shape(self):
        return (1,)

def rigid_motions(mesh):

    gdim = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    c = np.array([assemble(xi*dx) for xi in x])
    volume = assemble(Constant(1)*dx(domain=mesh))
    c /= volume
    c_ = c
    c = Constant(c)

    if gdim == 1:       
        translations = [(OneExpression())]        
        return translations
    
    if gdim == 2:
        translations = [Constant((1./sqrt(volume), 0)),
                        Constant((0, 1./sqrt(volume)))]

        # The rotational energy
        r = assemble(inner(x-c, x-c)*dx)

        C0, C1 = c.values()
        rotations = [Expression(('-(x[1]-C1)/A', '(x[0]-C0)/A'), 
                                C0=C0, C1=C1, A=sqrt(r), degree=1)]
        
        return translations + rotations

    if gdim == 3:
        # Gram matrix of rotations
        R = np.zeros((3, 3))

        ei_vectors = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0,1))]
        for i, ei in enumerate(ei_vectors):
            R[i, i] = assemble(inner(cross(x-c, ei), cross(x-c, ei))*dx)
            for j, ej in enumerate(ei_vectors[i+1:], i+1):
                R[i, j] = assemble(inner(cross(x-c, ei), cross(x-c, ej))*dx)
                R[j, i] = R[i, j]

        # Eigenpairs
        eigw, eigv = np.linalg.eigh(R)
        if np.min(eigw) < 1E-8: warning('Small eigenvalues %g' % np.min(eigw))
        eigv = eigv.T
        # info('Eigs %r' % eigw)

        # Translations: ON basis of translation in direction of rot. axis
        # The axis of eigenvectors is ON but dont forget the volume
        translations = [Constant(v/sqrt(volume)) for v in eigv]

        # Rotations using the eigenpairs
        C0, C1, C2 = c_

        def rot_axis_v(pair):
            '''cross((x-c), v)/sqrt(w) as an expression'''
            v, w = pair
            return Expression(('((x[1]-C1)*v2-(x[2]-C2)*v1)/A',
                               '((x[2]-C2)*v0-(x[0]-C0)*v2)/A',
                               '((x[0]-C0)*v1-(x[1]-C1)*v0)/A'),
                               C0=C0, C1=C1, C2=C2, 
                               v0=v[0], v1=v[1], v2=v[2], A=sqrt(w),
                               degree=1)
        # Roations are discrebed as rot around v-axis centered in center of
        # gravity 
        rotations = list(map(rot_axis_v, zip(eigv, eigw)))
   
        return translations + rotations

