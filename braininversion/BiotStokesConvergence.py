from fenics import *
import sympy as sym
from sympy import symbols, pi
from sympy import derive_by_array, tensorcontraction
from numpy import isclose
from multiphenics import *
import numpy as np
import os
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from braininversion.BiotNavierStokesSolver import solve_biot_navier_stokes

dirs = ["../results/MMS/", "../results/MMS/spatial_expressions/",
        "../results/MMS/temporal_expressions/" ]
for directories in dirs:
    try:
        os.mkdir(directories)
    except:
        pass


porous_id = 1
fluid_id = 2
interf_id = 1
boundary_id = 2

names = ["u", "d", "pF", "pP", "phi"]

fluid_restriction_file = "../meshes/MMS_fluid.rtc.xdmf"
porous_restriction_file = "../meshes/MMS_porous.rtc.xdmf"
outfile_path = "../results/MMS_sol/sol.xdmf"

c, kappa, lmbda, mu_s, rho_s, mu_f,rho_f, alpha, t, gamma = sym.symbols("c, kappa, lambda, mu_s, rho_s, mu_f,rho_f, " +
                                                                 "alpha, t, gamma")

x,y,z = sym.symbols("x, y, z")


def row_wise_div(tensor):
    return sym.Matrix(tensorcontraction(derive_by_array( tensor, [x,y]), (0,2)))
    
def row_wise_grad(u):
    return u.jacobian([x,y])

def eps(u):
    return 0.5*(row_wise_grad(u) + sym.transpose(row_wise_grad(u)))

def div(u):
    return sym.trace(u.jacobian([x,y]))

def grad(p):
    return sym.Matrix([sym.diff(p,x), sym.diff(p,y)])

n = sym.Matrix([1,0])
tau = sym.Matrix([0,1])
alpha = 1
I = sym.eye(2)

def compute_spatial_mm(u, interf_x):

    # by 1
    dtd2 = sym.sqrt(kappa)/gamma*(sym.diff(u[0], y) + sym.diff(u[1], x)) + u[1]
    d_2 = sym.integrate(dtd2, t)

    dxd1 = -lmbda*sym.diff(d_2, y) / (2*mu_s + lmbda)

    d_1 = sym.integrate(dxd1, x) 
    
    d = sym.Matrix([d_1, d_2])
    rem = 2*(mu_s*eps(d)[1,0] - mu_f*eps(u)[1,0])
    d = sym.Matrix([d_1 - sym.integrate(rem/(mu_s), y).subs({x:interf_x}), d_2])

    p_p = sym.integrate(mu_f/kappa *( sym.diff(d, t) - u)[0] , x)
    p_f = (p_p + 2*mu_f*sym.diff(u[0], x))# + 10*sym.sin(pi*x)*sym.cos(pi*y)

    phi = p_p - lmbda*div(d)
    return d, p_p, p_f, phi

def compute_temporal_mm(u, p_p=1.0):
    d = sym.integrate(u, t)
    phi = p_p
    p_f = p_p
    return d, p_p, p_f, phi
    

def compute_forcing_terms(u, p_f, d, p_p):
    phi = alpha*p_p - lmbda*div(d)

    f_fluid = sym.diff(u, t) - row_wise_div(2*mu_f*eps(u) - p_f*I)/rho_f
    f_porous = -row_wise_div(2*mu_s*eps(d) - phi*I)/rho_s
    
    g_source = (c + alpha**2/lmbda)*sym.diff(p_p, t) - alpha/lmbda*sym.diff(phi,t) - div(kappa/mu_f * grad(p_p))
    
    return f_fluid, f_porous, g_source
    
    
def check_interface_conditions(u, p_f, d, p_p, interf_x):
    print(f"div u = 0 :  {div(u).equals(0)}")
    #u = u.subs({x:interf_x})
    #p_f = p_f.subs({x:interf_x})
    #d = d.subs({x:interf_x})
    #p_p = p_p.subs({x:interf_x})
    
    phi = alpha*p_p - lmbda*div(d)

    dtd = sym.diff(d, t)
    # check continuity of normal flux
    lhs = n.T*u
    rhs = n.T * (dtd - (kappa/mu_f * grad(p_p)))
    
    print(f"continuity of normal flux :  { rhs.equals(lhs)}")
    
    #check momentum conservation
    rhs = (2*mu_f*eps(u) - p_f*I)*n
    lhs = (2*mu_s*eps(d) - phi*I)*n
    rhs = rhs.subs({x:interf_x})
    lhs = lhs.subs({x:interf_x})

    print(f"momentum conservation 1 :  { rhs[0].equals(lhs[0])}")
    print(f"momentum conservation 2 :  { rhs[1].equals(lhs[1])}")

    # balance of fluid normal stress
    lhs = (-n.T*(2*mu_f*eps(u) - p_f*I)*n).simplify()[0]
    rhs = p_p
    rhs = rhs.subs({x:interf_x})
    lhs = lhs.subs({x:interf_x})
    
    print(f"balance of fluid normal stress :  { rhs.equals(lhs)}")
    
    # BJS 
    lhs = -n.T*(((2*mu_f*eps(u) - p_f*I))*tau)
    rhs =  gamma*mu_f/sym.sqrt(kappa)*(u - dtd).T*tau
    rhs = rhs.subs({x:interf_x})
    lhs = lhs.subs({x:interf_x})
    
    print(f"BJS :  { rhs.equals(lhs)}")
    
def generate_expression(func, material_parameter, expr_degree=6):
    xcpp, ycpp = sym.symbols('x[0], x[1]')
    func  = func.subs(x, xcpp).subs(y, ycpp)
    try:
        s = func.shape
        cpp0 = sym.printing.ccode(func[0])
        cpp1 = sym.printing.ccode(func[1])

        expr = Expression((cpp0, cpp1), **material_parameter, t=0, degree=expr_degree + 1)
        
    except AttributeError:
        cpp = sym.printing.ccode(func)
        cpp.replace('M_PI', 'pi') 
        expr = Expression(cpp, **material_parameter, t=0, degree=expr_degree)

    return expr
    
def generate_mesh(N):

    mesh = RectangleMesh(Point(-1.0, 0), Point(1.0, 1), 2*N, N)
    #mesh = UnitSquareMesh(N, N)
    subdomains = MeshFunction("size_t", mesh, 2)
    subdomains.set_all(0)
    boundaries = MeshFunction("size_t", mesh, 1)
    boundaries.set_all(0)
    
    x_coord_interf = 0
    class MStokes(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= x_coord_interf

    class MBiot(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= x_coord_interf

    class Interface(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_coord_interf)
        
    outerBoundary = CompiledSubDomain("on_boundary")

    MBiot().mark(subdomains, porous_id)
    MStokes().mark(subdomains, fluid_id)
    Interface().mark(boundaries, interf_id)
    outerBoundary.mark(boundaries, boundary_id)

    # ******* Set subdomains, boundaries, and interface ****** #

    OmS = MeshRestriction(mesh, MStokes())
    OmB = MeshRestriction(mesh, MBiot())
    OmS._write(fluid_restriction_file)
    OmB._write(porous_restriction_file)
    
    return mesh, boundaries, subdomains


def compute_order(error, h):
    h = np.array(h)
    err_ratio = np.array(error[:-1]) / np.array(error[1:])
    return np.log(err_ratio)/np.log(h[:-1] / h[1:])

def compute_errornorms(num_results, exact_sols, names, times, domains, norm_type="L2"):
    errors = {n:0.0 for n in names}
    for n in names:
        #for i,t in enumerate(times[1:]):
        e_sol = exact_sols[n]
        e_sol.t = times[-1]
        try:
            e = errornorm(e_sol, num_results[n][-1], norm_type=norm_type, mesh=domains[n])
        except RuntimeError:
            e = np.nan
        errors[n] += e
    return errors

def compute_spatial_convergence(resolutions, T, num_steps, 
                                u, p_f, d, p_p, phi, material_parameter,
                                f_fluid, f_porous, g_source,
                                u_degree, p_degree):
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    
    u_exact, p_f_exact, d_exact, p_p_exact, phi_exact = [generate_expression(func, material_parameter,expr_degree=6)
                                                  for func in [u, p_f, d, p_p, phi]]

    exact_sols = {"pF":p_f_exact, "pP":p_p_exact, "phi":phi_exact, "d":d_exact, "u":u_exact}

    f_fluid, f_porous, g_source = [generate_expression(func, material_parameter, expr_degree=6) for
                                  func in (f_fluid, f_porous, g_source,)]

    sliprate = material_parameter["gamma"]

    time_dep_expr = [u_exact,p_f_exact, d_exact,p_p_exact,phi_exact, f_fluid, f_porous, g_source]

    boundary_conditions = [
        {boundary_id: {0:u_exact}},
        {boundary_id: {1:p_f_exact}},
        {boundary_id: {2:d_exact}},
        {boundary_id: {3:p_p_exact}},
        {boundary_id: {4:phi_exact}},
        #{interf_id: {0:u_exact}},
        #{interf_id: {1:p_f_exact}},
        #{interf_id: {2:d_exact}},
        #{interf_id: {3:p_p_exact}},
        #{interf_id: {4:phi_exact}},
        #{0: {0:u_exact}},
        #{0: {1:p_f_exact}},
        #{0: {2:d_exact}},
        #{0: {3:p_p_exact}},
        ]


    L2_convergence = {n:[] for n in names}
    H1_convergence = {n:[] for n in names}

    h = []
    for N in resolutions:
        mesh, boundaries, subdomains = generate_mesh(N)
        h.append(mesh.hmin())
        dxF = Measure("dx", domain=mesh, subdomain_data=subdomains)(fluid_id)
        dxP = Measure("dx", domain=mesh, subdomain_data=subdomains)(porous_id)
        por_mesh = SubMesh(mesh, subdomains, porous_id)
        fluid_mesh = SubMesh(mesh, subdomains, fluid_id)

        domains = {"u":fluid_mesh, "d":por_mesh, "pF":fluid_mesh, "pP":por_mesh, "phi":por_mesh}
        #domains = {"u":dxF, "d":dxP, "pF":dxF, "pP":dxP, "phi":dxP}

        sol = solve_biot_navier_stokes(mesh, T, num_steps,
                                 material_parameter, 
                                 boundaries,
                                 subdomains,
                                 boundary_conditions,
                                 porous_restriction_file,
                                 fluid_restriction_file,
                                 sliprate=sliprate,
                                 g_source=g_source,
                                 f_porous=f_porous,
                                 f_fluid=f_fluid,
                                 initial_pP=p_p_exact,
                                 initial_pF=p_f_exact,
                                 initial_phi=phi_exact,
                                 initial_u=u_exact,
                                 initial_d=d_exact,
                                 filename=outfile_path,
                                 elem_type="TH",
                                 interf_id=interf_id,
                                 linearize=True,
                                 move_mesh=False,
                                 time_dep_expr=time_dep_expr,
                                 u_degree=u_degree, p_degree=p_degree)
        infile = XDMFFile(outfile_path)


        W = FunctionSpace(mesh, "CG", p_degree)
        V = VectorFunctionSpace(mesh, "CG", u_degree)

        spaces = {"pF":W, "pP":W, "phi":W, "d":V, "u":V}
        num_results = {n:[] for n in names}
        for n in names:
            for i in range(num_steps + 1):
                f = Function(spaces[n])
                f.set_allow_extrapolation(True)
                infile.read_checkpoint(f, n, i)
                num_results[n].append(f)
        infile.close()

        L2_norm = compute_errornorms(num_results, exact_sols, names, times,domains, "L2")
        H1_norm = compute_errornorms(num_results, exact_sols, names, times,domains, "H1")

        for n in names:
            L2_convergence[n].append(L2_norm[n])
            H1_convergence[n].append(H1_norm[n])

    return L2_convergence,H1_convergence, num_results, np.array(h), exact_sols


def compute_temporal_convergence(num_time_steps, T,
                                u, p_f, d, p_p, phi, material_parameter,
                                u_degree, p_degree, spatial_resolution=40):

    u_exact, p_f_exact, d_exact, p_p_exact, phi_exact = [generate_expression(func, material_parameter, expr_degree=6)
                                                  for func in [u, p_f, d, p_p, phi]]

    exact_sols = {"pF":p_f_exact, "pP":p_p_exact, "phi":phi_exact, "d":d_exact, "u":u_exact}


    f_fluid, f_porous, g_source = [generate_expression(func, material_parameter, expr_degree=6) for
                                    func in compute_forcing_terms(u, p_f, d, p_p)]

    sliprate = material_parameter["gamma"]

    time_dep_expr = [u_exact,p_f_exact, d_exact,p_p_exact,phi_exact, f_fluid, f_porous, g_source]


    boundary_conditions = [
        {boundary_id: {0:u_exact}},
        {boundary_id: {1:p_f_exact}},
        {boundary_id: {2:d_exact}},
        {boundary_id: {3:p_p_exact}},
        {boundary_id: {4:phi_exact}},]

    convergence = {n:[] for n in names}
    delta_t = []
    spatial_resolution = 40
    mesh, boundaries, subdomains = generate_mesh(spatial_resolution)
    #dxF = Measure("dx", domain=mesh, subdomain_data=subdomains)(fluid_id)
    #dxP = Measure("dx", domain=mesh, subdomain_data=subdomains)(porous_id)
    por_mesh = SubMesh(mesh, subdomains, porous_id)
    fluid_mesh = SubMesh(mesh, subdomains, fluid_id)

    domains = {"u":fluid_mesh, "d":por_mesh, "pF":fluid_mesh, "pP":por_mesh, "phi":por_mesh}
    #domains = {"u":dxF, "d":dxP, "pF":dxF, "pP":dxP, "phi":dxP}

    for num_steps in num_time_steps:
        solve_biot_navier_stokes(mesh, T, num_steps,
                                 material_parameter, 
                                 boundaries,
                                 subdomains,
                                 boundary_conditions,
                                 porous_restriction_file,
                                 fluid_restriction_file,
                                 sliprate=sliprate,
                                 g_source=g_source,
                                 f_porous=f_porous,
                                 f_fluid=f_fluid,
                                 initial_pP=p_p_exact,
                                 initial_pF=p_f_exact,
                                 initial_phi=phi_exact,
                                 initial_u=u_exact,
                                 initial_d=d_exact,
                                 filename=outfile_path,
                                 elem_type="TH",
                                 interf_id=interf_id,
                                 linearize=True,
                                 move_mesh=False,
                                 time_dep_expr=time_dep_expr)
        
        infile = XDMFFile(outfile_path)

        W = FunctionSpace(mesh, "CG", 1)
        V = VectorFunctionSpace(mesh, "CG", 2)

        spaces = {"pF":W, "pP":W, "phi":W, "d":V, "u":V}
        num_results = {n:[] for n in names}
        dt = T/num_steps
        times = np.linspace(0, T, num_steps + 1)
        delta_t.append(dt)
        for n in names:
            for i in range(num_steps + 1):
                f = Function(spaces[n])
                infile.read_checkpoint(f, n, i)
                num_results[n].append(f)
        infile.close()

        e_norms = compute_errornorms(num_results, exact_sols, names, times, domains)
        for n in names:
            convergence[n].append(e_norms[n])
    return convergence, num_results, np.array(delta_t)
