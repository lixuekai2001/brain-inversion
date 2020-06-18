from fenics import *
from fenics_adjoint import *
import moola
from .DarcySolver import solve_darcy
from .BiotSolver import solve_biot
import ufl

def update_expression_time(list_of_expressions, time):

    for expr in list_of_expressions:
        if isinstance(expr, ufl.Coefficient):
            expr.t = time
        elif isinstance(expr, ufl.algebra.Operator):
            for op in expr.ufl_operands:
                op.t = time
        elif isinstance(expr, ufl.tensors.ComponentTensor):
            for dimexpr in expr.ufl_operands:
                for op in dimexpr.ufl_operands:
                    try:
                        op.t = time
                    except:
                        pass

def compute_minimization_target(p, minimization_target, boundary_marker):
    mesh = p.function_space().mesh()
    dS = Measure("dS", domain=mesh)
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)
    
    J = 0.0
    for measure_type, target in minimization_target.items():
        for marker_id, func in target.items():
            if measure_type=="dx":
                J += assemble(func(p)*dx(marker_id))
            elif measure_type=="dS":
                J += assemble(func(p)*dS(marker_id))
            elif measure_type=="ds":
                J += assemble(func(p)*ds(marker_id))
            else:
                print(f"Measure type {measure_type} unknown.")
    return J

    
def optimize_darcy_source(mesh, material_parameter, times, minimization_target,
                          boundary_marker, boundary_conditions,
                          time_dep_expr=[], opt_solver="moola_bfgs",
                          control_args=["CG", 1], optimization_parameters=None,
                          initial_guess=None):
    """
    minimization_target = {"ds":{0, func1},
                            "dx":{1:func2}, ...}
    """

    num_steps = len(times)
    T = times[-1]
    K = material_parameter["K"]
    c = material_parameter["c"]

    if control_args=="constant":
        ctrls = [Constant(0.0) for i in times]
    else:
        control_space = FunctionSpace(mesh, *control_args)
        if initial_guess is None:
            initial_guess = [Constant(0.0) for i in times]
        ctrls = [interpolate(c, control_space) for c in initial_guess] 

            
    control = [Control(c) for c in ctrls]
    solution = solve_darcy(mesh, ctrls, T, num_steps, K,
                          boundary_marker, boundary_conditions,
                          c=c, degree=1, theta=0.5)
    initial_solution = []
    J = 0.0
    for i,p in enumerate(solution):
        update_expression_time(time_dep_expr, times[i])
        J += compute_minimization_target(p, minimization_target, boundary_marker)
        initial_solution.append(p.copy())
        
    rf = ReducedFunctional(J, control)
    if opt_solver=="moola_bfgs":
        opt_ctrls = solve_moola_bfgs(rf, ctrls, optimization_parameters)
    elif opt_solver=="ipopt":
        opt_ctrls = solve_ipopt(rf, optimization_parameters)
    elif opt_solver=="scipy":
        opt_ctrls = solve_scipy(rf)
    else:
        print(f"error: {opt_solver} not supported")

    opt_solution = solve_darcy(mesh, opt_ctrls, T, num_steps, K,
                              boundary_marker, boundary_conditions,
                              c=c, degree=1, theta=0.5)
    opt_solution = [s.copy() for s in opt_solution]

    return opt_ctrls, opt_solution, initial_solution

def solve_moola_bfgs(rf, ctrls, optimization_parameters=None):
    if optimization_parameters is None:
        optimization_parameters = {'jtol': 1e-6,
                                    'gtol': 1e-6,
                                    'Hinit': "default",
                                    'maxiter': 100,
                                    'mem_lim': 10}

    problem = MoolaOptimizationProblem(rf)

    f_moola = moola.DolfinPrimalVectorSet(
        [moola.DolfinPrimalVector(c, inner_product="L2") for c in ctrls])
    
    solver = moola.BFGS(problem, f_moola, options=optimization_parameters)
    sol = solver.solve()
    opt_ctrls = sol['control'].data
    return opt_ctrls

def solve_ipopt(rf, optimization_parameters=None):
    if optimization_parameters is None:
        optimization_parameters = {'maximum_iterations': 100}
    problem = MinimizationProblem(rf)
    solver = IPOPTSolver(problem, parameters=optimization_parameters)
    return solver.solve()

def solve_scipy(rf):
    return minimize(rf, options = {'disp': True, "maxiter":2})


def optimize_biot_source(mesh, material_parameter, times, minimization_target,
                         boundary_marker_p, boundary_conditions_p,
                         boundary_marker_u, boundary_conditions_u,
                         time_dep_expr=[], opt_solver="moola_bfgs",
                         control_args=["CG", 1], optimization_parameters=None,
                         initial_guess=None):
    num_steps = len(times)
    T = times[-1]
    if control_args=="constant":
        ctrls = [Constant(0.0) for i in times]
    else:
        control_space = FunctionSpace(mesh, *control_args)
        if initial_guess is None:
            initial_guess = [Constant(0.0) for i in times]
        ctrls = [interpolate(c, control_space) for c in initial_guess] 
    control = [Control(c) for c in ctrls]

    f = Constant((0.0,0.0))

    solution = solve_biot(mesh, f, ctrls, T, num_steps, material_parameter,
                          boundary_marker_p, boundary_conditions_p,
                          boundary_marker_u, boundary_conditions_u)
    initial_solution = []
    J = 0
    for i,up in enumerate(solution):
        u, p_T, p = up.split()
        update_expression_time(time_dep_expr, times[i])
        J += compute_minimization_target(p, minimization_target,
                                         boundary_marker_p)
        initial_solution.append(up.copy())
        
    rf = ReducedFunctional(J, control)
    if opt_solver=="moola_bfgs":
        opt_ctrls = solve_moola_bfgs(rf, ctrls, optimization_parameters)
    if opt_solver=="ipopt":
        opt_ctrls = solve_ipopt(rf, optimization_parameters)
    elif opt_solver=="scipy":
        opt_ctrls = solve_scipy(rf)
    else:
        print(f"error: {opt_solver} not supported")

    opt_solution = solve_biot(mesh, f, opt_ctrls, T, num_steps, material_parameter,
                              boundary_marker_p, boundary_conditions_p,
                              boundary_marker_u, boundary_conditions_u)
    opt_solution = [s.copy() for s in opt_solution]
    
    return opt_ctrls, opt_solution, initial_solution

def optimize_biot_force(mesh, material_parameter, times, minimization_target,
                         boundary_marker_p, boundary_conditions_p,
                         boundary_marker_u, boundary_conditions_u,
                         time_dep_expr=[], opt_solver="moola_bfgs",
                         control_args=["CG", 1], optimization_parameters=None,
                         initial_guess=None, **biot_kwargs):
    gdim = mesh.geometric_dimension()
    num_steps = len(times)
    T = times[-1]
    if control_args=="constant":
        ctrls = [Constant([0.0]*gdim) for i in times]
    else:
        control_space = VectorFunctionSpace(mesh, *control_args)
        if initial_guess is None:
            initial_guess = [Constant([0.0]*gdim) for i in times]
        ctrls = [interpolate(c, control_space) for c in initial_guess] 
    g = Constant(0.0)
    control = [Control(c) for c in ctrls]

    solution = solve_biot(mesh, ctrls, g, T, num_steps, material_parameter,
                          boundary_marker_p, boundary_conditions_p,
                          boundary_marker_u, boundary_conditions_u,
                          **biot_kwargs)
    initial_solution = []
    J = 0
    for i,up in enumerate(solution):
        u, p_T, p = up.split()[0:3]
        update_expression_time(time_dep_expr, times[i])
        J += compute_minimization_target(p, minimization_target,
                                         boundary_marker_p)
        initial_solution.append(up.copy())
        
    rf = ReducedFunctional(J, control)
    if opt_solver=="moola_bfgs":
        opt_ctrls = solve_moola_bfgs(rf, ctrls, optimization_parameters)
    elif opt_solver=="ipopt":
        opt_ctrls = solve_ipopt(rf, optimization_parameters)
    elif opt_solver=="scipy":
        opt_ctrls = solve_scipy(rf)
    else:
        print(f"error: {opt_solver} not supported")

    opt_solution = solve_biot(mesh, opt_ctrls, g, T, num_steps, material_parameter,
                              boundary_marker_p, boundary_conditions_p,
                              boundary_marker_u, boundary_conditions_u,
                              **biot_kwargs)
    opt_solution = [s.copy() for s in opt_solution]
    
    return opt_ctrls, opt_solution, initial_solution