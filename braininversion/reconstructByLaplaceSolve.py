from fenics import *
from fenics_adjoint import *


def solve_laplace(mesh, boundary_marker, boundary_conditions, degree=1):
    V = FunctionSpace(mesh, "CG", degree)
    p = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx", domain=mesh)
    F = inner(grad(p), grad(v))*dx
    a, L = system(F)
    dirichlet_bcs = []
    for marker_id, bc in boundary_conditions.items():
        for bc_type, bc_val in bc.items():
            if bc_type=="Dirichlet":
                bc_d = DirichletBC(V, bc_val, boundary_marker, marker_id)
                dirichlet_bcs.append(bc_d)
                #print(f"Dirichlet boundary added (ID: {marker_id})")
                break
            if bc_type=="Neumann":
                F += sc*K*v*bc_val*ds(marker_id)
                #print(f"Neumann boundary added (ID: {marker_id})")
                break
            if bc_type=="Robin":
                beta = bc_val[0]
                r = bc_val[1]
                F += sc*(-beta*p + r)*K*v*ds(marker_id)
                break
            print(f"Warning! bc_type {bc_type} not supported!")
    p = Function(V)
    solve(a==L, p, bcs=dirichlet_bcs, solver_parameters={"linear_solver": "cg",
                                                         "preconditioner":"hypre_amg"})

    return p

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


def solve_laplace_for_all_timesteps(mesh, boundary_marker, boundary_conditions, 
                                    times, time_dep_expr, degree=1):
    solutions = []
    for t in times:
        update_expression_time(time_dep_expr, t)
        p = solve_laplace(mesh, boundary_marker, boundary_conditions, degree=degree)
        solutions.append(p)
    return solutions
    


def compute_sources_from_pressures(laplace_pressures, c, dt,
                                  initial_pressure=Constant(0.0)):
    V = laplace_pressures[0].function_space()
    forces = []
    for i,p  in enumerate(laplace_pressures):
        if i==0:
            pb = initial_pressure
        else:
            pb = laplace_pressures[i-1]
        f = project(c*(p - pb)/dt, V)
        forces.append(f)
    return forces