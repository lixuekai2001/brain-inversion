from fenics import * 
#from fenics_adjoint import *
from mshr import *

def generate_doughnut_mesh(brain_radius, ventricle_radius, N):
    brain = Circle(Point(0,0), brain_radius)
    ventricle = Circle(Point(0,0), ventricle_radius)
    #aqueduct = Rectangle(Point(-aqueduct_width/2, -brain_radius),
                         #Point(aqueduct_width/2, 0))
    brain = brain - ventricle #- aqueduct
    mesh = Mesh(generate_mesh(brain, N))
    ventricle = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]<r*r*0.9",
                                  r=brain_radius)
    skull = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]>r*r*0.9", 
                                r=brain_radius)
    boundarymarker = MeshFunction("size_t",mesh, 1)
    ventricle.mark(boundarymarker, 1)
    skull.mark(boundarymarker, 2)
    return mesh, boundarymarker

def generate_flower_mesh(brain_radius, n_petals, N):
    r = brain_radius*0.75
    brain = Circle(Point(0,0), r)
    for n in range(n_petals):
        x = Point(r*sin(n*2*pi/n_petals), r*cos(n*2*pi/n_petals))
        brain += Ellipse(x, r/2, r/2)
    for x in [-r/5, r/5]:
        brain -= Ellipse(Point(x, 0), r/8, r/3)
    mesh = Mesh(generate_mesh(brain, N))
    ventricle = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]<r*r*0.9", r=r)
    skull = CompiledSubDomain("on_boundary && x[0]*x[0] + x[1]*x[1]>r*r*0.9", r=r)
    boundarymarker = MeshFunction("size_t",mesh, 1)
    ventricle.mark(boundarymarker, 1)
    skull.mark(boundarymarker, 2)
    return mesh, boundarymarker