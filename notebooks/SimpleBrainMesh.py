from fenics import * 
from mshr import *

def generate_doughnut_mesh(brain_radius, ventricle_radius, aqueduct_width, N):
    brain = Circle(Point(0,0), brain_radius)
    ventricle = Circle(Point(0,0), ventricle_radius)
    aqueduct = Rectangle(Point(-aqueduct_width/2, -brain_radius),
                         Point(aqueduct_width/2, 0))
    brain = brain - ventricle - aqueduct
    mesh = generate_mesh(brain, N)
    return mesh