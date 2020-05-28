import pyvista as pv
from itkwidgets import view

def interactive_vizualize_ITK(filename):
    data = pv.read(filename)
    plotter = pv.PlotterITK()    
    plotter.add_mesh(data)       
    return plotter.show() 

def interactive_clip_ITK(filename):
    data = pv.read(filename)
    clipped_data = data.clip()
    plotter = pv.PlotterITK()     
    plotter.add_mesh(clipped_data) 
    return plotter.show()               