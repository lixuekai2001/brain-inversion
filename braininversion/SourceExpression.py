from fenics import *
from braininversion.ArrayExpression import getArrayExpression
import numpy as np

def get_source_expression(source_conf, mesh, subdomains,
                          source_domain_id, times):
    vol_scal = 1
    if source_conf["scale_by_total_vol"]:
            dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
            tot_parenchyma_vol = assemble(1.0*dx(source_domain_id))
            vol_scal=1.0/tot_parenchyma_vol
    # source is given by expression:
    if "source_expression" in source_conf.keys():
        g_source = Expression("vol_scal*" + source_conf["source_expression"],
                        t=0.0,degree=2, vol_scal=vol_scal,
                        **source_conf["source_params"])

    # source is given by file:             
    elif "source_file" in source_conf.keys():
        #g_source = InterpolatedSource(source_conf["source_file"])
        data = np.loadtxt(source_conf["source_file"], delimiter=",")
        t = data[:,0]
        inflow = data[:,1]*vol_scal
        if "scaling" in source_conf.keys():
            inflow *= source_conf["scaling"]
        
        values = np.interp(times, t, inflow, period = t[-1])
        g_source = getArrayExpression(values)
    return g_source