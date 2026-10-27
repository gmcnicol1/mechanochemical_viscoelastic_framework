import numpy as np
from biochemistry_differential_equations import biochemistry_differential_equations

def overall_differential_equations(t,y,N,params,mech_rhs):
    
    # biochemistry
    dcadt,dcbdt=biochemistry_differential_equations(t,y,N,params)
    
    # mechanics
    duddt,dsigmadt=mech_rhs(t,y,N,params)
    
    fval=np.concatenate((dcadt,dcbdt,duddt,dsigmadt))
    
    return fval

