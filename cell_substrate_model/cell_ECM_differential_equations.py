import numpy as np
from signalling_proteins_ODEs import signalling_proteins_ODEs
from scaffolding_proteins_ODEs import scaffolding_proteins_ODEs
from mechanics_ODEs import mechanics_ODEs

def cell_ECM_differential_equations(t,y,N,params):
    
    # signalling proteins
    dcRdt,dcRAdt,dMLCPdt,dMLCPPdt,dMLCKdt,dMLCKPdt,dcCdt,dcCPdt=signalling_proteins_ODEs(y,N,params)
    
    # scaffolding proteins
    dcGdt,dcF1dt,dcF2dt,dcMdt,dcMpdt,dcSpdt,dnfdt,dnhdt,dnb1dt,dnb2dt,dnA1dt,dnA2dt,dns1dt,dns2dt=scaffolding_proteins_ODEs(y,N,params)
    
    # cell-ECM mechanics
    duddt,dwddt, dsigma_celldt, dsigma_ECMdt=mechanics_ODEs(y,N,params)
    
    # concatenate
    fval=np.concatenate((dcGdt,dcF1dt,dcF2dt,dcSpdt,dcMdt,dcMpdt,dnfdt,dnhdt,dnb1dt,dnb2dt,dnA1dt,dnA2dt,dns1dt,dns2dt,
        dcRdt,dcRAdt,dMLCPdt,dMLCPPdt,dcCdt,dcCPdt,dMLCKdt,dMLCKPdt,duddt,dwddt,dsigma_celldt,dsigma_ECMdt))
    
    return fval
