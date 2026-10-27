import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def Maxwell_differential_equations(t,y,N,params):
    
    # -------------------------------
    # Purpose: FOR A MAXWELL MATERIAL 
    #          compute the material displacement and stress at a given time t.
    # Inputs:
    #   t: current time (dimensionless)
    #   y: vector of unknowns, y=[u_0,...,u_N,sigma_0,...,sigma_N]
    #   N: number of spatial points
    #   params: dictionary containing model parameters
    # Outputs:
    #   fval: concatenated derivatives [du/dt, dsigma/dt]
    # -------------------------------
        
    # -------------------------------
    # Extract variables from y
    # -------------------------------
    ca=y[0:N+1] # concentration of chemical A at N+1 spatial locations
    cb=y[N+1:2*N+2] # concentration of chemical B at N+1 spatial locations
    u=y[2*N+2:3*N+3] # displacement at N+1 spatial locations
    sigma=y[3*N+3:4*N+4] # Cauchy stress at N+1 spatial locations
    
    # initialise derivatives
    dudt=np.zeros(N+1)
    dsigmadt=np.zeros(N+1)
    
    # -------------------------------
    # Extract model parameters
    # -------------------------------

    mu0=params['mu0'] # material viscosity
    E0=params['E0'] # Young's modulus
    beta0=params['beta0'] # substrate drag
    hxd=params['hxd'] # spatial step size
    durbdt=params['durbdt'](t) # applied displacement at right boundary
    
    # -------------------------------
    # Mechanical quantities
    # -------------------------------  
    
    Emat=E0*(1e-2+cb) # material Young's modulus
    mumat=mu0*(1e-2+cb) # viscosity array
    beta=beta0*ca # substrate drag coefficient
    
    # -------------------------------
    # Construct linear system M*[dudt;dsigmadt]=F, accounts for substrate drag, viscosity, etc.
    # ------------------------------- 
 
    # Right hand side (forcing) vector, F
    
    F=np.zeros(2*N+2) # initialise 
    
    F[0]=0 # fixed boundary
    F[1:N]=(sigma[2:N+1]-sigma[0:N-1])/(2*hxd)
    F[N]=durbdt # forcing on material displacement U
    
    F[N+1]=-Emat[0]*sigma[0]/mumat[0]
    F[N+2:2*N+1]=-Emat[1:N]*sigma[1:N]/mumat[1:N]
    F[2*N+1]=-Emat[N]*sigma[N]/mumat[N]
    
    # Mass matrix, M (sparse assembly)
    
    M=sparse.lil_matrix((2*N+2, 2*N+2))
    
    # left boundary condition
    
    M[0,0] = 1
    
    M[N+1,0]=3*Emat[0]/(2*hxd)
    M[N+1,1]=-2*Emat[0]/hxd
    M[N+1,2]=Emat[0]/(2*hxd)
    M[N+1,N+1]=1
    
    # interior rows
    for i in range(1,N):
    
        # beta du/dt equation
        M[i,i]=beta[i]
    
        # Maxwell constitutive equation
        M[N+1+i,i-1]=Emat[i]/(2*hxd)
        M[N+1+i,N+1+i]=1
        M[N+1+i,i+1]=-Emat[i]/(2*hxd)
    
    # right boundary condition
    
    M[N,N]=1
    
    M[2*N+1,N-2]=-Emat[N]/(2*hxd)
    M[2*N+1,N-1]=2*Emat[N]/hxd
    M[2*N+1,N]=-3*Emat[N]/(2*hxd)
    M[2*N+1,2*N+1]=1
    
    M=M.tocsc() # convert to CSC for sparse solver
    
    # -------------------------------
    # Solve linear system for time derivatives
    # -------------------------------
    
    Fout = spsolve(M,F)
    
    dudt=Fout[0:N+1]
    dsigmadt=Fout[N+1:2*N+2]
    
    fval=dudt, dsigmadt
    
    return fval
