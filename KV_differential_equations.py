import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def KV_differential_equations(t,y,N,params):
    
    # -------------------------------
    # Purpose: FOR A KELVIN-VOIGT MATERIAL 
    #          compute the material displacement and stress at a given time t.
    # Inputs:
    #   t: current time (dimensionless)
    #   y: vector of unknowns, y=[u_0,...,u_N]
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
    sigma=y[3*N+3:4*N+4] # Cauchy stress at N+1 spatial locations diagnostic only (not evolved in Kelvin-Voigt model model)
    
    # initialise derivatives
    dudt=np.zeros(N+1)
   
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
    # Construct linear system M*[dudt]=F, accounts for substrate drag, viscosity, etc.
    # ------------------------------- 
 
    # Right hand side (forcing) vector, F
    
    F=np.zeros(N+1) # initialise 
    
    F[0]=0 # fixed boundary
    F[1:N]=(((Emat[2:N+1]-Emat[0:N-1])/(4*hxd**2)+Emat[1:N]/hxd**2)*u[2:N+1]
         +(-(Emat[2:N+1]-Emat[0:N-1])/(4*hxd**2)+Emat[1:N]/hxd**2)*u[0:N-1]
         -(2*Emat[1:N]/hxd**2)*u[1:N])
    F[N]=durbdt # forcing on material displacement U
    
    # Mass matrix, M (sparse)
    
    lower=np.zeros(N)
    diag=np.zeros(N+1)
    upper=np.zeros(N)

    # interior rows
    for i in range(1, N):

        lower[i-1]=(-mumat[i]/hxd**2+(mumat[i+1]-mumat[i-1])/(4*hxd**2))
        diag[i]=beta[i]+2*mumat[i]/hxd**2
        upper[i]=(-mumat[i]/hxd**2-(mumat[i+1]-mumat[i-1])/(4*hxd**2))

    # boundary conditions
    diag[0]=1 # right boundary condition beta*du/dt=0
    diag[N]=1 # right boundary condition beta*du/dt=durbdt

    # assemble sparse tridiagonal matrix
    M=sparse.diags([lower, diag, upper], offsets=[-1,0,1], shape=(N+1,N+1),format="csc")
       
    # -------------------------------
    # Solve linear system for time derivatives
    # -------------------------------
    
    Fout=spsolve(M,F)
    dudt=Fout[0:N+1]
    
    # -------------------------------
    # Compute KV stress
    # -------------------------------
    
    strain=np.zeros(N+1)
    strain[0]=(-3*u[0]+4*u[1]-u[2])/(2*hxd)
    strain[1:N]=(u[2:N+1]-u[0:N-1])/(2*hxd)  # central difference
    strain[N]=(3*u[N]-4*u[N-1]+u[N-2])/(2*hxd)

    strain_rate=np.zeros(N+1)
    strain_rate[0]=(-3*dudt[0]+4*dudt[1]-dudt[2])/(2*hxd)
    strain_rate[1:N]=(dudt[2:N+1]-dudt[0:N-1])/(2*hxd)  # central difference
    strain_rate[N]=(3*dudt[N]-4*dudt[N-1]+dudt[N-2])/(2*hxd)

    sigma_actual=Emat*strain+mumat*strain_rate

    dsigmadt=sigma_actual-sigma  # store change
    
    fval=dudt, dsigmadt
    
    return fval
