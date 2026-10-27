import numpy as np
from overall_differential_equations import overall_differential_equations
from scipy.integrate import solve_ivp
from SLS_Maxwell_differential_equations import SLS_Maxwell_differential_equations # Standard Linear Solid (Maxwell)
from SLS_Kelvin_differential_equations import SLS_Kelvin_differential_equations # Standard Linear Solid (Maxwell)
from KV_differential_equations import KV_differential_equations # Kelvin-Voigt
from Maxwell_differential_equations import Maxwell_differential_equations # Maxwell

# driver code for simple example model

# -------------------------------
# choose mechanical model
# -------------------------------

mechanics_model="Maxwell" # "KV" or "Maxwell" or "SLS_Maxwell" or "SLS_Kelvin"

if mechanics_model=="SLS_Maxwell":
    mech_rhs=SLS_Maxwell_differential_equations
elif mechanics_model=="SLS_Kelvin":
    mech_rhs=SLS_Kelvin_differential_equations
elif mechanics_model=="KV":
    mech_rhs=KV_differential_equations
elif mechanics_model=="Maxwell":
    mech_rhs=Maxwell_differential_equations
else:
    raise ValueError("Unknown mechanical model")

# -------------------------------
# domain and discretisation
# -------------------------------

L=1 # domain length
x0=0 # left boundary of the spatial domain (x=0)
x1=L # right boundary of the spatial domain (x=L)
N=100 # number of spatial steps
xd=np.linspace(x0,x1,N+1) # domain nodes
hxd=xd[1]-xd[0] # uniform grid spacing

t0=0 # simulation start time
t1=20 # simulation end time
Nt=10 # break solver into smaller substeps

# -------------------------------
# initial conditions 
# -------------------------------

ca0=1*np.ones(N+1) # chemical A
cb0=np.zeros(N+1) # chemical B
u0=np.zeros(N+1) # zero displacement 
sigma0=np.zeros(N+1) # zero stress

# -------------------------------
# dimensionless parameters
# -------------------------------

Kfp=10 # rate of A-->B
Kfm=1 # rate of B-->A
Da=1 # diffusivity of chemical A
Db=1 # diffusitivity of chemical B
mu0=0.5 # viscosity multiplier
E0=1 # Young's modulus multiplier
beta0=1 # drag multiplier

amp=0.25 # maximum amplitude pull-hold
t_pull=5.0 

# Driver code
# -------------------------------
# Pull-and-hold boundary displacement
# -------------------------------

def urb(t): # right boundary displacement
    if t<t_pull:
        return amp*t/t_pull
    else:
        return amp

def durbdt(t): # rate of change of right boundary displacement
    if t<t_pull:
        return amp/t_pull
    else:
        return 0.0

# -------------------------------
# Initial arrays and solver setup
# -------------------------------

tspan=np.linspace(t0,t1,Nt+1) # time points to output 

y0d=np.concatenate((ca0,cb0,u0,sigma0)) # initial condition vector for solver [ca0, cb0, u0, sigma0]

# storing output
Tsol=[] # time
Ysol=[] # [ca,cb,u,sigma]

# Pack all model parameters into a dictionary
params=dict(Kfp=Kfp,Kfm=Kfm,Da=Da,Db=Db,mu0=mu0,E0=E0,beta0=beta0,hxd=hxd,urb=urb,durbdt=durbdt,N=N)

# -------------------------------
# Run solver for the simple model
# -------------------------------

# Loop over consecutive tspan segments, each segment goes from t_start to t_end

for j in range(len(tspan)-1):
    t_start=tspan[j]
    t_end=tspan[j+1]

    t_eval=np.linspace(t_start,t_end,1000) # Generate finer internal time points

    # Solve the ODE system for concentrations ca, cb, and material displacement u
    sol=solve_ivp(fun=lambda t, y: overall_differential_equations(t, y, N, params, mech_rhs),
        t_span=(t_start, t_end), y0=y0d, method="BDF", t_eval=t_eval, rtol=1e-3, atol=1e-6)

    # Append the solver output to the lists
    Tsol.append(sol.t)
    Ysol.append(sol.y.T)

    # Update initial condition for next segment
    y0d=sol.y[:,-1]

# -------------------------------
# Store solution for post processing
# -------------------------------

# Concatenate all segments into full solution arrays
Tall=np.concatenate(Tsol) # full time vector
Yall=np.vstack(Ysol) # full solution array 

# Save the full solution to disk
np.save(f'Tall_{mechanics_model}.npy',Tall)
np.save(f'Yall_{mechanics_model}.npy',Yall)
np.save('xd.npy',xd)



