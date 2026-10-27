import numpy as np
from scipy.integrate import simpson
from cell_ECM_differential_equations import cell_ECM_differential_equations
from scipy.integrate import solve_ivp

# -------------------------------
# domain and discretisation
# -------------------------------

L=25.0  # cell length
x0=-L/2 # left boundary of the spatial domain 
x1=L/2 # right boundary of the spatial domain
N=100 # number of spatial steps
xn=np.linspace(x0,x1,N+1) # domain nodes

A=L*L # cross-sectional area parallel to substrate
W=A/L # width of cell
H=2.0 # height of cell
S=H*W # cross-sectional area orthogonal to substrate

x0d=x0/L # dimensionless left boundary
x1d=x1/L # dimensionless right boundary
xd=np.linspace(x0d,x1d,N+1) # dimensionless domain nodes
hxd=xd[1]-xd[0] # uniform grid spacing

alpha=1.0 # ECM dimensionless far-field length (left)
beta=alpha # ECM dimensionless far-field length (right)
L1=alpha*L # dimensional far-field length (left)
L2=beta*L # dimensional far-field length (right)

t0=0 # simulation start time
t1=10000 # simulation end time
Nt=10 # break solver into Nt smaller substeps

# -------------------------------
# dimensional biochemistry initial conditions and parameters
# -------------------------------

# scaffolding proteins ICs
cG0=100*np.ones(N+1) # actin monomers
cF10=np.zeros(N+1) # actin filaments (family 1)
cF20=np.zeros(N+1) # actin filaments (family 2)
cSp0=np.zeros(N+1) # stress fibres
cM0=30*np.ones(N+1) # inactivated myosin ii
cMp0=np.zeros(N+1) # activated myosin ii
nf0=100*np.ones(N+1) # free integrins
nh0=np.zeros(N+1) # high-affinity integrins
nb10=np.zeros(N+1) # bound integrins (family 1)
nb20=np.zeros(N+1) # bound integrins (family 2)
nA10=np.zeros(N+1) # focal adhesions (family 1)
nA20=np.zeros(N+1) # focal adhesions (family 2))

# Ligand patterning ICs
NS0=500
ns10=NS0*(np.tanh(-10*xd)+1) # family 1 ligand profile
ns20=NS0*(np.tanh(10*xd)+1) # family 2 ligand profile

# signalling proteins ICs
cR0=1e-3*np.ones(N+1) # inactivated ROCK
cRA0=np.zeros(N+1) # activated ROCK 
cMLCP0=1*np.ones(N+1) # unphosphorylated MLCP
cMLCPP0=np.zeros(N+1) # phosphorylated MLCP
cC0=1*np.ones(N+1) # unphosphorylated cofilin
cCP0=np.zeros(N+1) # phosphorylated cofilin
cMLCK0=1e-1*np.ones(N+1) # unphosphorylated MLCK
cMLCKP0=np.zeros(N+1) # phosphorylated MLCK

# Rate/diffusion constants
Kpp, Kpm, Kap, Kam, KFp, KFm=2e-3, 1e-2, 1e-2, 1e-2, 5e1, 1e-3 # actin polymerisation/depolymerisation, myosin ii activation/inactivation, adhesion maturation/disassembly
kmp, kmm, khp, khm, kbp, kbm=1e-4, 1e-2, 0.5, 5, 1e-4, 1e-2 # myosin ii cross-linking/disassembly, integrin activation/inactivation, integrin-ligand binding/unbinding
DG, DF, DM, DMP, Dnf=10, 0.3, 1, 1, 0.1 # diffusivity of actin monomers, actin filaments, inactivated myosin ii, activated myosin ii, free integrins

KRp, kRm=1e-2, 1e-1 # ROCK activation/inactivation
K1p, k1m=1e-2, 1e-2 # MLCP phosphorylation/dephosphorylation
K2p, k2m=1e-2, 1e-2 # cofilin phosphorylation/dephosphorylation
K3p, k3m=5e-2, 1e-2 # MLCK phosphorylation/dephosphorylation
DP, DPP=15, 15 # MLCP diffusivity
DK, DKP=1, 1 # MLCK diffusivity 
DC, DCP=10, 10 # cofilin diffusivity

v0=1 # stall velocity
delta=4 # feedback strength focal adhesion formation

# -------------------------------
# dimensional scales
# ------------------------------- 

cAtyp=(1/L)*simpson(cG0+cF10+cF20+cSp0,xn) # actin scale
nItyp=(1/L)*simpson(nf0+nh0+nb10+nb20+nA10+nA20,xn) # integrin/adhesion scales
nStyp=(1/L)*simpson(ns10+ns20,xn) # ligand scale
cMtyp=(1/L)*simpson(cM0+cMp0,xn) # myosin ii scale
cRtyp=(1/L)*simpson(cR0+cRA0,xn) # ROCK scale
cMLCPtyp=(1/L)*simpson(cMLCP0+cMLCPP0,xn) # MLCP scale 
cCtyp= (1/L)*simpson(cC0+cCP0,xn) # cofilin scale
cMLCKtyp=(1/L)*simpson(cMLCK0+cMLCKP0,xn) # MLCK scale

tchar=1/nStyp/kbp # time scale
vchar=L/tchar # velocity scale 

# -------------------------------
# dimensional biochemistry initial conditions and parameters
# ------------------------------- 

# Dimensionless initial conditions
cG0d=cG0/cAtyp
cF10d=cF10/cAtyp
cF20d=cF20/cAtyp
cSp0d=cSp0/cAtyp
cM0d=cM0/cMtyp
cMp0d=cMp0/cMtyp
nf0d=nf0/nItyp
nh0d=nh0/nItyp
nb10d=nb10/nItyp
nb20d=nb20/nItyp
nA10d=nA10/nItyp
nA20d=nA20/nItyp
ns10d=ns10/nStyp
ns20d=ns20/nStyp
cR0d=cR0/cRtyp
cRA0d=cRA0/cRtyp
cMLCP0d=cMLCP0/cMLCPtyp
cMLCPP0d=cMLCPP0/cMLCPtyp
cMLCK0d=cMLCK0/cMLCKtyp
cMLCKP0d=cMLCKP0/cMLCKtyp
cC0d=cC0/cCtyp
cCP0d=cCP0/cCtyp

# Dimensionless parameters (see e.g. McNicol et al. (2025), J. Theor. Biol.)
kmpMAt=kmp*cMtyp*cAtyp*tchar
kmpAAt=kmp*cAtyp*cAtyp*tchar
kmmt=kmm*tchar
kmmAdMt=kmm*cAtyp*tchar/cMtyp
khmt=khm*tchar
kbpNt=kbp*tchar*nItyp
kbpSt=kbp*tchar*nStyp
kbmt=kbm*tchar
kbmNdSt=kbm*nItyp*tchar/nStyp
khpt=khp*tchar
DGt=DG*tchar/L**2
DFt=DF*tchar/L**2
DMt=DM*tchar/L**2
DMPt=DMP*tchar/L**2
Dnft=Dnf*tchar/L**2

kRmt=kRm*tchar
k1mt=k1m*tchar
k2mt=k2m*tchar
k3mt=k3m*tchar
DPt=DP*tchar/L**2
DPPt=DPP*tchar/L**2
DKt=DK*tchar/L**2
DKPt=DKP*tchar/L**2
DCt=DC*tchar/L**2
DCPt=DCP*tchar/L**2

v0d=v0*tchar/L # dimensionless stall velocity

# -------------------------------
# actin treadmilling 
# ------------------------------- 

U0=0.1
U1=(np.tanh(8*(xd-x0d))-np.tanh(8*(xd-x1d)))-1 # family 1 (dimensional)
U2=1-(np.tanh(8*(xd-x0d))-np.tanh(8*(xd-x1d))) # family 2 (dimensional)

U0sd=U0/vchar
U1d=U0sd*U1 # family 1 (dimensionless)
U2d=U0sd*U2 # family 2 (dimensionless)

# -------------------------------
# dimensional mechanics
# ------------------------------- 

u0=np.zeros(N+3) # initial cell displacement
w0=np.zeros(N+3) # initial ECM displacement
sigma_cell0=np.zeros(N+1) # initial cell stress 
sigma_ECM0=np.zeros(N+1) # initial ECM stress

ElF0=1000e-3 # Young's modulus (actin filaments)
ElS0=5000e-3 # Young's modulus (stress fibres)
ElP0=5000e-3 # Young's modulus (nucleus/microtubules/intermediate filaments)
mu0=100e-3 # cytosol viscosity

ElB0=0.01 # bound integrin stiffness 
ElA0=0.05 # focal adhesion stiffness 
bB0=1e-3 # bound integrin drag
bA0=5e-3 # focal adhesion drag 

ElC0=1000 # Young's modulus (ECM collagen)
ElO0=0 # Young's modulus (ECM other components)
muE0=1e-3 # ECM viscosity

Fmax=1000e-3 # myosin ii generated contraction stresses 
F1=-0.0 # applied stress (left boundary)
F2=0.0 # applied stress (right boundary)

# -------------------------------
# dimensionless mechanics 
# ------------------------------- 

u0d=u0/L # initial cell displacement (dimensionless)
w0d=w0/L # initial ECM displacement (dimensionless)
sigma_cell0d=sigma_cell0/ElF0 # initial cell stress (dimensionless)
sigma_ECM0d=sigma_ECM0/ElF0 # initial ECM stress (dimensionless)

ElS0d=ElS0/ElF0 # scale all stresses on E0F
ElP0d=ElP0/ElF0
mu0d=mu0/ElF0/tchar

ElB0d=ElB0*L**2/ElF0
ElA0d=ElA0*L**2/ElF0
bB0d=bB0*L**2/ElF0/tchar
bA0d=bA0*L**2/ElF0/tchar

ElC0d=ElC0/ElF0
ElO0d=ElO0/ElF0
muE0d=muE0/ElF0/tchar

Fd=Fmax/ElF0
F1d=F1/ElF0
F2d=F2/ElF0

fFd=cF10d+cF20d # Young's modulus actin filaments depends on filament concentration
fSd=cSp0d # Young's modulus stress fibres dependes on stres fibre concentration
fPd=0.9*np.cos(np.pi*xd)**6+0.1 # Young's modulus background (nucleus, microtubules, intermediate filaments)
g1d=np.ones(N+1) # constant background viscosity cytosol
fCd=np.ones(N+1) # constant ECM collagen Young's modulus 
fOd=np.ones(N+1) # constant ECM other proteins Young's modulus 
g2d=np.ones(N+1) # constant ECM viscosity

fBd=nb10d+nb20d # bound integrin adhesion stiffness depends on bound integrin density
fAd=nA10d+nA20d # focal adhesion stiffness depends on focal adhesion density
hd=cSp0d # contractile stresses depends on stress fibre concentration

Eldcell=fFd+ElS0d*fSd+ElP0d*fPd # Young's modulus overall (cell cytoplasm)
EldECM=ElC0d*fCd+ElO0d*fOd # Young's modulus overall (ECM)
mueffd=mu0d*g1d+Fd*hd/v0d # viscosity overall (cell cytoplasm)
muECMd=muE0d*g2d # viscosity overall (ECM)
Eldadh=ElB0d*fBd+ElA0d*fAd # total adhesion stiffness 
bBadhd=bB0d*fBd+bA0d*fAd+1e-8*np.ones(N+1) # total adhesion drag

# -------------------------------
# solver set-up
# ------------------------------- 

lambda_arr=np.zeros(N+1) # stretch adhesions
kfa=lambda_arr

y0d=np.concatenate((cG0d,cF10d,cF20d,cSp0d,cM0d,cMp0d,nf0d,nh0d,nb10d,nb20d,nA10d,nA20d,ns10d,ns20d,
cR0d,cRA0d,cMLCP0d,cMLCPP0d,cMLCK0d,cMLCKP0d,cC0d,cCP0d,u0d,w0d,sigma_cell0d,sigma_ECM0d)) # concatenated ICs

tspan=np.linspace(t0,t1,Nt+1)/tchar # dimensionless time domain

outud=np.zeros((len(tspan),N+3))
outwd=np.zeros((len(tspan),N+3))

outud[0,:]=u0d
outwd[0,:]=w0d

# storing output
Tsol=[] # time
Ysol=[] # [ca,cb,u,sigma]

xcurrentcell=np.zeros((len(tspan),N+1)) # track current cell position
xcurrentECM=np.zeros((len(tspan),N+1)) # track current ECM position

xcurrentcell[0,:]=xd
xcurrentECM[0,:]=xd


# Pack all model parameters into a dictionary
params=dict(KRp=KRp,delta=delta,kRmt=kRmt,tchar=tchar,K1p=K1p,K2p=K2p,K3p=K3p,k1mt=k1mt,k2mt=k2mt,k3mt=k3mt,Kpp=Kpp,
    Kpm=Kpm,Kap=Kap,Kam=Kam,KFp=KFp,KFm=KFm,nItyp=nItyp,kmmt=kmmt,kmpMAt=kmpMAt,kmpAAt=kmpAAt,kmmAdMt=kmmAdMt,
    kbpSt=kbpSt,kbmt=kbmt,khpt=khpt,khmt=khmt,kbpNt=kbpNt,kbmNdSt=kbmNdSt,
    DGt=DGt,DFt=DFt,DMt=DMt,DMPt=DMPt,DPt=DPt,DPPt=DPPt,DKt=DKt,DKPt=DKPt,DCt=DCt,DCPt=DCPt,Dnft=Dnft,
    ElS0d=ElS0d,ElP0d=ElP0d,ElC0d=ElC0d,ElO0d=ElO0d,ElB0d=ElB0d,ElA0d=ElA0d,Fd=Fd,mu0d=mu0d,v0d=v0d,g1d=g1d,hxd=hxd,
    alpha=alpha,beta=beta,fPd=fPd,fCd=fCd,fOd=fOd,bB0d=bB0d,bA0d=bA0d,muECMd=muECMd,F1d=F1d,F2d=F2d,U1d=U1d,U2d=U2d,N=N)

# -------------------------------
# Run solver for the cell-substrate interaction model
# -------------------------------

# Loop over consecutive tspan segments, each segment goes from t_start to t_end

for j in range(len(tspan)-1):

    t_start=tspan[j]
    t_end=tspan[j+1]

    t_eval=np.linspace(t_start,t_end,1000) 

    # Solve the ODE system for concentrations ca, cb, and material displacement u
    sol=solve_ivp(fun=lambda t, y: cell_ECM_differential_equations(t,y,N,params),
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

np.save('Tsol_cell_substrate_interaction',Tall)
np.save('Ysol_cell_substrate_interaction',Yall)
np.save('xd.npy',xd)