import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def mechanics_ODEs(y,N,params):
    
    # -------------------------------
    # current biochemistry and deformation
    # -------------------------------
    
    cF1d=y[N+1:2*N+2]
    cF2d=y[2*N+2:3*N+3]
    cSpd=y[3*N+3:4*N+4]
    nb1d=y[8*N+8:9*N+9]
    nb2d=y[9*N+9:10*N+10]
    nA1d=y[10*N+10:11*N+11]
    nA2d=y[11*N+11:12*N+12]
    ud=y[22*N+22:23*N+25]   
    wd=y[23*N+25:24*N+28]  
    sigma_cell=y[24*N+28:25*N+29]
    sigma_ECM=y[25*N+29:26*N+30]
    
    # -------------------------------
    # initalise differential equations
    # -------------------------------
    
    duddt=np.zeros(N+3)
    dwddt=np.zeros(N+3)
    
    # -------------------------------
    # extract relevant parameters
    # -------------------------------
    
    ElS0d=params['ElS0d']
    ElP0d=params['ElP0d']
    ElC0d=params['ElC0d']
    ElO0d=params['ElO0d']
    ElB0d=params['ElB0d']
    ElA0d=params['ElA0d']
    Fd=params['Fd']
    mu0d=params['mu0d']
    v0d=params['v0d']
    g1d=params['g1d']
    hxd=params['hxd']
    alpha=params['alpha']
    beta=params['beta']
    fPd=params['fPd']
    fCd=params['fCd']
    fOd=params['fOd']
    bB0d=params['bB0d']
    bA0d=params['bA0d']
    muECMd=params['muECMd']
    F1d=params['F1d']
    F2d=params['F2d']
    
    # -------------------------------
    # update biochemistry dependent mechanics
    # ------------------------------- 
     
    fFd=cF1d+cF2d # total actin filaments
    fSd=cSpd # total stress fibres
    fBd=nb1d+nb2d # total bound integrins
    fAd=nA1d+nA2d # total focal adhesions
      
    Eldcell=fFd+ElS0d*fSd+ElP0d*fPd # Young's modulus (cell cytoplasm)
    EldECM=ElC0d*fCd+ElO0d*fOd # Young's modulus (ECM)
    Eldadh=ElB0d*fBd+ElA0d*fAd # cell-substrate adhesion stiffness
    bBadhd=bB0d*fBd+bA0d*fAd+1e-8*np.ones(N+1) # cell-substrate adhesion drag
    hd=cSpd # stress dependence on stress fibres 
    mueffd=mu0d*g1d+Fd*hd/v0d # effective viscosity of cell cytoplasm
    
    # -------------------------------
    # Construct linear system M*[du/dt,dw/dt]=F, accounts for substrate drag, viscosity, etc.
    # ------------------------------- 
     
    # Right hand side (forcing) vector, F
    
    F=np.zeros(2*N+6)
     
    # cell
    F[0]=-Eldcell[0]*ud[2]/(2*hxd)+Eldcell[0]*ud[0]/(2*hxd)-Fd*hd[0]-F1d # BC
    F[1]=((-Eldcell[0]/hxd**2-(-3*Eldcell[0]+4*Eldcell[1]-Eldcell[2])/(4*hxd**2))*ud[2]
         +(-Eldcell[0]/hxd**2+(-3*Eldcell[0]+4*Eldcell[1]-Eldcell[2])/(4*hxd**2))*ud[0]
         +(2*Eldcell[0]/hxd**2+Eldadh[0])*ud[1]-Eldadh[0]*wd[1]-Fd*(-3*hd[0]+4*hd[1]-hd[2])/(2*hxd))
    F[2:N+1]=((-(Eldcell[2:N+1]-Eldcell[0:N-1])/(4*hxd**2)-Eldcell[1:N]/hxd**2)*ud[3:N+2]
         +((Eldcell[2:N+1]-Eldcell[0:N-1])/(4*hxd**2)-Eldcell[1:N]/hxd**2)*ud[1:N]
         +(2*Eldcell[1:N]/hxd**2+Eldadh[1:N])*ud[2:N+1]-Eldadh[1:N]*wd[2:N+1]-Fd*(hd[2:N+1]-hd[0:N-1])/(2*hxd))
    F[N+1]=((-Eldcell[N]/hxd**2-(3*Eldcell[N]-4*Eldcell[N-1]+Eldcell[N-2])/(4*hxd**2))*ud[N+2]
         +(-Eldcell[N]/hxd**2+(3*Eldcell[N]-4*Eldcell[N-1]+Eldcell[N-2])/(4*hxd**2))*ud[N]
         +(2*Eldcell[N]/hxd**2+Eldadh[N])*ud[N+1]-Eldadh[N]*wd[N+1]-Fd*(3*hd[N]-4*hd[N-1]+hd[N-2])/(2*hxd)) 
    F[N+2]=(-Eldcell[N]*ud[N+2]/(2*hxd)+Eldcell[N]*ud[N]/(2*hxd)-Fd*hd[N]+F2d) # BC
     
    # ECM
    F[N+3]=EldECM[0]*wd[1]/alpha-EldECM[0]*wd[2]/(2*hxd)+EldECM[0]*wd[0]/(2*hxd) # BC
    F[N+4]=((-EldECM[0]/hxd**2-(-3*EldECM[0]+4*EldECM[1]-EldECM[2])/(4*hxd**2))*wd[2]
         +(-EldECM[0]/hxd**2+(-3*EldECM[0]+4*EldECM[1]-EldECM[2])/(4*hxd**2))*wd[0]+(2*EldECM[0]/hxd**2+Eldadh[0])*wd[1]-Eldadh[0]*ud[1])
    F[N+5:2*N+4]=((-(EldECM[2:N+1]-EldECM[0:N-1])/(4*hxd**2)-EldECM[1:N]/hxd**2)*wd[3:N+2]
         +((EldECM[2:N+1]-EldECM[0:N-1])/(4*hxd**2)-EldECM[1:N]/hxd**2)*wd[1:N]
         +(2*EldECM[1:N]/hxd**2+Eldadh[1:N])*wd[2:N+1]-Eldadh[1:N]*ud[2:N+1])
    F[2*N+4]=((-EldECM[N]/hxd**2-(3*EldECM[N]-4*EldECM[N-1]+EldECM[N-2])/(4*hxd**2))*wd[N+2]
         +(-EldECM[N]/hxd**2+(3*EldECM[N]-4*EldECM[N-1]+EldECM[N-2])/(4*hxd**2))*wd[N]
         +(2*EldECM[N]/hxd**2+Eldadh[N])*wd[N+1]-Eldadh[N]*ud[N+1])  
    F[2*N+5]=(-EldECM[N]*wd[N+1]/beta-EldECM[N]*wd[N+2]/(2*hxd)+EldECM[N]*wd[N]/(2*hxd)) # BC
     
     
    # Mass matrix, M
    M=lil_matrix((2*N+6,2*N+6))
     
    # cell BC rows
    M[0,0]=-mueffd[0]/(2*hxd)
    M[0,2]=mueffd[0]/(2*hxd)
     
    M[1,0]=mueffd[0]/hxd**2-(-3*mueffd[0]+4*mueffd[1]-mueffd[2])/(4*hxd**2)
    M[1,1]=-2*mueffd[0]/hxd**2-bBadhd[0]
    M[1,2]=mueffd[0]/hxd**2+(-3*mueffd[0]+4*mueffd[1]-mueffd[2])/(4*hxd**2)
    M[1,N+4]=bBadhd[0]
     
    # ECM BC rows
    M[N+3,N+3]=-muECMd[0]/(2*hxd)
    M[N+3,N+4]=-muECMd[0]/alpha
    M[N+3,N+5]=muECMd[0]/(2*hxd)
     
    M[N+4,N+3]=muECMd[0]/hxd**2-(-3*muECMd[0]+4*muECMd[1]-muECMd[2])/(4*hxd**2)
    M[N+4,N+4]=-2*muECMd[0]/hxd**2-bBadhd[0]
    M[N+4,N+5]=muECMd[0]/hxd**2+(-3*muECMd[0]+4*muECMd[1]-muECMd[2])/(4*hxd**2)
    M[N+4,1]=bBadhd[0]
     
    # interior rows
    for i in range(2,N+1):
         M[i,i-1]=-(mueffd[i]-mueffd[i-2])/(4*hxd**2)+mueffd[i-1]/hxd**2
         M[i,i]=-2*mueffd[i-1]/hxd**2-bBadhd[i-1]
         M[i,i+1]=(mueffd[i]-mueffd[i-2])/(4*hxd**2)+mueffd[i-1]/hxd**2
         M[i,i+N+3]=bBadhd[i-1]
     
         M[N+3+i,N+2+i]=-(muECMd[i]-muECMd[i-2])/(4*hxd**2)+muECMd[i-1]/hxd**2
         M[N+3+i,N+3+i]=-2*muECMd[i-1]/hxd**2-bBadhd[i-1]
         M[N+3+i,N+4+i]=(muECMd[i]-muECMd[i-2])/(4*hxd**2)+muECMd[i-1]/hxd**2
         M[N+3+i,i]=bBadhd[i-1]
     
    # final BC rows
    M[N+1,N]=mueffd[N]/hxd**2-(3*mueffd[N]-4*mueffd[N-1]+mueffd[N-2])/(4*hxd**2)
    M[N+1,N+1]=-2*mueffd[N]/hxd**2-bBadhd[N]
    M[N+1,N+2]=mueffd[N]/hxd**2+(3*mueffd[N]-4*mueffd[N-1]+mueffd[N-2])/(4*hxd**2)
    M[N+1,2*N+4]=bBadhd[N]
     
    M[N+2,N]=-mueffd[N]/(2*hxd)
    M[N+2,N+2]=mueffd[N]/(2*hxd)
     
    M[2*N+4,2*N+3]=muECMd[N]/hxd**2-(3*muECMd[N]-4*muECMd[N-1]+muECMd[N-2])/(4*hxd**2)
    M[2*N+4,2*N+4]=-2*muECMd[N]/hxd**2-bBadhd[N]
    M[2*N+4,2*N+5]=muECMd[N]/hxd**2+(3*muECMd[N]-4*muECMd[N-1]+muECMd[N-2])/(4*hxd**2)
    M[2*N+4,N+1]=bBadhd[N]
     
    M[2*N+5,2*N+3]=-muECMd[N]/(2*hxd)
    M[2*N+5,2*N+4]=muECMd[N]/beta
    M[2*N+5,2*N+5]=muECMd[N]/(2*hxd)
    
    Fout=spsolve(M.tocsc(),F)
     
    duddt=Fout[0:N+3]
    dwddt=Fout[N+3:2*N+6]
    
    # -------------------------------
    # Compute KV stress (cell and ECM)
    # -------------------------------
    
    strain_cell=np.zeros(N+1)
    strain_cell[0]=(-3*ud[1]+4*ud[2]-ud[3])/(2*hxd)
    strain_cell[1:N]=(ud[3:N+2]-ud[1:N])/(2*hxd)  # central difference
    strain_cell[N]=(3*ud[N+1]-4*ud[N]+ud[N-1])/(2*hxd)

    strain_rate_cell=np.zeros(N+1)
    strain_rate_cell[0]=(-3*duddt[1]+4*duddt[2]-duddt[3])/(2*hxd)
    strain_rate_cell[1:N]=(duddt[3:N+2]-duddt[1:N])/(2*hxd)  # central difference
    strain_rate_cell[N]=(3*duddt[N+1]-4*duddt[N]+duddt[N-1])/(2*hxd)

    sigma_actual_cell=Eldcell*strain_cell+mueffd*strain_rate_cell+Fd*hd

    dsigma_celldt=sigma_actual_cell-sigma_cell # store change
    
    strain_ECM=np.zeros(N+1)
    strain_ECM[0]=(-3*wd[1]+4*wd[2]-wd[3])/(2*hxd)
    strain_ECM[1:N]=(wd[3:N+2]-wd[1:N])/(2*hxd)  # central difference
    strain_ECM[N]=(3*wd[N+1]-4*wd[N]+wd[N-1])/(2*hxd)

    strain_rate_ECM=np.zeros(N+1)
    strain_rate_ECM[0]=(-3*dwddt[1]+4*dwddt[2]-dwddt[3])/(2*hxd)
    strain_rate_ECM[1:N]=(dwddt[3:N+2]-dwddt[1:N])/(2*hxd)  # central difference
    strain_rate_ECM[N]=(3*dwddt[N+1]-4*dwddt[N]+dwddt[N-1])/(2*hxd)

    sigma_actual_ECM=EldECM*strain_ECM+muECMd*strain_rate_ECM

    dsigma_ECMdt=sigma_actual_ECM-sigma_ECM # store change
    
    return duddt, dwddt, dsigma_celldt, dsigma_ECMdt


