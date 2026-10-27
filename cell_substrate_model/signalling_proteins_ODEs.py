import numpy as np

def signalling_proteins_ODEs(y,N,params):

    # -------------------------------
    # current biochemistry and deformation
    # -------------------------------
    
    nb1d=y[8*N+8:9*N+9]
    nb2d=y[9*N+9:10*N+10]
    nA1d=y[10*N+10:11*N+11]
    nA2d=y[11*N+11:12*N+12]
    cRd=y[14*N+14:15*N+15]
    cRAd=y[15*N+15:16*N+16]
    cMLCPd=y[16*N+16:17*N+17]
    cMLCPPd=y[17*N+17:18*N+18]
    cCd=y[18*N+18:19*N+19]
    cCPd=y[19*N+19:20*N+20]
    cMLCKd=y[20*N+20:21*N+21]
    cMLCKPd=y[21*N+21:22*N+22]
    
    # -------------------------------
    # initialise differential equations
    # -------------------------------
    
    dcRdt=np.zeros(N+1)
    dcRAdt=np.zeros(N+1)
    dMLCPdt=np.zeros(N+1)
    dMLCPPdt=np.zeros(N+1)
    dMLCKdt=np.zeros(N+1)
    dMLCKPdt=np.zeros(N+1)
    dcCdt=np.zeros(N+1)
    dcCPdt=np.zeros(N+1)
    
    # -------------------------------
    # extract relevant parameters
    # -------------------------------

    KRp=params['KRp']
    delta=params['delta']
    kRmt=params['kRmt']
    tchar=params['tchar']
    K1p=params['K1p']
    K2p=params['K2p']
    K3p=params['K3p']
    k1mt=params['k1mt']
    k2mt=params['k2mt']
    k3mt=params['k3mt']
    DPt=params['DPt']
    DPPt=params['DPPt']
    DKt=params['DKt']
    DKPt=params['DKPt']
    DCt=params['DCt']
    DCPt=params['DCPt']
    hxd=params['hxd']
    
    # -------------------------------
    # update ROCK reaction rate
    # -------------------------------
    
    nbtd=nb1d+nb2d # total bound integrins
    nAtd=nA1d+nA2d # total focal adhesions
    kRp=KRp*(nbtd+delta*nAtd) # dimensional ROCK activation
    kRpt=kRp*tchar # dimensionless ROCK activation
    
    # -------------------------------
    # ROCK
    # -------------------------------
    
    dcRdt=-kRpt*cRd+kRmt*cRAd
    dcRAdt=kRpt*cRd-kRmt*cRAd
    
    # -------------------------------
    # update ROCK dependent reaction rates (dimensionless)
    # -------------------------------
    
    k1pt=K1p*cRAd*tchar
    k2pt=K2p*cRAd*tchar
    k3pt=K3p*cRAd*tchar

    # -------------------------------
    # MLCP and phosphorylated MLCP (MLCP-P)
    # -------------------------------
    
    dMLCPdt[0]=-k1pt[0]*cMLCPd[0]+k1mt*cMLCPPd[0]+2*DPt*(cMLCPd[1]-cMLCPd[0])/hxd**2
    dMLCPPdt[0]=k1pt[0]*cMLCPd[0]-k1mt*cMLCPPd[0]+2*DPPt*(cMLCPPd[1]-cMLCPPd[0])/hxd**2
    dMLCPdt[1:N]=-k1pt[1:N]*cMLCPd[1:N]+k1mt*cMLCPPd[1:N] +DPt*(cMLCPd[2:N+1]-2*cMLCPd[1:N]+cMLCPd[0:N-1])/hxd**2
    dMLCPPdt[1:N]=k1pt[1:N]*cMLCPd[1:N]-k1mt*cMLCPPd[1:N]+DPPt*(cMLCPPd[2:N+1]-2*cMLCPPd[1:N]+cMLCPPd[0:N-1])/hxd**2
    dMLCPdt[N]=-k1pt[N]*cMLCPd[N]+k1mt*cMLCPPd[N]+2*DPt*(cMLCPd[N-1]-cMLCPd[N])/hxd**2
    dMLCPPdt[N]=k1pt[N]*cMLCPd[N]-k1mt*cMLCPPd[N]+2*DPPt*(cMLCPPd[N-1]-cMLCPPd[N])/hxd**2
    
    # -------------------------------
    # MLCK and phosphorylated MLCK (MLCK-P)
    # -------------------------------
    
    dMLCKdt[0]=-k2pt[0]*cMLCKd[0]+k2mt*cMLCKPd[0]+2*DKt*(cMLCKd[1]-cMLCKd[0])/hxd**2
    dMLCKPdt[0]=k2pt[0]*cMLCKd[0]-k2mt*cMLCKPd[0]+2*DKPt*(cMLCKPd[1]-cMLCKPd[0])/hxd**2
    dMLCKdt[1:N]=-k2pt[1:N]*cMLCKd[1:N]+k2mt*cMLCKPd[1:N]+DKt*(cMLCKd[2:N+1]-2*cMLCKd[1:N]+cMLCKd[0:N-1])/hxd**2
    dMLCKPdt[1:N]=k2pt[1:N]*cMLCKd[1:N]-k2mt*cMLCKPd[1:N]+DKPt*(cMLCKPd[2:N+1]-2*cMLCKPd[1:N]+cMLCKPd[0:N-1])/hxd**2
    dMLCKdt[N]=-k2pt[N]*cMLCKd[N]+k2mt*cMLCKPd[N]+2*DKt*(cMLCKd[N-1]-cMLCKd[N])/hxd**2
    dMLCKPdt[N]=k2pt[N]*cMLCKd[N]-k2mt*cMLCKPd[N]+2*DKPt*(cMLCKPd[N-1]-cMLCKPd[N])/hxd**2
    
    # -------------------------------
    # cofilin and phosphorylated cofilin (C-P)
    # -------------------------------
                  
    dcCdt[0]=-k3pt[0]*cCd[0]+k3mt*cCPd[0]+ 2*DCt*(cCd[1]-cCd[0])/hxd**2
    dcCPdt[0]=k3pt[0]*cCd[0]-k3mt*cCPd[0]+2*DCPt*(cCPd[1]-cCPd[0])/hxd**2
    dcCdt[1:N]=-k3pt[1:N]*cCd[1:N]+k3mt*cCPd[1:N]+DCt*(cCd[2:N+1]-2*cCd[1:N]+cCd[0:N-1])/hxd**2
    dcCPdt[1:N]=k3pt[1:N]*cCd[1:N]-k3mt*cCPd[1:N]+DCPt*(cCPd[2:N+1]-2*cCPd[1:N]+cCPd[0:N-1])/hxd**2
    dcCdt[N]=-k3pt[N]*cCd[N]+k3mt*cCPd[N]+2*DCt*(cCd[N-1]-cCd[N])/hxd**2
    dcCPdt[N]=k3pt[N]*cCd[N]-k3mt*cCPd[N]+2*DCPt*(cCPd[N-1]-cCPd[N])/hxd**2
    
    
    return dcRdt, dcRAdt, dMLCPdt, dMLCPPdt, dMLCKdt, dMLCKPdt, dcCdt, dcCPdt
