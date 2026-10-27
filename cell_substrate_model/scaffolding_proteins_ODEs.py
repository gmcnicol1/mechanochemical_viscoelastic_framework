import numpy as np

def scaffolding_proteins_ODEs(y,N,params):
    
    # -------------------------------
    # current biochemistry and deformation
    # -------------------------------
    
    cGd=y[0:N+1]
    cF1d=y[N+1:2*N+2]
    cF2d=y[2*N+2:3*N+3]
    cSpd=y[3*N+3:4*N+4]
    cMd=y[4*N+4:5*N+5]
    cMpd=y[5*N+5:6*N+6]
    nfd=y[6*N+6:7*N+7]
    nhd=y[7*N+7:8*N+8]
    nb1d=y[8*N+8:9*N+9]
    nb2d=y[9*N+9:10*N+10]
    nA1d=y[10*N+10:11*N+11]
    nA2d=y[11*N+11:12*N+12]
    ns1d=y[12*N+12:13*N+13]
    ns2d=y[13*N+13:14*N+14]
    cRAd=y[15*N+15:16*N+16]
    cMLCPd=y[16*N+16:17*N+17]
    cCd=y[18*N+18:19*N+19]
    cMLCKPd=y[21*N+21:22*N+22]
    ud=y[22*N+22:23*N+25]   
    wd=y[23*N+25:24*N+28]   
    
    # -------------------------------
    # initialise differential equations
    # -------------------------------
    
    dcGdt=np.zeros(N+1)
    dcF1dt=np.zeros(N+1)
    dcF2dt=np.zeros(N+1)
    dcSpdt=np.zeros(N+1)
    dcMdt=np.zeros(N+1)
    dcMpdt=np.zeros(N+1)
    dnfdt=np.zeros(N+1)
    dnhdt=np.zeros(N+1)
    dnb1dt=np.zeros(N+1)
    dnb2dt=np.zeros(N+1)
    dnA1dt=np.zeros(N+1)
    dnA2dt=np.zeros(N+1)
    dns1dt=np.zeros(N+1)
    dns2dt=np.zeros(N+1)
    
    # -------------------------------
    # extract relevant parameters
    # -------------------------------
    
    tchar=params['tchar']
    Kpp=params['Kpp']
    Kpm=params['Kpm']
    Kap=params['Kap']
    Kam=params['Kam']
    KFp=params['KFp']
    KFm=params['KFm']
    nItyp=params['nItyp']
    kmmt=params['kmmt']
    kmpMAt=params['kmpMAt']
    kmpAAt=params['kmpAAt']
    kmmAdMt=params['kmmAdMt']
    kbpSt=params['kbpSt']
    kbmt=params['kbmt']
    kbpNt=params['kbpNt']
    kbmNdSt=params['kbmNdSt']
    khpt=params['khpt']
    khmt=params['khmt']
    DGt=params['DGt']
    DFt=params['DFt']
    DMt=params['DMt']
    DMPt=params['DMPt']
    hxd=params['hxd']
    U1d=params['U1d']
    U2d=params['U2d']
    Dnft=params['Dnft']
    
    # -------------------------------
    # update non-constant reaction rates (dimensional)
    # -------------------------------
    
    # actin polymerisation/depolymerisation
    kpp=Kpp*cRAd
    kpm=Kpm*cCd
     
    # myosin II activation/inactivation
    kap=Kap*cMLCKPd
    kam=Kam*cMLCPd
     
    # adhesion muturation/disassembly
    lambda_adh=ud[1:N+2]-wd[1:N+2]
    kfa=0.5*lambda_adh**2
    kFp=KFp*kfa
    kFm=KFm*np.ones(N+1) 
     
    # -------------------------------
    # non-constant reaction rates (dimensionless)
    # -------------------------------
    
    kppNt=kpp*nItyp*tchar
    kpmt=kpm*tchar
    kapt=kap*tchar
    kamt=kam*tchar
    kFpt=kFp*tchar
    kFmt=kFm*tchar
    
    # -------------------------------
    # actin monomers
    # ------------------------------- 
    
    dcGdt[0]=-kppNt[0]*cGd[0]*(nb1d[0]+nb2d[0]+nA1d[0]+nA2d[0])+kpmt[0]*(cF1d[0]+cF2d[0])+kmmt*cSpd[0]+ 2*DGt*(cGd[1]-cGd[0])/hxd**2
    dcGdt[1:N]=(-kppNt[1:N]*cGd[1:N]*(nb1d[1:N]+nb2d[1:N]+nA1d[1:N]+nA2d[1:N])+kpmt[1:N]*(cF1d[1:N]+cF2d[1:N])
         +kmmt*cSpd[1:N]+DGt*(cGd[2:N+1]-2*cGd[1:N]+cGd[0:N-1])/hxd**2)
    dcGdt[N]=-kppNt[N]*cGd[N]*(nb1d[N]+nb2d[N]+nA1d[N]+nA2d[N])+kpmt[N]*(cF1d[N]+cF2d[N])+kmmt*cSpd[N]+2*DGt*(cGd[N-1]-cGd[N])/hxd**2
    
    # -------------------------------
    # actin filaments (both families)
    # -------------------------------
    
    zeta1=cF1d*U1d
    zeta2=cF2d*U2d
     
    dcF1dt[0]=(kppNt[0]*cGd[0]*(nb1d[0]+nA1d[0])-kpmt[0]*cF1d[0]-kmpMAt*(cF1d[0]*cF2d[0]*cMpd[0]+ cF1d[0]*cSpd[0]*cMpd[0]+cF1d[0]*cF1d[0]*cMpd[0])
         +2*DFt*(cF1d[1]-cF1d[0]-hxd*U1d[0]*cF1d[0]/DFt)/hxd**2-((-3*U1d[0]+4*U1d[1]-U1d[2])*cF1d[0])/(2*hxd)-U1d[0]*zeta1[0]/DFt)   
    dcF1dt[1:N]=(kppNt[1:N]*cGd[1:N]*(nb1d[1:N]+nA1d[1:N])-kpmt[1:N]*cF1d[1:N]-kmpMAt*(cF1d[1:N]*cF2d[1:N]*cMpd[1:N]+cF1d[1:N]*cSpd[1:N]*cMpd[1:N]
                   +cF1d[1:N]*cF1d[1:N]*cMpd[1:N])+DFt*(cF1d[2:N+1]-2*cF1d[1:N]+cF1d[0:N-1])/hxd**2-(zeta1[2:N+1]-zeta1[0:N-1])/(2*hxd))
    dcF1dt[N]=(kppNt[N]*cGd[N]*(nb1d[N]+nA1d[N])-kpmt[N]*cF1d[N]-kmpMAt*(cF1d[N]*cF2d[N]*cMpd[N]+cF1d[N]*cSpd[N]*cMpd[N]+cF1d[N]*cF1d[N]*cMpd[N])
         +2*DFt*(cF1d[N-1]-cF1d[N]+hxd*U1d[N]*cF1d[N]/DFt)/hxd**2-((3*U1d[N]-4*U1d[N-1]+U1d[N-2])*cF1d[N])/(2*hxd)-U1d[N]*zeta1[N]/DFt)
     
    
    dcF2dt[0]=(kppNt[0]*cGd[0]*(nb2d[0]+nA2d[0])-kpmt[0]*cF2d[0]-kmpMAt*(cF1d[0]*cF2d[0]*cMpd[0]+cF2d[0]*cSpd[0]*cMpd[0]+cF2d[0]*cF2d[0]*cMpd[0])
         +2*DFt*(cF2d[1]-cF2d[0]-hxd*U2d[0]*cF2d[0]/DFt)/hxd**2-((-3*U2d[0]+4*U2d[1]-U2d[2])*cF2d[0])/(2*hxd)-U2d[0]*zeta2[0]/DFt)    
    dcF2dt[1:N]=(kppNt[1:N]*cGd[1:N]*(nb2d[1:N]+nA2d[1:N])-kpmt[1:N]*cF2d[1:N]-kmpMAt*(cF1d[1:N]*cF2d[1:N]*cMpd[1:N]+cF2d[1:N]*cSpd[1:N]*cMpd[1:N]
                   +cF2d[1:N]*cF2d[1:N]*cMpd[1:N])+DFt*(cF2d[2:N+1]-2*cF2d[1:N]+cF2d[0:N-1])/hxd**2-(zeta2[2:N+1]-zeta2[0:N-1])/(2*hxd))
    dcF2dt[N]=(kppNt[N]*cGd[N]*(nb2d[N]+nA2d[N])-kpmt[N]*cF2d[N]-kmpMAt*(cF1d[N]*cF2d[N]*cMpd[N]+cF2d[N]*cSpd[N]*cMpd[N]+cF2d[N]*cF2d[N]*cMpd[N])
         +2*DFt*(cF2d[N-1]-cF2d[N]+hxd*U2d[N]*cF2d[N]/DFt)/hxd**2-((3*U2d[N]-4*U2d[N-1]+U2d[N-2])*cF2d[N])/(2*hxd)-U2d[N]*zeta2[N]/DFt)
     
     
   # -------------------------------
   # myosin inactivated/activated
   # -------------------------------
   
    dcMdt[0]=-kapt[0]*cMd[0]+kamt[0]*cMpd[0]+2*DMt*(cMd[1]-cMd[0])/hxd**2
    dcMdt[1:N]=-kapt[1:N]*cMd[1:N]+kamt[1:N]*cMpd[1:N]+DMt*(cMd[2:N+1]-2*cMd[1:N]+cMd[0:N-1])/hxd**2
    dcMdt[N]=-kapt[N]*cMd[N]+kamt[N]*cMpd[N]+2*DMt*(cMd[N-1]-cMd[N])/hxd**2
         
    dcMpdt[0]=(kapt[0]*cMd[0]-kamt[0]*cMpd[0]-kmpAAt*(2*cF1d[0]*cF2d[0]*cMpd[0]+cF1d[0]*cSpd[0]*cMpd[0]+cF2d[0]*cSpd[0]*cMpd[0]+cF1d[0]**2*cMpd[0] + cF2d[0]**2*cMpd[0])
                  +kmmAdMt*cSpd[0]+2*DMPt*(cMpd[1]-cMpd[0])/hxd**2)    
    dcMpdt[1:N]=(kapt[1:N]*cMd[1:N]-kamt[1:N]*cMpd[1:N]-kmpAAt*(2*cF1d[1:N]*cF2d[1:N]*cMpd[1:N]+cF1d[1:N]*cSpd[1:N]*cMpd[1:N]
                              +cF2d[1:N]*cSpd[1:N]*cMpd[1:N]+cF1d[1:N]**2*cMpd[1:N]+cF2d[1:N]**2*cMpd[1:N])
                    +kmmAdMt*cSpd[1:N]+DMPt*(cMpd[2:N+1]-2*cMpd[1:N]+cMpd[0:N-1])/hxd**2) 
    dcMpdt[N]=(kapt[N]*cMd[N]-kamt[N]*cMpd[N]-kmpAAt*(2*cF1d[N]*cF2d[N]*cMpd[N]+cF1d[N]*cSpd[N]*cMpd[N]+cF2d[N]*cSpd[N]*cMpd[N]+cF1d[N]**2*cMpd[N]+cF2d[N]**2*cMpd[N])
                +kmmAdMt*cSpd[N]+2*DMPt*(cMpd[N-1]-cMpd[N])/hxd**2)
      
    
    # -------------------------------
    # stress fibres
    # -------------------------------
    
    dcSpdt[:]=2*kmpMAt*cF1d*cF2d*cMpd+kmpMAt*(cF1d+cF2d)*cSpd*cMpd+kmpMAt*cF1d*cF1d*cMpd+kmpMAt*cF2d*cF2d*cMpd-kmmt*cSpd
    
    # -------------------------------
    # free integrins
    # -------------------------------
    
    dnfdt[0]=-khpt*nfd[0]+khmt*nhd[0]+2*Dnft*(nfd[1]-nfd[0])/hxd**2
    dnfdt[1:N]=-khpt*nfd[1:N]+khmt*nhd[1:N]+Dnft*(nfd[2:N+1]-2*nfd[1:N]+nfd[0:N-1])/hxd**2
    dnfdt[N]=-khpt*nfd[N]+khmt*nhd[N]+2*Dnft*(nfd[N-1]-nfd[N])/hxd**2

    # -------------------------------
    # high-affinity integrins 
    # -------------------------------
    
    dnhdt[:]=khpt*nfd-khmt*nhd-kbpSt*nhd*(ns1d+ns2d)+kbmt*(nb1d+nb2d)
    
    # -------------------------------
    # bound integrins (both families)
    # -------------------------------
    dnb1dt[:]=kbpSt*nhd*ns1d-kbmt*nb1d-kFpt*nb1d+kFmt*nA1d
    dnb2dt[:]=kbpSt*nhd*ns2d-kbmt*nb2d-kFpt*nb2d+kFmt*nA2d
    
    # -------------------------------
    # focal adhesions (both families)
    # -------------------------------
    
    dnA1dt[:]=kFpt*nb1d-kFmt*nA1d
    dnA2dt[:]=kFpt*nb2d-kFmt*nA2d
    
    # -------------------------------
    # ECM ligands (both families)
    # -------------------------------
    
    dns1dt[:]=-kbpNt*nhd*ns1d+kbmNdSt*nb1d
    dns2dt[:]=-kbpNt*nhd*ns2d+kbmNdSt*nb2d
     
    return dcGdt, dcF1dt, dcF2dt, dcMdt, dcMpdt, dcSpdt, dnfdt, dnhdt, dnb1dt, dnb2dt, dnA1dt, dnA2dt, dns1dt, dns2dt


