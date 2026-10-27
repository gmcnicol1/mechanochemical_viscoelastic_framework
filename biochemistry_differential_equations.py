import numpy as np

def biochemistry_differential_equations(t,y,N,params):

    ca=y[0:N+1]
    cb=y[N+1:2*N+2]
    u=y[2*N+2:3*N+3]
    
    dcadt=np.zeros(N+1)
    dcbdt=np.zeros(N+1)

    # extract parameters
    Kfp=params['Kfp']
    Kfm=params['Kfm']
    hxd=params['hxd']
    Da=params['Da']
    Db=params['Db']
    
    kfp=Kfp*abs(u)
    kfm=Kfm*np.ones(N+1)
    
    dcadt[0]=-kfp[0]*ca[0]+kfm[0]*cb[0]+2*Da*(ca[1]-ca[0])/hxd**2
    dcadt[1:N]=-kfp[1:N]*ca[1:N]+kfm[1:N]*cb[1:N] +Da*(ca[2:N+1]-2*ca[1:N]+ca[0:N-1])/hxd**2
    dcadt[N]=-kfp[N]*ca[N]+kfm[N]*cb[N]+2*Da*(ca[N-1]-ca[N])/hxd**2

    dcbdt[0]=kfp[0]*ca[0]-kfm[0]*cb[0]+2*Db*(cb[1]-cb[0])/hxd**2
    dcbdt[1:N]=kfp[1:N]*ca[1:N]-kfm[1:N]*cb[1:N]+Db*(cb[2:N+1]-2*cb[1:N]+cb[0:N-1])/hxd**2
    dcbdt[N]=kfp[N]*ca[N]-kfm[N]*cb[N]+2*Db*(cb[N-1]-cb[N])/hxd**2
                     
    return dcadt, dcbdt
