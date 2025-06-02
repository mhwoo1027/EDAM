#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import *
from scipy import interpolate
from scipy.optimize import broyden1
from functools import partial
import matplotlib.pyplot as plt
import pickle
from FEM_Solver import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from copy import deepcopy

"""
 This is Grad-Shafranov Equation Solver

   input :
       1. mesh - Mathematica-type mesh information
       2. Fs - Source term (profile-related)
       3. gefit - EFIT structure
       4. psie - Boundary Value of GS equation (normailzed flux)

   output : 
       1. Solution of Grad-Shafranov Equation Normalized Psi at mesh points
       2. Normalized Sigma - eigenvalue multiplied by source term 
    
"""

def GS_Solver(mesh,Fs,gefit,psie):
    
    
    Niter=20
    alpha=0.8 # 0.5~0.8 very accurate value
    
    dpsi=gefit['sibry'][0]-gefit['simag'][0] # Flux difference
    Ip=gefit['current'][0]

    psim=distr_mesh(mesh,gefit) # Psi value at every grid point
    
    BDelm=np.unique(mesh['BoundaryElements'].flatten()) # Boundary Elements
    
# Stiffness Matrix and Deploying Boundary condition for GS equation

    stiffnessMatrix=stiff_GS(mesh)

    deploy_BC_st(BDelm,stiffnessMatrix)

    spsM=csr_matrix(stiffnessMatrix) # Make matrix sparse to solve equtaion    

# Iteratively solve G-S equation
    
    Ip=gefit["current"][0]
    sig=1/dpsi**2.
    psitarget=psim
    psiold=psim
    
    for ii in range(Niter):
        
        [loadvector,IpsiN,LERR]=lv_GS(mesh,psitarget,psiold,Fs)
        sig=Ip/(IpsiN*dpsi) 
#        loadvector=(sig/dpsi**2.)*loadvector #Re-normalize load vector
        loadvector=sig*loadvector #Re-normalize load vector
        
        deploy_BC_lv(BDelm,loadvector,psie) # Deploy boundary condition for load vector
        psitmp=spsolve(spsM,loadvector)# Solve Linear Sparse equation to get solution
        psimin=min(psitmp)
        psisol=np.abs(psie*(psitmp-psimin)/(psie-psimin)) # Renormailze old flux
        psiold=deepcopy(psitarget) # Store old target value
        psitarget=psitarget*(1-alpha)+psisol*alpha # New target value 
        psimin=min(psitarget)
        psitarget=np.abs(psie*(psitarget-psimin)/(psie-psimin)) # Renormailze old flux
        
        Ip=(IpsiN/dpsi) # Updated Ip
        print(ii," th step")
        print("L2 error is",LERR)
        print("Sigma is",sig)
        print("Ip is",Ip)
    
    print('Calculation Complete' )
    
    #Here (sig/dpsi**2.) should be multiplied in front of F(X,psi), see eq.(9) of the paper 
    return [(psie/(psie-psimin)*sig), psitarget]

"""
Calculation of the stiffness Matrix for GS equation
  
GS Equation of the right hand term is given as

-(dpsi/dX)^2-(dpsi/dZ)^2-1/X(dpsi/dX)

Stiffness matrix is calculated using 'Qunitic(7point) integral order'

    input :  
        mesh

    output : 
        Stiffness Matrix

"""

def stiff_GS(mesh):
    
    qintp=quintic_inp()
    Nm=np.shape(mesh["MeshElements"])[0]
    Ns=np.shape(mesh["MeshElements"])[1]
    evp=np.array(qintp.coords)
    
    weight=np.array(qintp.weight)
    
    ws=np.shape(evp)[0]
    msize=np.shape(mesh['Coordinates'])[0]
    st=np.zeros((msize,msize),dtype=float)
    
    stcoefzeta=stm_coefzeta()
    stcoefeta=stm_coefeta()
    stcoefez=stm_coefez()
    
    stcoefvze=stm_coefvze()
    stcoefve=stm_coefve()

    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points
        # Edge points for triangle elements
        x1=pts[0,0]
        y1=pts[0,1]
        x2=pts[1,0]
        y2=pts[1,1]
        x3=pts[2,0]
        y3=pts[2,1]
        # Area of the mesh  
        Ak=(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2
        h3=(x3-x1)**2.+(y3-y1)**2.
        h2=(x2-x1)**2.+(y2-y1)**2.        
        hc=-(x1**2.+x2*x3-x1*(x2+x3)+(y1-y2)*(y1-y3))
        g31=y3-y1
        g12=y1-y2       
        
        for jj in range(ws):    
            zeta=evp[jj,0]
            eta=evp[jj,1]    
            xeval=x1*(1-eta-zeta)+x2*zeta+x3*eta
            
            for kk in range(Ns) :
                for ll in range(Ns) :
                    # Diffusion Coefficient contribution
                    st[cods[kk],cods[ll]]=st[cods[kk],cods[ll]]+\
                       (-1)*weight[jj]/(2.*Ak)*(h3*stcoefzeta[kk,ll,jj]+h2*stcoefeta[kk,ll,jj]+2*hc*stcoefez[kk,ll,jj])
                    # Convection Coefficient contribution
                    st[cods[kk],cods[ll]]=st[cods[kk],cods[ll]]+\
                       (-1)*weight[jj]/xeval*(g31*stcoefvze[kk,ll,jj]+g12*stcoefve[kk,ll,jj])
    
    return st


"""
Calculation of the Load Vecotr for GS equation
  
  GS Equation of the left hand term is given as

   Fs=(mu0*x**2.*ppri+ffpri)

   It is calculated using 'Qunitic(7point) integral order'

input :  1. mesh 
         2. psim -  Guessing Psi value 
         3. psiold - Old Psi value
         4. Fs - Source term

output : 1. lv -Load Vector (Later this term is multiplied by sig to give right term)
         2. IpsN - Value for Ip calculation
         3. sqrt(IpsiE) - L2 error

"""

def lv_GS(mesh,psim,psiold,Fs):
    
    qintp=quintic_inp()
    Nm=np.shape(mesh["MeshElements"])[0]
    Ns=np.shape(mesh["MeshElements"])[1]

    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    weight=np.array(qintp.weight)

    msize=np.shape(mesh['Coordinates'])[0]
    lv=np.zeros((msize,1),dtype=float)
    p=lv_coef()
 
    IpsiN=0.
    IpsiE=0.
    mu0=4*np.pi*1.e-7
    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points
        # Area of the mesh  
        Ak=(pts[0,0]*(pts[1,1]-pts[2,1])+pts[1,0]*(pts[2,1]-pts[0,1])+pts[2,0]*(pts[0,1]-pts[1,1]))/2
        psnd=psim[cods]
        psndold=psiold[cods]
        IpsiNK=0.      
        IpsiEK=0.
        
        for jj in range(ws):
            
            zeta=evp[jj,0]
            eta=evp[jj,1]
            xeval=pts[0,0]*(1-eta-zeta)+pts[1,0]*zeta+pts[2,0]*eta
            yeval=pts[0,1]*(1-eta-zeta)+pts[1,1]*zeta+pts[2,1]*eta
            psi=0.
            psio=0.
            for kk in range(Ns):
                psi=psi+p[kk,jj]*psnd[kk]  # Psi value at given point jj
                psio=psio+p[kk,jj]*psndold[kk] # Old Psi value at given point
                    
            tmp=Fs(psi,xeval,0) #This is F(psiN,X) - zero means no derivative
            
            IpsiNK=IpsiNK+2*Ak*weight[jj]*tmp/(mu0*xeval) # Total Plasma current calculation
            IpsiEK=IpsiEK+2*Ak*weight[jj]*(psi-psio)**2. # Total L2 error calculation

            for kk in range(Ns):
                lv[cods[kk]]=lv[cods[kk]]+2*Ak*weight[jj]*tmp*p[kk,jj]
       
        IpsiN=IpsiN+IpsiNK
        IpsiE=IpsiE+IpsiEK

    return [lv,IpsiN,np.sqrt(IpsiE)]


"""
Depoly Boundary condition for stiffnessMatrix

   Stiffness Matrix is changed to contain boundary condition
"""
def deploy_BC_st(BDelm,stiffnessMatrix):
    
    NM=np.size(BDelm)
    for ii in range(NM):
        idx=BDelm[ii]-1
        stiffnessMatrix[idx,:]=0.
        stiffnessMatrix[idx,idx]=1.

    return 

"""
Depoly Boundary condition for load vector

   load vector for boundary condition

"""
def deploy_BC_lv(BDelm,loadvector,psie):
    
    NM=np.size(BDelm)
    for ii in range(NM):
        loadvector[BDelm[ii]-1]=psie

    return 

"""
 Distribution of flux into mesh-point

   input : 
       1. meshm - standard Mathematica type mesh structure
       2. gefit - standard gefit structure
 
   output :  
       psim : Normailzed flux values at mesh points
             
"""
def distr_mesh(mesh,gefit):

    r=gefit['r'][0]
    z=gefit['z'][0]
    psin=(gefit['psirz'][0]-gefit['simag'][0])/(gefit['sibry'][0]-gefit['simag'][0])
    psi_intp=interpolate.RectBivariateSpline(z,r,psin)

    psim=np.zeros(size(mesh['Coordinates'][:,0]),float)
    for ii in range(size(mesh['Coordinates'][:,0])):
        psim[ii]=psi_intp(mesh['Coordinates'][ii,1],mesh['Coordinates'][ii,0])    
    
    return psim
