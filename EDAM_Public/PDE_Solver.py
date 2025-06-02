#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun May  1 00:25:16 2022

Solve 2D PDE using FEM solver

PDE coefficient - Basic Notation follows as

    D   D         D
Aij -- -- U + Bj -- U  +aU  = f
   Dxi  Dxj      Dxj

xi,xj = {x,y} coordinate 
Aij : Diffusion Coefficients - Only Diagonal matrices aviable right now
Bj  : Convection Coefficients - We do not have this term right now
a   : Reaction Coefficients 
f   : Load Coefficients

Note that this convention a little different from Mathematica 

@author: mhwoo

"""

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
 PDE solver for first derivatives

   input : 
       1. mesh - Mathematica-type mesh information
       2. Coefs - Diffusion, Convection and Load Coefficients 
       3. psim - normailzed flux
       4. BDv - boundary values

   output : 
       sol - soltuion to Grad-Shafranov derivative equation (1st derivative)

"""
    
def PDE_Solver(mesh,Coefs,psim,BDv):
    
    print("Entering the PDE Solver")
    
    stiffnessMatrix=stiff_mat(Coefs,mesh,psim)
    
    BDelm=mesh['BoundaryNodes'] # Boundary Elements

    deploy_BC_stm(BDelm,stiffnessMatrix)
    
    spsM=csr_matrix(stiffnessMatrix) # Make matrix sparse to solve equtaion      
    
    lv=lv_mat(Coefs,mesh,psim) # Load vector solution
    
    deploy_BC_lvm(BDelm,lv,BDv)
    
    sol=spsolve(spsM,lv)
    
    return sol
"""
 PDE solver for second derivatives

   input : 
       1. mesh - Mathematica-type mesh information
       2. Coefs - Diffusion, Convection and Load Coefficients 
       3. psd - first derivatives at mesh points
       3. sdm - second derivatives at boundary
       4. tpe - boundary type ( xx,xz,zz derivatives )

   output : 
       sol - soltuion to Grad-Shafranov derivative equation (2nd derivative)

"""
def PDE_Solver_snd(mesh,Coefs,psd,sdm,tpe):
    
    print("Entering the second derivative - PDE Solver")
    
    stiffnessMatrix=stiff_mat(Coefs,mesh,psd["psi"])
    
    BDelm=mesh['BoundaryNodes'] # Boundary Elements

    deploy_BC_stm(BDelm,stiffnessMatrix)
    
    spsM=csr_matrix(stiffnessMatrix) # Make matrix sparse to solve equtaion      
    
    lv=lv_mat_snd(Coefs,mesh,psd,tpe) # Load vector
    
    if tpe =='xx':
        BDv=sdm["Uxx"]
    if tpe =='zz':
        BDv=sdm["Uzz"]
    if tpe == 'xz':
        BDv=sdm["Uxz"]
    
    deploy_BC_lvm(BDelm,lv,BDv)
    
    sol=spsolve(spsM,lv)
    
    return sol
"""
 PDE solver for third derivatives

   input : 
       1. mesh - Mathematica-type mesh information
       2. Coefs - Diffusion, Convection and Load Coefficients 
       3. psd - first, second derivatives at mesh points
       3. tdm - third derivatives at boundary
       4. tpe - boundary type ( xxx,zzz,xxz,xzz derivatives )

   output : 
       sol - soltuion to Grad-Shafranov derivative equation (2nd derivative)

"""
def PDE_Solver_tdm(mesh,Coefs,psd,tdm,tpe):
    
    print("Entering the third derivative - PDE Solver")
    
    stiffnessMatrix=stiff_mat(Coefs,mesh,psd["psi"])
    
    BDelm=mesh['BoundaryNodes'] # Boundary Elements

    deploy_BC_stm(BDelm,stiffnessMatrix)
    
    spsM=csr_matrix(stiffnessMatrix) # Make matrix sparse to solve equtaion      
    
    lv=lv_mat_tdm(Coefs,mesh,psd,tpe) # Load vector
    
    if tpe =='xxx':
        BDv=tdm["Uxxx"]
    if tpe =='zzz':
        BDv=tdm["Uzzz"]
    if tpe == 'xxz':
        BDv=tdm["Uxxz"]
    if tpe == 'xzz':
        BDv=tdm["Uxzz"]
    
    deploy_BC_lvm(BDelm,lv,BDv)
    
    sol=spsolve(spsM,lv)
    
    return sol

"""
Calculation of the stiffness Matrix for GS-D equation
  
GS Equation of the right hand term is given as

-(dU/dX)^2-(dU/dZ)^2+FuU

Stiffness matrix is calculated using 'Qunitic(7point) integral order'


    input :  
        1. mesh  - Mathematica type mesh information
        2. Coefs  - Diffusion, Convection and Load Coefficients 
        3. psim - flux at mesh points
    output :
        st - stiffness matrix
"""

def stiff_mat(Coefs,mesh,psim):
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
    stcoefra=stm_coefra()
    
    RC =Coefs.ReactionCoefficients    
    
    p=lv_coef()
    
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

        psnd=psim[cods]
        
        for jj in range(ws):    
            zeta=evp[jj,0]
            eta=evp[jj,1]    
            xeval=x1*(1-eta-zeta)+x2*zeta+x3*eta
            yeval=y1*(1-eta-zeta)+y2*zeta+y3*eta
            psi=0.
            for kk in range(Ns):
                psi=psi+p[kk,jj]*psnd[kk]  # Psi value at given point jj
            
            for kk in range(Ns) :
                for ll in range(Ns) :
                    # Diffusion Coefficient contribution
                    st[cods[kk],cods[ll]]=st[cods[kk],cods[ll]]+\
                       (-1)*weight[jj]/(2.*Ak)*(h3*stcoefzeta[kk,ll,jj]+h2*stcoefeta[kk,ll,jj]+2*hc*stcoefez[kk,ll,jj])
                    # Reaction Coefficient contribution
                    st[cods[kk],cods[ll]]=st[cods[kk],cods[ll]]-weight[jj]*RC(psi,xeval)*2*Ak*stcoefra[kk,ll,jj] # Fu goes as - in stiffness matrix
    return st



"""
Calculation of the Load Vecotr for GSD equation (1st derivative)
  
It is calculated using 'Qunitic(7point) integral order'

input :  1. Coefs - Diffusion, Convection and Load Coefficients 
         2. mesh - Mathematica type mesh
         3. psim - Calculated psi value 
output : 
        lv - loadvector
"""

def lv_mat(Coefs,mesh,psim):
    
    FF=Coefs.LoadCoefficients
    
    qintp=quintic_inp()
    Nm=np.shape(mesh["MeshElements"])[0]
    Ns=np.shape(mesh["MeshElements"])[1]

    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    weight=np.array(qintp.weight)


    msize=np.shape(mesh['Coordinates'])[0]
    lv=np.zeros((msize,1),dtype=float)
    p=lv_coef()
 
    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points
        # Area of the mesh  
        Ak=(pts[0,0]*(pts[1,1]-pts[2,1])+pts[1,0]*(pts[2,1]-pts[0,1])+pts[2,0]*(pts[0,1]-pts[1,1]))/2
        psnd=psim[cods]
        
        for jj in range(ws):
            zeta=evp[jj,0]
            eta=evp[jj,1]
            xeval=pts[0,0]*(1-eta-zeta)+pts[1,0]*zeta+pts[2,0]*eta
#            yeval=pts[0,1]*(1-eta-zeta)+pts[1,1]*zeta+pts[2,1]*eta
            psi=0.
            for kk in range(Ns):
                psi=psi+p[kk,jj]*psnd[kk]  # Psi value at given point jj
            
            tmp=FF(psi,xeval)
            for kk in range(Ns):
                lv[cods[kk]]=lv[cods[kk]]+2*Ak*weight[jj]*tmp*p[kk,jj]
       
    return lv
"""
Calculation of the Load Vecotr for GSD equation (2nd derivative)
  
It is calculated using 'Qunitic(7point) integral order'

    input :  1. Coefs - Diffusion, Convection and Load Coefficients 
             2. mesh - Mathematica type mesh
             3. psd - first derivative at mesh points
             4. tpe - type of derivative (xx,xz,xz)
    output : 
            lv - loadvector
"""

def lv_mat_snd(Coefs,mesh,psd,tpe):
    Fps=Coefs.LoadCoefficients
    
    qintp=quintic_inp()
    Nm=np.shape(mesh["MeshElements"])[0]
    Ns=np.shape(mesh["MeshElements"])[1]

    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    weight=np.array(qintp.weight)

    msize=np.shape(mesh['Coordinates'])[0]
    lv=np.zeros((msize,1),dtype=float)
    p=lv_coef()
 
    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points
        # Area of the mesh  
        Ak=(pts[0,0]*(pts[1,1]-pts[2,1])+pts[1,0]*(pts[2,1]-pts[0,1])+pts[2,0]*(pts[0,1]-pts[1,1]))/2

        psnd=psd["psi"][cods]
        if tpe == 'xx':
            usnd=psd["ux"][cods]
            usnd2=np.zeros(len(usnd))
        if tpe == 'zz':
            usnd=psd["uz"][cods]
            usnd2=np.zeros(len(usnd))
        if tpe == 'xz':
            usnd=psd["ux"][cods]
            usnd2=psd["uz"][cods]
      
        for jj in range(ws):
            
            zeta=evp[jj,0]
            eta=evp[jj,1]
            xeval=pts[0,0]*(1-eta-zeta)+pts[1,0]*zeta+pts[2,0]*eta
            
            psi=0.
            usi=0.
            usi2=0.
            for kk in range(Ns):
                psi=psi+p[kk,jj]*psnd[kk]  # Psi value at given point jj
                usi=usi+p[kk,jj]*usnd[kk]  # Psi value at given point jj
                usi2=usi2+p[kk,jj]*usnd2[kk]  # Psi value at given point jj
                
            if tpe == 'xx':
                tmp=Fps["FXX"](psi,xeval)+Fps["FUU"](psi,xeval)*usi**2.+2*Fps["FUX"](psi,xeval)*usi
            if tpe == 'zz':
                tmp=Fps["FUU"](psi,xeval)*usi**2.
            if tpe == 'xz':
                tmp=Fps["FUU"](psi,xeval)*usi*usi2+Fps["FUX"](psi,xeval)*usi2
            
            for kk in range(Ns):
                lv[cods[kk]]=lv[cods[kk]]+2*Ak*weight[jj]*tmp*p[kk,jj]
       
    return lv

"""
Calculation of the Load Vecotr for GSD equation (3rd derivative)
  
It is calculated using 'Qunitic(7point) integral order'

    input :  1. Coefs - Diffusion, Convection and Load Coefficients 
             2. mesh - Mathematica type mesh
             3. psd - first,second derivative at mesh points
             4. tpe - type of derivative (xxx,zzz,xxz,xzz)
    output : 
            lv - loadvector
"""
def lv_mat_tdm(Coefs,mesh,psd,tpe):

    Fps=Coefs.LoadCoefficients    
    qintp=quintic_inp()
    Nm=np.shape(mesh["MeshElements"])[0]
    Ns=np.shape(mesh["MeshElements"])[1]

    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    weight=np.array(qintp.weight)

    msize=np.shape(mesh['Coordinates'])[0]
    lv=np.zeros((msize,1),dtype=float)
    p=lv_coef()
 
    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points
        # Area of the mesh  
        Ak=(pts[0,0]*(pts[1,1]-pts[2,1])+pts[1,0]*(pts[2,1]-pts[0,1])+pts[2,0]*(pts[0,1]-pts[1,1]))/2

        psnd=psd["psi"][cods]
        uxnd=psd["ux"][cods]
        uznd=psd["uz"][cods]
        uxxnd=psd["uxx"][cods]
        uzznd=psd["uzz"][cods]
        uxznd=psd["uxz"][cods]
      
        for jj in range(ws):
            
            zeta=evp[jj,0]
            eta=evp[jj,1]
            xeval=pts[0,0]*(1-eta-zeta)+pts[1,0]*zeta+pts[2,0]*eta
            
            psi=0.
            ux=0.
            uz=0.
            uxx=0.
            uzz=0.
            uxz=0.
            for kk in range(Ns):
                psi=psi+p[kk,jj]*psnd[kk]
                ux=ux+p[kk,jj]*uxnd[kk]
                uz=uz+p[kk,jj]*uznd[kk]
                uxx=uxx+p[kk,jj]*uxxnd[kk]
                uzz=uzz+p[kk,jj]*uzznd[kk]
                uxz=uxz+p[kk,jj]*uxznd[kk]

            if tpe == 'xzz':
                tmp=Fps["FUX"](psi,xeval)*uzz+Fps["FUU"](psi,xeval)*(uzz*ux+2*uz*uxz)\
                    +uz**2.*(Fps["FUUU"](psi,xeval)*ux+Fps["FUUX"](psi,xeval))

            if tpe == 'xxx':
                tmp=ux**3.*Fps["FUUU"](psi,xeval)+3*ux**2.*Fps["FUUX"](psi,xeval)+3*uxx*Fps["FUX"](psi,xeval)\
                    +3*ux*(uxx*Fps["FUU"](psi,xeval)+Fps["FUXX"](psi,xeval))+Fps["FXXX"](psi,xeval)
                
            if tpe == 'zzz':
                tmp=uz**3.*Fps["FUUU"](psi,xeval)+3*uzz*uz*Fps["FUU"](psi,xeval)
                
            if tpe == 'xxz':
                tmp= 2*uxz*(ux*Fps["FUU"](psi,xeval)+Fps["FUX"](psi,xeval))\
                    +uz*(ux**2.*Fps["FUUU"](psi,xeval)+2*ux*Fps["FUUX"](psi,xeval)\
                         +uxx*Fps["FUU"](psi,xeval)+Fps["FUXX"](psi,xeval))
            
            for kk in range(Ns):
                lv[cods[kk]]=lv[cods[kk]]+2*Ak*weight[jj]*tmp*p[kk,jj]
       
    return lv

"""

Depoly Boundary condition for stiffnessMatrix

  --------  Stiffness Matrix is changed to evaluate boundary condition

"""
def deploy_BC_stm(BDelm,stiffnessMatrix):
    
    NM=np.size(BDelm)
    for ii in range(NM):
        idx=BDelm[ii]-1
        stiffnessMatrix[idx,:]=0.
        stiffnessMatrix[idx,idx]=1.

    return stiffnessMatrix

"""

Depoly Boundary condition for load vector

   load vector changed for boundary element points to load boundary condition

"""
def deploy_BC_lvm(BDelm,lv,BDv):
    NM=np.size(BDelm)
    for ii in range(NM):
        lv[BDelm[ii]-1]=BDv[ii]

    return lv



