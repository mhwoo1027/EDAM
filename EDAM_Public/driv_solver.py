#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:23:28 2023

Derivative Solver - for first, second and third derivatives

@author: mhwoo
"""
from scipy import interpolate
from PDE_Solver import *
from spectrDiffMh import *

import matplotlib.tri as mtri


"""
Filtered derivatives at the boundary - see the documentation for more details

    input :
        1. DM - differentiation matrix
        2. prms - derivative parameters
        3. psixz - derivative in x,z direction
        4. kc - filtering parameter
        
    output :
        dUxp,dUzp - fitlered derivative

"""
def filter_drv(DM,prms,psixz,kc):
    
    Tx1=prms["Tx1"]
    Tx2=prms["Tx2"]
    Tz1=prms["Tz1"]
    Tz2=prms["Tz2"]    
    
    nTx=Tx1/np.sqrt(Tx1**2.+Tz1**2.)
    nTz=Tz1/np.sqrt(Tx1**2.+Tz1**2.)
    tmp=-nTx*np.sqrt(psixz["Uxp"]**2.+psixz["Uzp"]**2.)
    pj=higPssTrip(np.sqrt(psixz["Uxp"]**2.+psixz["Uzp"]**2.),kc)    
    duzdt=-np.dot(DM[0],nTx)*pj-nTx*np.dot(DM[0],pj)
    duxdt=np.dot(DM[0],nTz)*pj+nTz*np.dot(DM[0],pj)
    
    res={}
    res["dUxp"]=duxdt
    res["dUzp"]=duzdt
    
    return res

"""
Second derivative along the boundary (By matrix inversion)

    input :
        1. DM - differentiation matrix
        2. prms - derivative parameters
        3. psie - boundary flux value
        4. dpsixz - t derivative of the ux, uz
        5. Fps - source related terms
    output :
        uxx,uzz,uxz - second derivative of u along the boundary
"""
def snd_drv_matrix(DM,prms,psie,dpsixz,Fps):
    
    Tx1=prms["Tx1"]
    Tx2=prms["Tx2"]
    Tz1=prms["Tz1"]
    Tz2=prms["Tz2"]    
    
# Matrix contruction
    Nsize=len(Tx1)
    DseUxx1=np.identity(Nsize)
    DseUxx2=np.diag(Tx1)
    DseUxx3=np.zeros((Nsize,Nsize))
    DseUzz1=np.identity(Nsize)
    DseUzz2=np.zeros((Nsize,Nsize))
    DseUzz3=np.diag(Tz1)
    DseUxz1=np.zeros((Nsize,Nsize))
    DseUxz2=np.diag(Tz1)
    DseUxz3=np.diag(Tx1)

    Uxx=np.concatenate((DseUxx1,DseUxx2,DseUxx3),axis=0) # Vertical concatenate
    Uzz=np.concatenate((DseUzz1,DseUzz2,DseUzz3),axis=0)
    Uxz=np.concatenate((DseUxz1,DseUxz2,DseUxz3),axis=0)
    Dse=np.concatenate((Uxx,Uzz,Uxz),axis=1) # Horizontal concatenate
    
# Source Term
    Fv=Fps["FF"](psie,prms["Xpts"])
    Dx=dpsixz["dUxp"]
    Dz=dpsixz["dUzp"]
    Sr=np.concatenate((Fv,Dx,Dz),axis=0)
    Sols=np.dot(np.linalg.inv(Dse),Sr)    
    res={} 
    res["Uxx"]=Sols[0:Nsize]
    res["Uzz"]=Sols[Nsize:2*Nsize]
    res["Uxz"]=Sols[2*Nsize:3*Nsize]
    
    return res

"""
Third derivative along the boundary (By matrix inversion)

    input :
        1. DM - differentiation matrix
        2. prms - derivative parameters
        3. psie - boundary flux value
        4. dpsixz - t derivative of the ux, uz
        5. psixz - ux,uz values at boundary
        5. sdm - second derivative along the boundary
        6. Fps - source related terms
    output :
        uxxx,uzzz,uxxz,uzxx - third derivative of u along the boundary

"""
def thrd_drv_matrix(DM,prms,psie,dpsixz,psixz,sdm,Fps):
    
    Tx1=prms["Tx1"]
    Tx2=prms["Tx2"]
    Tz1=prms["Tz1"]
    Tz2=prms["Tz2"]    
    xpts=prms["Xpts"] 
    
# Direct Insverse matrices (see the F_derivative.nb for details)
    den=(Tx1**2.+Tz1**2.)**2.
    D11=(Tz1**2.-Tx1**2.)/den
    D12=-2*Tx1*Tz1/den
    D13=2*Tx1**3.*Tz1/den
    D14=(Tx1**4.+3*Tx1**2.*Tz1**2.)/den
    D21=-2*Tx1*Tz1/den
    D22=(Tx1**2.-Tz1**2.)/den
    D23=(3*Tx1**2.*Tz1**2.+Tz1**4.)/den
    D24=2*Tx1*Tz1**3./den
    D31=2*Tx1*Tz1/den
    D32=(Tz1**2.-Tx1**2.)/den
    D33=(Tx1**4.-Tx1**2.*Tz1**2.)/den
    D34=-2*Tx1*Tz1**3./den
    D41=(Tx1**2.-Tz1**2.)/den
    D42=2*Tx1*Tz1/den
    D43=-2*Tx1**3.*Tz1/den
    D44=(Tz1**4.-Tx1**2.*Tz1**2.)/den
    
# Direct source term
    DDUZ=np.dot(DM[0],dpsixz['dUzp'])
    DDUX=np.dot(DM[0],dpsixz['dUxp'])
    F1=DDUZ-Tz2*sdm["Uzz"]-Tx2*sdm["Uxz"]
    F2=DDUX-Tz2*sdm["Uxz"]-Tx2*sdm["Uxx"]
    F3=Fps["FX"](psie,xpts)+Fps["FU"](psie,xpts)*psixz['Uxp']
    F4=Fps["FU"](psie,xpts)*psixz['Uzp']
    
    res={} 
    res["Uzzz"]=D11*F1+D12*F2+D13*F3+D14*F4
    res["Uxxx"]=D21*F1+D22*F2+D23*F3+D24*F4
    res["Uxzz"]=D31*F1+D32*F2+D33*F3+D34*F4
    res["Uxxz"]=D41*F1+D42*F2+D43*F3+D44*F4
    
    return res


"""
First Derivative for entire region :
    A second order PDE for ux and uz is solved here, this is linear 2nd order inhomogeneous PDE
    which can be solved by formal FEM method

    input :
        1. psixz - ux,uz values at boundary
        2. psixz_mid - ux,uz values at boundary mid-points
        3. Fps - source related terms
        4. mesh - Mathematica type mesh
        5. psim - normalized flux value
        
    output :
        uxr,uzr - first derivative at mesh points (including mid-points)

"""
def fstdrv_region(psixz,psixz_mid,Fps,mesh,psim):
    
#Convection coefficient
    CC= lambda : 0
#Diffusion coefficient
    D11= lambda : 1
    D21= lambda : 0
    D12= lambda : 0
    D22= lambda : 1
#Reaction coefficient  -----> -Fu
    RC= Fps["FU"]

#Distributue values at integral points
    Coefs=PDEcoef()
    Coefs.DiffusionCoefficients=np.array([[D11,D12],[D21,D22]])
    Coefs.ConvectionCoefficients=CC
    Coefs.ReactionCoefficients = RC

#Construct boundary condition size of psixz = mesh["BoundaryNodes"]
    Ns=len(psixz["Uxp"])
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(psixz["Uxp"][ii])
        bc_dirich.append(psixz_mid["Uxp"][ii])

    Coefs.LoadCoefficients = Fps["FX"]
    solx=PDE_Solver(mesh,Coefs,psim,np.array(bc_dirich))

    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(psixz["Uzp"][ii])
        bc_dirich.append(psixz_mid["Uzp"][ii])

    Coefs.LoadCoefficients = Fps["FZ"]
    solz=PDE_Solver(mesh,Coefs,psim,np.array(bc_dirich))
    
    sol={}
    sol["Uxr"]=solx
    sol["Uzr"]=solz    
    
    return sol

"""
Second Derivative for entire region

    input :
        1. sdm - second derivatives at the boundary
        2. psd - first derivatives at mesh points
        3. Fps - source related terms
        4. mesh - Mathematica type mesh
        
    output :
        uxxr,uxzr,uzzr - second derivative at mesh points (including mid-points)

"""
def scndrv_region(sdm,psd,Fps,mesh):

#Convection coefficient
    CC= lambda : 0
    
#Diffusion coefficient
    D11= lambda : 1
    D21= lambda : 0
    D12= lambda : 0
    D22= lambda : 1
    
#Reaction coefficient
    RC= Fps["FU"]
    Coefs=PDEcoef()
    Coefs.DiffusionCoefficients=np.array([[D11,D12],[D21,D22]])
    Coefs.ConvectionCoefficients=CC
    Coefs.ReactionCoefficients = RC

#Define boundary condition 
    Coefs.LoadCoefficients = Fps  
    solxx=PDE_Solver_snd(mesh,Coefs,psd,sdm,'xx')
    solzz=PDE_Solver_snd(mesh,Coefs,psd,sdm,'zz')
    solxz=PDE_Solver_snd(mesh,Coefs,psd,sdm,'xz')
  
    sol={}
    sol["Uxxr"]=solxx
    sol["Uzzr"]=solzz   
    sol["Uxzr"]=solxz
    
    return sol


"""
Third Derivative in region

    input :
        1. tdm - third derivatives at the boundary
        2. psd - first,second derivatives at mesh points
        3. Fps - source related terms
        4. mesh - Mathematica type mesh
        
    output :
        uxxxr,uxxzr,uxzzr,uzzzr - third derivative at mesh points (including mid-points)

"""
def tdmdrv_region(tdm,psd,Fps,mesh):
    
#Convection coefficient
    CC= lambda : 0
    
#Diffusion coefficient
    D11= lambda : 1
    D21= lambda : 0
    D12= lambda : 0
    D22= lambda : 1
    
#Reaction coefficient
    RC= Fps["FU"]
    Coefs=PDEcoef()
    Coefs.DiffusionCoefficients=np.array([[D11,D12],[D21,D22]])
    Coefs.ConvectionCoefficients=CC
    Coefs.ReactionCoefficients = RC

#Define boundary condition 
    
    Coefs.LoadCoefficients = Fps
    
    solxxx=PDE_Solver_tdm(mesh,Coefs,psd,tdm,'xxx')
        
    solzzz=PDE_Solver_tdm(mesh,Coefs,psd,tdm,'zzz')
    
    solxzz=PDE_Solver_tdm(mesh,Coefs,psd,tdm,'xzz')

    solxxz=PDE_Solver_tdm(mesh,Coefs,psd,tdm,'xxz')
    
    sol={}
    sol["Uxxxr"]=solxxx
    sol["Uzzzr"]=solzzz 
    sol["Uxzzr"]=solxzz
    sol["Uxxzr"]=solxxz
    
    return sol

