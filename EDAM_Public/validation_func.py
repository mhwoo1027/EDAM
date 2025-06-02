#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:49:48 2023

Validation functions for comparision with EFIT result and calculated result

@author: mhwoo
"""


import numpy as np
from scipy import interpolate

########################### A First derivative simple EFIT comparison function #################
def psi_didi(gefit,mesh,psie):
    
    rp=gefit['r'][0]
    zp=gefit['z'][0]
        
    r0=gefit['rmaxis'][0]
    z0=gefit['zmaxis'][0]
    psin=(gefit['psirz'][0]-gefit['simag'][0])/(gefit['sibry'][0]-gefit['simag'][0])
    psi_intp=interpolate.RectBivariateSpline(zp,rp,psin)       
    
    cods=mesh["Coordinates"]
    Nm=np.shape(cods)[0]
    
    dpsix=np.zeros(Nm,dtype=float)
 
    for ii in range(Nm):
        x=cods[ii,0]
        dpsix[ii]=(2*x*psi_intp(cods[ii,1],cods[ii,0],dy=1)+psie-psi_intp(cods[ii,1],cods[ii,0]))/(2*x**1.5)
    
    dpsiz=np.zeros(Nm,dtype=float)    
    for ii in range(Nm):
        x=cods[ii,0]
        dpsiz[ii]=(psi_intp(cods[ii,1],cods[ii,0],dx=1))/(x**0.5)
    
    psi=np.zeros(Nm,dtype=float)    
    for ii in range(Nm):
        psi[ii]=(psi_intp(cods[ii,1],cods[ii,0]))
    
    
    sol={}
    sol["Uxrd"]=dpsix
    sol["Uzrd"]=dpsiz
    sol["Psi"]=psi

    return sol

########################### A second derivtiave EFIT comparison function ######
def psi_didi2(gefit,mesh,psie,Xpts,Zpts):
    
    rp=gefit['r'][0]
    zp=gefit['z'][0]
        
    r0=gefit['rmaxis'][0]
    z0=gefit['zmaxis'][0]
    psin=(gefit['psirz'][0]-gefit['simag'][0])/(gefit['sibry'][0]-gefit['simag'][0])
    psi_intp=interpolate.RectBivariateSpline(zp,rp,psin)       
    
    cods=mesh["Coordinates"]
    Nm=np.shape(cods)[0]
    
    Uxx=np.zeros(Nm,dtype=float)
 
    for ii in range(Nm):
        x=cods[ii,0]
        psi=psi_intp(cods[ii,1],cods[ii,0])
        dpX=psi_intp(cods[ii,1],cods[ii,0],dy=1)
        ddpX=psi_intp(cods[ii,1],cods[ii,0],dy=2)
        Uxx[ii]=(-3*psie+3*psi-4*x*dpX+4*x**2.*ddpX)/(4*x**2.5)

    Uzz=np.zeros(Nm,dtype=float)
    for ii in range(Nm):
        x=cods[ii,0]
        Uzz[ii]=(psi_intp(cods[ii,1],cods[ii,0],dx=2))/(x**0.5)
    
    Uxz=np.zeros(Nm,dtype=float)
    for ii in range(Nm):
        x=cods[ii,0]
        Uxz[ii]=(-psi_intp(cods[ii,1],cods[ii,0],dx=1)+2*x*psi_intp(cods[ii,1],cods[ii,0],dx=1,dy=1))/(2*x**1.5)


    Uxxb=np.zeros(np.size(Xpts),dtype=float)
    Uzzb=np.zeros(np.size(Xpts),dtype=float)
    Uxzb=np.zeros(np.size(Xpts),dtype=float)
    
    for ii in range(np.size(Xpts)):
        x=Xpts[ii]
        z=Zpts[ii]
        psi=psi_intp(z,x)
        dpX=psi_intp(z,x,dy=1)
        ddpX=psi_intp(z,x,dy=2)
        Uxxb[ii]=(-3*psie+3*psi-4*x*dpX+4*x**2.*ddpX)/(4*x**2.5)
        Uzzb[ii]=(psi_intp(z,x,dx=2))/(x**0.5)
        Uxzb[ii]=(-psi_intp(z,x,dx=1)+2*x*psi_intp(z,x,dx=1,dy=1))/(2*x**1.5)
    
    sol={}
    sol["Uxxr"]=Uxx
    sol["Uzzr"]=Uzz
    sol["Uxzr"]=Uxz
    sol["Uxxb"]=Uxxb
    sol["Uzzb"]=Uzzb
    sol["Uxzb"]=Uxzb
    
    return sol



########################### A third derivtiave EFIT comparison function ######
def psi_didi3(gefit,mesh,psie,Xpts,Zpts):
    
    rp=gefit['r'][0]
    zp=gefit['z'][0]
        
    r0=gefit['rmaxis'][0]
    z0=gefit['zmaxis'][0]
    psin=(gefit['psirz'][0]-gefit['simag'][0])/(gefit['sibry'][0]-gefit['simag'][0])
    psi_intp=interpolate.RectBivariateSpline(zp,rp,psin)
    psi_intp=interpolate.RectBivariateSpline(zp,rp,psin,kx=4,ky=4)
    
    cods=mesh["Coordinates"]
    Nm=np.shape(cods)[0]
    
    Uxxx=np.zeros(Nm,dtype=float)
 
    for ii in range(Nm):
        x=cods[ii,0]
        psi=psi_intp(cods[ii,1],cods[ii,0])
        dpX=psi_intp(cods[ii,1],cods[ii,0],dy=1)
        ddpX=psi_intp(cods[ii,1],cods[ii,0],dy=2)
        dddpX=psi_intp(cods[ii,1],cods[ii,0],dy=3)
        Uxxx[ii]=(15*psie-15*psi+2*x*(9*dpX-6*x*ddpX+4*x**2.*dddpX))/(8*x**3.5)

    Uzzz=np.zeros(Nm,dtype=float)
    for ii in range(Nm):
        x=cods[ii,0]
        Uzzz[ii]=(psi_intp(cods[ii,1],cods[ii,0],dx=3))/(x**0.5)

    Uxzz=np.zeros(Nm,dtype=float)
    for ii in range(Nm):
        x=cods[ii,0]
        Uxzz[ii]=(-psi_intp(cods[ii,1],cods[ii,0],dx=2)+2*psi_intp(cods[ii,1],cods[ii,0],dy=1,dx=2))/(2*x**1.5)

    Uxxz=np.zeros(Nm,dtype=float)
    for ii in range(Nm):
        x=cods[ii,0]
        Uxxz[ii]=(3*psi_intp(cods[ii,1],cods[ii,0],dx=1)+4*x*(-psi_intp(cods[ii,1],cods[ii,0],dy=1,dx=1)+x*psi_intp(cods[ii,1],cods[ii,0],dy=2,dx=1)))/(4*x**2.5)


    Uxxxb=np.zeros(np.size(Xpts),dtype=float)
    Uzzzb=np.zeros(np.size(Xpts),dtype=float)
    Uxzzb=np.zeros(np.size(Xpts),dtype=float)
    Uxxzb=np.zeros(np.size(Xpts),dtype=float)
    
    for ii in range(np.size(Xpts)):
        x=Xpts[ii]
        z=Zpts[ii]
        psi=psi_intp(z,x)
        dpX=psi_intp(z,x,dy=1)
        ddpX=psi_intp(z,x,dy=2)
        dddpX=psi_intp(z,x,dy=3)
        Uxxxb[ii]=(15*psie-15*psi+2*x*(9*dpX-6*x*ddpX+4*x**2.*dddpX))/(8*x**3.5)
        Uzzzb[ii]=(psi_intp(z,x,dx=3))/(x**0.5)
        Uxzzb[ii]=(-psi_intp(z,x,dx=2)+2*psi_intp(z,x,dy=1,dx=2))/(2*x**1.5)
        Uxxzb[ii]=(3*psi_intp(z,x,dx=1)+4*x*(-psi_intp(z,x,dy=1,dx=1)+x*psi_intp(z,x,dy=2,dx=1)))/(4*x**2.5)        
        
    sol={}
    sol["Uxxxr"]=Uxxx
    sol["Uzzzr"]=Uzzz
    sol["Uxzzr"]=Uxzz
    sol["Uxxzr"]=Uxxz
    sol["Uxxxb"]=Uxxxb
    sol["Uzzzb"]=Uzzzb
    sol["Uxzzb"]=Uxzzb
    sol["Uxxzb"]=Uxxzb
    
    return sol

