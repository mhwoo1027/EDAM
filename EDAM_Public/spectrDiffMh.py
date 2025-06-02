#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:16:40 2021

@author: mhwoo
"""

import numpy as np
from numpy import *

"""
Spectral Differentiation Matrix Calculation - From Books of Trefthen and Paper

    input :
        1. Nsize - number of points
        2. kc    - filter factor (w ave number smaller than kc is deleted )
    output :
        (DMFh1, DMFh2) - 1st and 2nd differentiaition Matrix 

"""

def spectDiffMh(Nsize,kc):
    
    h=2*np.pi/Nsize
    A=(np.pi/h-kc)
    B=h
    
    DFMh1=np.zeros((Nsize,Nsize),float)
    DFMh2=np.zeros((Nsize,Nsize),float)
    for ii in range(Nsize):
        for jj in range(Nsize):
            if ii==jj : 
                DFMh1[ii,jj]=0
                DFMh2[ii,jj]=-(((A+2*A**3.)*B)/(6*np.pi))
            else:
                DFMh1[ii,jj]=(B*(2*A*np.cos(A*h*(ii-jj))*1/(np.tan(1/2*h*(ii-jj)))-1/np.sin(1/2*h*(ii-jj))**2.*np.sin(A*h*(ii-jj))))/(4.*np.pi)
                DFMh2[ii,jj]=(B*(2*A*np.cos(A*h*(ii-jj))+(-1/np.tan(1/2*h*(ii-jj))+A**2.*np.sin(h*(ii-jj)))*sin(A*h*(ii-jj))))/(2*np.pi*(-1+np.cos(h*(ii-jj))))           

    return [DFMh1,DFMh2]


"""
Spectral Differentiation Matrix Calculation ( without filtering )
    input :
           Nsize - number of points
    output :
           DMFh - 1st differentiaition Matrix 

"""
def spectDiffMO(Nsize):
    
    h=2*np.pi/Nsize
    DFMh=np.zeros((Nsize,Nsize),float)
    for ii in range(Nsize):
        for jj in range(Nsize):
            if ii==jj : 
                DFMh[ii,jj]=0
            else:
                DFMh[ii,jj]=1/2*(-1)**(ii-jj)*1/np.tan((ii-jj)*h/2)

    return [DFMh]

"""
High wave number filter

Method : 
    1. Take FFT of the original data
    2. Strip wave numbers higher than kc
    3. Take inverse FFT of the spectral data

    input :
           1. data 
           2. kc - Filtering factor
    output :
           iv - filtered data 

"""
def higPssTrip(data,kc):
        
    n=size(data)
    if n%2 == 1 : 
        print("Number of array is not even !, quitting the program")
    h=2*np.pi/n
    nh=int(n/2) # Half of n
    x=np.linspace(h,2*pi,n)
    k=np.linspace(-nh+1,nh,n)
    
    v=np.zeros((n),complex)
    for ii in range(n) :
        v[ii]=h*sum(np.exp(-1j*x*k[ii])*data)

    vnew=np.zeros((n+1),complex)
    vnew[1:]=v[:]
    vnew[0]=v[-1]/2
    vnew[-1]=v[-1]/2
    knew=np.linspace(-nh,nh,n+1)
    for ii in range(n+1):
        if abs(ii-(nh)) > kc :
            vnew[ii]=0

#  Taking Inverse Fourrier Transformation
    iv=np.zeros((n),float)
    for jj in range(n):
        iv[jj]=np.real(1/(2*np.pi)*sum(np.exp(1j*knew*x[jj])*vnew))
    
    return iv

"""
FFT of given data
"""
def FFTv(data):
    
    n=size(data)
    if n%2 == 1 : 
        print("Number of array is not even !, quitting the program")
    h=2*np.pi/n
    nh=int(n/2) # Half of n
    x=np.linspace(h,2*pi,n)
    k=np.linspace(-nh+1,nh,n)

    v=np.zeros((n),complex)
    for ii in range(n) :
        v[ii]=h*sum(np.exp(-1j*x*k[ii])*data)
    
        
    return v
"""
inverse FFT of given spectral data
"""
def iFFTv(v):
    n=size(v)
    if n%2 == 1 : 
        print("Number of array is not even !, quitting the program")
    h=2*np.pi/n
    nh=int(n/2) # Half of n
    
    vnew=np.zeros((n+1),complex)
    vnew[1:]=v[:]
    vnew[0]=v[-1]/2
    vnew[-1]=v[-1]/2
    knew=np.linspace(-nh,nh,n+1)
    x=np.linspace(h,2*np.pi,n)

#  Taking Inverse Fourrier Transformation
    iv=np.zeros((n),float)
    for jj in range(n):
        iv[jj]=np.real(1/(2*np.pi)*sum(np.exp(1j*knew*x[jj])*vnew))
    
    return iv

"""
Derivative Parameters - stores tangential and normal derivative along the curve

    input :
           1. DrvM - Differentiation matrix 
           2. (Xpts,Zpts) - points of the curve
    output :
           1. Tx1,Tx2 - first and second tagential derivative
           2. PNX,PNZ - normalized normal vector
           3. PTX,PTZ - normailzed tangential vector

"""
def deriv_parameters(DrvM,Xpts,Zpts):
        
    # First derivative
    Tx1=np.dot(DrvM[0],Xpts)
    Tz1=np.dot(DrvM[0],Zpts)
    # Second derivative
    Tx2=np.dot(DrvM[0],Tx1)
    Tz2=np.dot(DrvM[0],Tz1)
        
    PNX= Tz1[:]/np.sqrt(Tz1[:]**2.+Tx1[:]**2.)
    PNZ=-Tx1[:]/np.sqrt(Tz1[:]**2.+Tx1[:]**2.)

    PTX= Tx1[:]/np.sqrt(Tz1[:]**2.+Tx1[:]**2.)
    PTZ= Tz1[:]/np.sqrt(Tz1[:]**2.+Tx1[:]**2.)    
        
    para={}
    para["Xpts"]=Xpts
    para["Zpts"]=Zpts
    para["Tx1"]=Tx1
    para["Tz1"]=Tz1
    para["Tx2"]=Tx2
    para["Tz2"]=Tz2
    para["PNX"]=PNX
    para["PNZ"]=PNZ
    para["PTX"]=PTX
    para["PTZ"]=PTZ

    return para
