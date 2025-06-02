#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:57:31 2022


*******************************************************************************

   Finite Element Solver of 2D Elliptic Equation
 
    - A python version of Mathematica NDSolve(FEM) 
    
   1. Element mesh is C^0 Quadratic elements
   2. Triangle mesh with mid-points
   3. Boundary condition is Dirichlet only
   4. Boundary values are given along the discretized boundary
   5. Numerical integration order is 5, quintic 7-point integral  

                                                     Written by M.H.Woo (2022)
*******************************************************************************

"""

import numpy as np
from scipy import interpolate


"""

PDE coefficient - Basic Notation follows from the Mathematica, it is given as

D      D         D
-- Aij -- U + Bj -- U  +aU  = f
Dxi    Dxj      Dxj

xi,xj = {x,y} coordinate 
Aij : Diffusion Coefficients
Bj  : Convection Coefficients
a   : Reaction Coefficients
f   : Load Coefficients

Overall U is defined as

P1 = 1 - 3 \[Eta] + 2 \[Eta]^2 + (-3 + 4 \[Eta]) \[Xi] + 2 \[Xi]^2;
P2 = \[Xi] (-1 + 2 \[Xi]);
P3 = \[Eta] (-1 + 2 \[Eta]);
P4 = -4 \[Xi] (-1 + \[Eta] + \[Xi]);
P5 = 4 \[Eta] \[Xi];
P6 = -4 \[Eta] (-1 + \[Eta] + \[Xi]);

with

U = P1*a1 + P2*a2 + P3*a3 + P4*a4 + P5*a5 + P6*a6 (for diffusion coefficient only)
U = P1*b1 + P2*b2 + P3*b3 + P4*b4 + P5*b5 + P6*b6 (including Convection coefficietn)



"""
class PDEcoef :
    
    def __init__(self):        
        self.LoadCoefficients=0
        self.DiffusionCoefficients=0
        self.ConvectionCoefficients=0
        self.ReactionCoefficients=0

"""
Gauss quadrature for integration points - A Mathematica Version 

See the integral

    INT_TRIANGLE g(x,y) dx dy =  sum_{i~Ng} weight * g(xi,yi)

Here, we assume degree of interpoation function is up to degree Ng=5,
   1,x,y,x^2y,y^2x, ... x^4y, x^5,y^5 - base element polynomial has dim(PN)=21 components

Total Required number of points are 7 points.

7 weighting factors & 7 evaluation points are given in eta,zeta coordinate - standard triangle

It is correct up to O(h^6)

Book of Zienkiewicz, page 165

Exact values are highly nonlinear and we just use numberical values for consistency

!!! Adding all weight factor gives 1/2 but not 1 as given by Note of UNCC.EDU note!!!

"""

class quintic_inp:
    
    def __init__(self):
        self.coords=[[0.10128650732345633, 0.10128650732345633], \
                     [0.7974269853530872, 0.10128650732345633], \
                     [0.10128650732345633, 0.7974269853530872], \
                     [0.47014206410511505, 0.0597158717897698], \
                     [0.47014206410511505, 0.47014206410511505], \
                     [0.0597158717897698, 0.47014206410511505], \
                     [0.3333333333333333, 0.3333333333333333]]
        self.weight=[0.06296959027241358, 0.06296959027241358, 0.06296959027241358, \
                     0.0661970763942531, 0.0661970763942531, 0.0661970763942531,0.1125]


"""
Gaussian Quadrature for order 4 in each direction, 16 points with 16 weight factors

correct up to h^4


"""
class quintic_inp_quad:
    
    def __init__(self):
        
        self.coords=[[-0.8611363115940526,-0.8611363115940526],\
                     [-0.8611363115940526, 0.8611363115940526],\
                     [-0.8611363115940526,-0.33998104358485626],\
                     [-0.8611363115940526, 0.33998104358485626],\
                     [ 0.8611363115940526,-0.8611363115940526],\
                     [ 0.8611363115940526, 0.8611363115940526],\
                     [ 0.8611363115940526,-0.33998104358485626],\
                     [ 0.8611363115940526, 0.33998104358485626],\
                     [-0.33998104358485626,-0.8611363115940526],\
                     [-0.33998104358485626, 0.8611363115940526],\
                     [-0.33998104358485626,-0.33998104358485626],\
                     [-0.33998104358485626, 0.33998104358485626],\
                     [0.33998104358485626,-0.8611363115940526],\
                     [0.33998104358485626, 0.8611363115940526],\
                     [0.33998104358485626,-0.33998104358485626],\
                     [0.33998104358485626,0.33998104358485626]]
        
        self.weight= [0.121002993285602, 0.121002993285602, 0.22685185185185186,\
                      0.22685185185185186, 0.121002993285602, 0.121002993285602,\
                      0.22685185185185186, 0.22685185185185186, 0.22685185185185186,\
                      0.22685185185185186, 0.42529330301069435, 0.42529330301069435,\
                      0.22685185185185186, 0.22685185185185186, 0.42529330301069435,\
                      0.42529330301069435]


"""
Integration Evaluation point (x,z) for each mesh element


input :   mesh - mesh element
          qintp - interpolation point specification for numerical integration

"""  

def evpt_mesh(mesh,qintp):
    
    
    Nm=np.shape(mesh["MeshElements"])[0]
    evp=np.array(qintp.coords)
    
    ws=np.shape(evp)[0]
    
    rvalue=np.zeros((Nm,ws,2),dtype=float)
    
    for ii in range(Nm):    
        cods=mesh["MeshElements"][ii]-1 #Mathematica Indexing Starts from 1 while Python array starts from 0
        pts=mesh["Coordinates"][cods][:3] # First three triangle points    
        
        
        for jj in range(ws):
            N1=1-evp[jj,0]-evp[jj,1]
            N2=evp[jj,0]
            N3=evp[jj,1]
            xeval=pts[0,0]*N1+pts[1,0]*N2+pts[2,0]*N3
            yeval=pts[0,1]*N1+pts[1,1]*N2+pts[2,1]*N3
            rvalue[ii,jj,0]=xeval
            rvalue[ii,jj,1]=yeval
            
            
    return rvalue
            

"""

Load Vector coefficient Calculation

  U = P1*a1+P2*a2+P3*a3+P4*a4+P5*a5+P6*a6 

"""          

def lv_coef():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    p=np.zeros((6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]

        p[0,jj]=1-3*eta+2*(eta**2.)+(-3+4*eta)*zeta+2*(zeta**2.)
        p[1,jj]=zeta*(-1+2*zeta)
        p[2,jj]=eta*(-1+2*eta)
        p[3,jj]=-4*zeta*(-1+eta+zeta)
        p[4,jj]=4*zeta*eta
        p[5,jj]=-4*eta*(-1+eta+zeta)
        
    return p


"""

Stiffness coefficient Calculation

  (d u / d zeta )^2 : 

1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefzeta():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
        
        stf_coef[0,0,jj]= (-3+4*eta+4*zeta)**2.
        stf_coef[0,1,jj]= (-1+4*zeta)*(-3+4*eta+4*zeta)
        stf_coef[1,0,jj]= (-1+4*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,2,jj]= 0.
        stf_coef[2,0,jj]= 0.
        stf_coef[0,3,jj]=-4*(-1+eta+2*zeta)*(-3+4*eta+4*zeta)
        stf_coef[3,0,jj]=-4*(-1+eta+2*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,4,jj]= 4*eta*(-3+4*eta+4*zeta)
        stf_coef[4,0,jj]= 4*eta*(-3+4*eta+4*zeta)           
        stf_coef[0,5,jj]=-4*eta*(-3+4*eta+4*zeta)
        stf_coef[5,0,jj]=-4*eta*(-3+4*eta+4*zeta)

        stf_coef[1,1,jj]= (1-4*zeta)**2.
        stf_coef[1,2,jj]= 0.
        stf_coef[2,1,jj]= 0.
        stf_coef[1,3,jj]=-4*(-1+eta+2*zeta)*(-1+4*zeta)
        stf_coef[3,1,jj]=-4*(-1+eta+2*zeta)*(-1+4*zeta)
        stf_coef[1,4,jj]=-4*eta*(1-4*zeta)
        stf_coef[4,1,jj]=-4*eta*(1-4*zeta)
        stf_coef[1,5,jj]=-4*eta*(-1+4*zeta)
        stf_coef[5,1,jj]=-4*eta*(-1+4*zeta)
             
        stf_coef[2,2,jj]= 0.
        stf_coef[2,3,jj]= 0.
        stf_coef[3,2,jj]= 0.
        stf_coef[2,4,jj]= 0.
        stf_coef[4,2,jj]= 0.
        stf_coef[2,5,jj]= 0.
        stf_coef[5,2,jj]= 0.
        
        stf_coef[3,3,jj]= 16*(-1+eta+2*zeta)**2.
        stf_coef[3,4,jj]=-16*eta*(-1+eta+2*zeta)
        stf_coef[4,3,jj]=-16*eta*(-1+eta+2*zeta)
        stf_coef[3,5,jj]= 16*eta*(-1+eta+2*zeta)
        stf_coef[5,3,jj]= 16*eta*(-1+eta+2*zeta)
        
        stf_coef[4,4,jj]= 16*eta**2.
        stf_coef[4,5,jj]=-16*eta**2.
        stf_coef[5,4,jj]=-16*eta**2.
        
        stf_coef[5,5,jj]= 16*eta**2.

    return stf_coef



"""

Stiffness coefficient Calculation 

 (d u /d eta)^2 : 

Here note the minus sign 

1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefeta():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
        
        stf_coef[0,0,jj]= (-3+4*eta+4*zeta)**2.
        stf_coef[0,1,jj]= 0.
        stf_coef[1,0,jj]= 0.
        stf_coef[0,2,jj]= (-1+4*eta)*(-3+4*eta+4*zeta)
        stf_coef[2,0,jj]= (-1+4*eta)*(-3+4*eta+4*zeta)
        stf_coef[0,3,jj]=-4*zeta*(-3+4*eta+4*zeta)
        stf_coef[3,0,jj]=-4*zeta*(-3+4*eta+4*zeta)
        stf_coef[0,4,jj]= 4*zeta*(-3+4*eta+4*zeta)
        stf_coef[4,0,jj]= 4*zeta*(-3+4*eta+4*zeta)           
        stf_coef[0,5,jj]=-4*(-1+2*eta+zeta)*(-3+4*eta+4*zeta)
        stf_coef[5,0,jj]=-4*(-1+2*eta+zeta)*(-3+4*eta+4*zeta)

        stf_coef[1,1,jj]= 0.
        stf_coef[1,2,jj]= 0.
        stf_coef[2,1,jj]= 0.
        stf_coef[1,3,jj]= 0.
        stf_coef[3,1,jj]= 0.
        stf_coef[1,4,jj]= 0.
        stf_coef[4,1,jj]= 0.
        stf_coef[1,5,jj]= 0.
        stf_coef[5,1,jj]= 0.
             
        stf_coef[2,2,jj]= (1-4*eta)**2.
        stf_coef[2,3,jj]=-4*(-1+4*eta)*zeta
        stf_coef[3,2,jj]=-4*(-1+4*eta)*zeta
        stf_coef[2,4,jj]=-4*(1-4*eta)*zeta
        stf_coef[4,2,jj]=-4*(1-4*eta)*zeta
        stf_coef[2,5,jj]=-4*(-1+4*eta)*(-1+2*eta+zeta)
        stf_coef[5,2,jj]=-4*(-1+4*eta)*(-1+2*eta+zeta)
        
        stf_coef[3,3,jj]= 16*zeta**2.
        stf_coef[3,4,jj]=-16*zeta**2.
        stf_coef[4,3,jj]=-16*zeta**2.
        stf_coef[3,5,jj]= 16*zeta*(-1+2*eta+zeta)
        stf_coef[5,3,jj]= 16*zeta*(-1+2*eta+zeta)
        
        stf_coef[4,4,jj]= 16*zeta**2.
        stf_coef[4,5,jj]=-16*zeta*(-1+2*eta+zeta)
        stf_coef[5,4,jj]=-16*zeta*(-1+2*eta+zeta)
        
        stf_coef[5,5,jj]= 16*(-1+2*eta+zeta)**2.

    return stf_coef





"""

Stiffness coefficient Calculation 

  (d u /d eta) * (d u /d zeta) : 

     
1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefez():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
        
        stf_coef[0,0,jj]= (-3+4*eta+4*zeta)**2.
        stf_coef[0,1,jj]= 0.5*(-1+4*zeta)*(-3+4*eta+4*zeta)
        stf_coef[1,0,jj]= 0.5*(-1+4*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,2,jj]= 0.5*(-1+4*eta)*(-3+4*eta+4*zeta)
        stf_coef[2,0,jj]= 0.5*(-1+4*eta)*(-3+4*eta+4*zeta)
        stf_coef[0,3,jj]=-2*(-1+eta+3*zeta)*(-3+4*eta+4*zeta)
        stf_coef[3,0,jj]=-2*(-1+eta+3*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,4,jj]= 2*(eta+zeta)*(-3+4*eta+4*zeta)
        stf_coef[4,0,jj]= 2*(eta+zeta)*(-3+4*eta+4*zeta)           
        stf_coef[0,5,jj]=-2*(-1+3*eta+zeta)*(-3+4*eta+4*zeta)
        stf_coef[5,0,jj]=-2*(-1+3*eta+zeta)*(-3+4*eta+4*zeta)

        stf_coef[1,1,jj]= 0.
        stf_coef[1,2,jj]= 0.5*(-1+4*eta)*(-1+4*zeta)
        stf_coef[2,1,jj]= 0.5*(-1+4*eta)*(-1+4*zeta)
        stf_coef[1,3,jj]=-2*zeta*(-1+4*zeta)
        stf_coef[3,1,jj]=-2*zeta*(-1+4*zeta)
        stf_coef[1,4,jj]=-2*zeta*(1-4*zeta)
        stf_coef[4,1,jj]=-2*zeta*(1-4*zeta)
        stf_coef[1,5,jj]=-2*(-1+2*eta+zeta)*(-1+4*zeta)
        stf_coef[5,1,jj]=-2*(-1+2*eta+zeta)*(-1+4*zeta)
             
        stf_coef[2,2,jj]= 0.
        stf_coef[2,3,jj]=-2*(-1+4*eta)*(-1+eta+2*zeta)
        stf_coef[3,2,jj]=-2*(-1+4*eta)*(-1+eta+2*zeta)
        stf_coef[2,4,jj]=-2*(1-4*eta)*eta
        stf_coef[4,2,jj]=-2*(1-4*eta)*eta
        stf_coef[2,5,jj]=-2*eta*(-1+4*eta)
        stf_coef[5,2,jj]=-2*eta*(-1+4*eta)
        
        stf_coef[3,3,jj]= 16*zeta*(-1+eta+2*zeta)
        stf_coef[3,4,jj]=-8*zeta*(-1+2*eta+2*zeta)
        stf_coef[4,3,jj]=-8*zeta*(-1+2*eta+2*zeta)
        stf_coef[3,5,jj]= 8*(1+eta*(-3+2*eta)-3*zeta+6*eta*zeta+2*zeta**2.)
        stf_coef[5,3,jj]= 8*(1+eta*(-3+2*eta)-3*zeta+6*eta*zeta+2*zeta**2.)
        
        stf_coef[4,4,jj]= 16*eta*zeta
        stf_coef[4,5,jj]=-8*eta*(-1+2*eta+2*zeta)
        stf_coef[5,4,jj]=-8*eta*(-1+2*eta+2*zeta)
        
        stf_coef[5,5,jj]=16*eta*(-1+2*eta+zeta)

    return stf_coef


"""

Convection Matrix coefficient Calculation 

  evaluation of u(a) * (d u(b) /d zeta) : 

first u is polynomial of coffecient a1,a2...a6
second du/dzeta is polynomial of coefficient b1,b2...b6     


1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefvze():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
        
        stf_coef[0,0,jj]=((-1+eta+zeta)*(-1+2*eta+2*zeta)*(-3+4*eta+4*zeta))
        
        stf_coef[0,1,jj]=(-1+eta+zeta)*(-1+2*eta+2*zeta)*(-1+4*zeta)
        stf_coef[1,0,jj]=zeta*(-1+2*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,2,jj]=0.
        stf_coef[2,0,jj]= eta*(-1+2*eta)*(-3+4*eta+4*zeta)
        stf_coef[0,3,jj]=-4*(-1+eta+zeta)*(-1+eta+2*zeta)*(-1+2*eta+2*zeta)
        stf_coef[3,0,jj]=-4*zeta*(-1+eta+zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,4,jj]= 4*eta*(-1+eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[4,0,jj]= 4*eta*zeta*(-3+4*eta+4*zeta)
        stf_coef[0,5,jj]=-4*eta*(-1+eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[5,0,jj]=-4*eta*(-1+eta+zeta)*(-3+4*eta+4*zeta)

        stf_coef[1,1,jj]= zeta*(1-6*zeta+8*zeta**2.)
        
        stf_coef[1,2,jj]= 0.
        stf_coef[2,1,jj]= eta*(-1+2*eta)*(-1+4*zeta)
        stf_coef[1,3,jj]= -4*zeta*(-1+2*zeta)*(-1+eta+2*zeta)
        stf_coef[3,1,jj]= -4*zeta*(-1+eta+zeta)*(-1+4*zeta)
        stf_coef[1,4,jj]= 4*eta*zeta*(-1+2*zeta)
        stf_coef[4,1,jj]= 4*eta*zeta*(-1+4*zeta)
        stf_coef[1,5,jj]= 4*eta*(1-2*zeta)*zeta
        stf_coef[5,1,jj]= -4*eta*(-1+eta+zeta)*(-1+4*zeta)
        
        stf_coef[2,2,jj]= 0.
        
        stf_coef[2,3,jj]= -4*eta*(-1+2*eta)*(-1+eta+2*zeta)
        stf_coef[3,2,jj]= 0.
        stf_coef[2,4,jj]= 4*eta**2.*(-1+2*eta)
        stf_coef[4,2,jj]= 0.
        stf_coef[2,5,jj]= 4*(1-2*eta)*eta**2.
        stf_coef[5,2,jj]= 0.
        
        
        stf_coef[3,3,jj]= 16*zeta*(-1+eta+zeta)*(-1+eta+2*zeta)
        
        stf_coef[3,4,jj]=-16*eta*zeta*(-1+eta+zeta)
        stf_coef[4,3,jj]=-16*eta*zeta*(-1+eta+2*zeta)
        stf_coef[3,5,jj]= 16*eta*zeta*(-1+eta+zeta)
        stf_coef[5,3,jj]= 16*eta*(-1+eta+zeta)*(-1+eta+2*zeta)
        
        stf_coef[4,4,jj]= 16*eta**2.*zeta
        stf_coef[4,5,jj]=-16*eta**2.*zeta
        stf_coef[5,4,jj]=-16*eta**2.*(-1+eta+zeta)
        
        stf_coef[5,5,jj]= 16*eta**2.*(-1+eta+zeta)

    return stf_coef


"""

Convection Matrix coefficient Calculation 

Evaluation of the  u(a) * (d u(b) /d eta) : 
first u is polynomial of coffecient a1,a2...a6
second du/dzeta is polynomial of coefficient b1,b2...b6    
     
1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefve():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
    
        stf_coef[0,0,jj]= (-1+eta+zeta)*(-1+2*eta+2*zeta)*(-3+4*eta+4*zeta)
        
        stf_coef[0,1,jj]= 0.
        stf_coef[1,0,jj]= zeta*(-1+2*zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,2,jj]= (-1+4*eta)*(-1+eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[2,0,jj]= eta*(-1+2*eta)*(-3+4*eta+4*zeta)
        stf_coef[0,3,jj]=-4*zeta*(-1+eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[3,0,jj]=-4*zeta*(-1+eta+zeta)*(-3+4*eta+4*zeta)
        stf_coef[0,4,jj]= 4*zeta*(-1+eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[4,0,jj]= 4*eta*zeta*(-3+4*eta+4*zeta)
        stf_coef[0,5,jj]=-4*(-1+eta+zeta)*(-1+2*eta+zeta)*(-1+2*eta+2*zeta)
        stf_coef[5,0,jj]=-4*eta*(-1+eta+zeta)*(-3+4*eta+4*zeta)

        stf_coef[1,1,jj]= 0.
        
        stf_coef[1,2,jj]= (-1+4*eta)*zeta*(-1+2*zeta)
        stf_coef[2,1,jj]= 0.
        stf_coef[1,3,jj]= 4*(1-2*zeta)*zeta**2.
        stf_coef[3,1,jj]= 0.
        stf_coef[1,4,jj]= 4*zeta**2.*(-1+2*zeta)
        stf_coef[4,1,jj]= 0.
        stf_coef[1,5,jj]=-4*zeta*(-1+2*eta+zeta)*(-1+2*zeta)
        stf_coef[5,1,jj]= 0.
        
        stf_coef[2,2,jj]= eta*(1-6*eta+8*eta**2.)
        
        stf_coef[2,3,jj]= 4*(1-2*eta)*eta*zeta
        stf_coef[3,2,jj]=-4*(-1+4*eta)*zeta*(-1+eta+zeta)
        stf_coef[2,4,jj]= 4*eta*(-1+2*eta)*zeta
        stf_coef[4,2,jj]= 4*eta*(-1+4*eta)*zeta
        stf_coef[2,5,jj]=-4*eta*(-1+2*eta)*(-1+2*eta+zeta)
        stf_coef[5,2,jj]=-4*eta*(-1+4*eta)*(-1+eta+zeta)
        
        
        
        stf_coef[3,3,jj]=16*zeta**2.*(-1+eta+zeta)
        
        stf_coef[3,4,jj]=-16*zeta**2.*(-1+eta+zeta)
        stf_coef[4,3,jj]=-16*eta*zeta**2.
        stf_coef[3,5,jj]= 16*zeta*(-1+eta+zeta)*(-1+2*eta+zeta)
        stf_coef[5,3,jj]= 16*eta*zeta*(-1+eta+zeta)
        
        
        stf_coef[4,4,jj]= 16*eta*zeta**2.
        
        stf_coef[4,5,jj]=-16*eta*zeta*(-1+2*eta+zeta)
        stf_coef[5,4,jj]=-16*eta*zeta*(-1+eta+zeta)
        
        stf_coef[5,5,jj]=16*eta*(-1+eta+zeta)*(-1+2*eta+zeta)

    return stf_coef


"""

Reaction Matrix coefficient Calculation 

Evaluation of the  u(a) * u(b) : 
first u is polynomial of coffecient a1,a2...a6
second du/dzeta is polynomial of coefficient b1,b2...b6    

1. works for order 2 quadratic mesh (6points)
2. Integration order is 5, 7 point integration

"""          

def stm_coefra():

    qintp=quintic_inp()
    evp=np.array(qintp.coords)
    ws=np.shape(evp)[0]
    
    stf_coef=np.zeros((6,6,ws),dtype=float)

    for jj in range(ws):    
        zeta=evp[jj,0]
        eta=evp[jj,1]
        
        P1= 1-3*eta+2*eta**2.+(-3+4*eta)*zeta+2*zeta**2.
        P2= zeta*(-1+2*zeta)
        P3= eta*(-1+2*eta)
        P4=-4*zeta*(-1+eta+zeta)
        P5= 4*eta*zeta
        P6= -4*eta*(-1+eta+zeta)
    
        stf_coef[0,0,jj]= P1*P1
        
        stf_coef[0,1,jj]= P1*P2
        stf_coef[1,0,jj]= P2*P1
        stf_coef[0,2,jj]= P1*P3
        stf_coef[2,0,jj]= P3*P1
        stf_coef[0,3,jj]= P1*P4
        stf_coef[3,0,jj]= P4*P1
        stf_coef[0,4,jj]= P1*P5
        stf_coef[4,0,jj]= P5*P1
        stf_coef[0,5,jj]= P1*P6
        stf_coef[5,0,jj]= P6*P1

        stf_coef[1,1,jj]= P2*P2
        
        stf_coef[1,2,jj]= P2*P3
        stf_coef[2,1,jj]= P3*P2
        stf_coef[1,3,jj]= P2*P4
        stf_coef[3,1,jj]= P4*P2
        stf_coef[1,4,jj]= P2*P5
        stf_coef[4,1,jj]= P5*P2
        stf_coef[1,5,jj]= P2*P6
        stf_coef[5,1,jj]= P6*P2
        
        stf_coef[2,2,jj]= P3*P3
        
        stf_coef[2,3,jj]= P3*P4
        stf_coef[3,2,jj]= P4*P3
        stf_coef[2,4,jj]= P3*P5
        stf_coef[4,2,jj]= P5*P3
        stf_coef[2,5,jj]= P3*P6
        stf_coef[5,2,jj]= P6*P3
        
        
        
        stf_coef[3,3,jj]= P4*P4
        
        stf_coef[3,4,jj]= P4*P5
        stf_coef[4,3,jj]= P5*P4
        stf_coef[3,5,jj]= P4*P6
        stf_coef[5,3,jj]= P6*P4
        
        
        stf_coef[4,4,jj]= P5*P5
        
        stf_coef[4,5,jj]= P5*P6
        stf_coef[5,4,jj]= P6*P5
        
        stf_coef[5,5,jj]= P6*P6

    return stf_coef



