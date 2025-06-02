#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Source term calculation - Here we give all the source term required for 
GS equation,1st,2nd and 3rd derivative equation


"""
import numpy as np


"""
Source term for GS equation

    input 
        coefs - fitted coeffecients
    output
        Fpsi - source function with normailzed flux as input

"""
def Fb(coefs):
    
    mu0=4*np.pi*1.e-7
    
# P-prime and FF prime for EFIT 
    pcoefs=coefs["pre_coef"]
    fcoefs=coefs["cur_coef"]
    
# Psi-related values - this should be normalized psi-value
    Fpsi =lambda psin,x,dx : -(mu0*x**2.*Fsi(psin,pcoefs,0)+Fsi(psin,fcoefs,0))

    return Fpsi



"""
Source term for 1st,2nd and 3rd derivative equation

    input :
        1. psie -  boundary normailzed flux
        2. coefs - Coefficients of the source profile
        3. sigma - eigenvalue multiplied by source term
    output :
        1. FF - source function for GS equation with u ( for integral purpose )

"""


def construct_Fpu(psie,coefs,sigma):
        
    mu0=4*np.pi*1.e-7
        
# P-prime and FF prime for EFIT  multiply by sigma
    pcoefs=np.transpose(np.array([coefs["pre_coef"][:,0],sigma*coefs["pre_coef"][:,1]]))
    fcoefs=np.transpose(np.array([coefs["cur_coef"][:,0],sigma*coefs["cur_coef"][:,1]]))

# Source F for u function
    FF = lambda psin,x : 3*(psin-psie)/(4*x**2.5)+x**(-0.5)*(-(mu0*x**2.*Fsi(psin,pcoefs,0)+Fsi(psin,fcoefs,0)))
#  partial F/partial u
    Fu = lambda psin,x : 3/(4*x**2.)-(mu0*x**2.*Fsi(psin,pcoefs,1)+Fsi(psin,fcoefs,1))
# parital F/ partial x
    Fx =  lambda psin,x : 1/(2*x**3.5)*(x**2.*Fsi(psin,fcoefs,0)-3*x**4.*mu0*Fsi(psin,pcoefs,0)+\
                          (psie-psin)*(3+x**2.*Fsi(psin,fcoefs,1)+x**4*mu0*Fsi(psin,pcoefs,1)))
# parital F/ partial z
    Fz = lambda psin,x : 0 # Zero!
# partial^2 F/ partial x^2
    Fxx = lambda psin,x : ((-18+x**2*(-3*Fsi(psin,fcoefs,1)+5*Fsi(psin,pcoefs,1)*x**2.*mu0\
                         -(psie-psin)*(Fsi(psin,fcoefs,2)+Fsi(psin,pcoefs,2)*x**2.*mu0)))*(psie-psin)\
                         -3*x**2*Fsi(psin,fcoefs,0)-3*x**4*mu0*Fsi(psin,pcoefs,0))/(4*x**4.5)
# partial^2 F/ partial u^2
    Fuu = lambda psin,x : -x**0.5*(Fsi(psin,fcoefs,2)+x**2.*mu0*Fsi(psin,pcoefs,2))
    
# partial^2 F/ partial u partial x
    Fux = lambda psin,x : (-3-4*x**4*mu0*Fsi(psin,pcoefs,1)\
                          +x**2*(psie-psin)*(Fsi(psin,fcoefs,2)+Fsi(psin,pcoefs,2)*x**2*mu0))/(2*x**3)

# partial^3 F/  partial x^3
    tmp1 = lambda psin,x : 6*Fsi(psin,fcoefs,2)-6*x**2*mu0*Fsi(psin,pcoefs,2)\
                              -(psin-psie)*(Fsi(psin,fcoefs,3)+x**2.*mu0*Fsi(psin,pcoefs,3))
    tmp = lambda psin,x : -15*Fsi(psin,fcoefs,1)-3*x**2*mu0*Fsi(psin,pcoefs,1)\
                             +(psin-psie)*tmp1(psin,x)
    Fxxx = lambda psin,x : (1/(8*x**5.5))*(15*x**2*Fsi(psin,fcoefs,0)+3*x**4*mu0*Fsi(psin,pcoefs,0)\
                            +(psin-psie)*(-144+x**2*tmp(psin,x)))
# partial^3 F/  partial u^3
    Fuuu = lambda psin,x : -x*(Fsi(psin,fcoefs,3)+x**2.*mu0*Fsi(psin,pcoefs,3))
# partial^3 F/  partial u^2 partial x
    Fuux = lambda psin,x :-1/(2*x**0.5)*(Fsi(psin,fcoefs,2)+5*x**2.*mu0*Fsi(psin,pcoefs,2)\
                        +(psin-psie)*(Fsi(psin,fcoefs,3)+x**2.*mu0*Fsi(psin,pcoefs,3)))
# partial^3 F/  partial u partial x^2
    Fuxx = lambda psin,x : -2*mu0*Fsi(psin,pcoefs,1)+(1/(4*x**4.))*(18+x**2*(psin-psie)*\
                        (Fsi(psin,fcoefs,2)-7*x**2.*mu0*Fsi(psin,pcoefs,2)\
                        -(psin-psie)*(Fsi(psin,fcoefs,3)+x**2*mu0*Fsi(psin,pcoefs,3))))
    res={}
    res["FF"]=FF
    res["FU"]=Fu
    res["FX"]=Fx
    res["FZ"]=Fz
    res["FXX"]=Fxx
    res["FUU"]=Fuu
    res["FUX"]=Fux
    res["FXXX"]=Fxxx
    res["FUXX"]=Fuxx
    res["FUUX"]=Fuux
    res["FUUU"]=Fuuu
        
    return res

"""
F(X,u) x-derivatives
"""
def dFxx(psie,pcoefs,fcoefs,gefit,dx):
    
    dpsi=gefit["sibry"][0]-gefit["simag"][0]
    psi0=gefit["simag"][0]
    
    pppx= lambda psin,x : (psin-psie)/(2*x) # partial psi / partial X with fixed u
    pppx2= lambda psin,x : -(psin-psie)/(4*x**2.) # partial2 psi / partial X2 
    
    mu0=4*np.pi*1.e-7
    if dx ==0 :
        res=lambda psin,x : 3/4*(psin-psie)/(x**0.5)*1/(x**2)\
            -mu0*x**1.5*Fsi(psin,pcoefs,0)-x**(-0.5)*Fsi(psin,fcoefs,0)
    if dx ==1 :    
        res=lambda psin,x : (-2)*3/4*(psin-psie)/(x**0.5)*1/(x**3) \
            -(1.5)*mu0*x**0.5*Fsi(psin,pcoefs,0)-mu0*x**1.5*Fsi(psin,pcoefs,1)*pppx(psin,x)\
                -(-0.5)*x**(-1.5)*Fsi(psin,fcoefs,0)-x**(-0.5)*Fsi(psin,fcoefs,1)*pppx(psin,x)
    if dx ==2 :    
        res=lambda psin,x : (-3)*(-2)*3/4*(psin-psie)/(x**0.5)*1/(x**4) \
            -(0.5)*(1.5)*mu0*x**(-0.5)*Fsi(dpsi*psin+psi0,pcoefs,0)-(1.5)*mu0*x**0.5*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)*pppx(psin,x)\
            -(1.5)*mu0*x**0.5*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)*pppx(psin,x)-mu0*x**1.5*dpsi**2.*Fsi(dpsi*psin+psi0,pcoefs,2)*(pppx(psin,x))**2.-mu0*x**1.5*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)*pppx2(psin,x)\
            -(-1.5)*(-0.5)*x**(-2.5)*Fsi(dpsi*psin+psi0,fcoefs,0)-(-0.5)*x**(-1.5)*dpsi*Fsi(dpsi*psin+psi0,fcoefs,1)*pppx(psin,x)\
            -(-0.5)*x**(-1.5)*dpsi*Fsi(dpsi*psin+psi0,fcoefs,1)*pppx(psin,x)-x**(-0.5)*dpsi**2.*Fsi(dpsi*psin+psi0,fcoefs,2)*(pppx(psin,x))**2.-x**(-0.5)*dpsi*Fsi(dpsi*psin+psi0,fcoefs,1)*pppx2(psin,x)
    
       
    
    
    return res

"""
F(X,u) u-derivatives
"""

def dFuu(psie,pcoefs,fcoefs,sigma,dx):
    
    mu0=4*np.pi*1.e-7
    pu= lambda psin,x : x**0.5 # partial psi / partial u with fixed X
        
    if dx ==1 :
        res=lambda psin,x : 3/4*1/(x**2)\
            -x**(-0.5)*(mu0*x**2.*Fsi(psin,pcoefs,1)+Fsi(psin,fcoefs,1))*pu(psin,x)
    if dx ==2 :
        res=lambda psin,x : -x**(-0.5)*(mu0*x**2.*dpsi**2.*Fsi(dpsi*psin+psi0,pcoefs,2)+dpsi**2.*Fsi(dpsi*psin+psi0,fcoefs,2))*(pu(psin,x))**2.
        
    return res

"""
F(X,u) ux-derivatives
"""

def dFux(psie,pcoefs,fcoefs,gefit,dx):
    
    mu0=4*np.pi*1.e-7
    
    dpsi=gefit["sibry"][0]-gefit["simag"][0]
    psi0=gefit["simag"][0]
    pu= lambda psin,x : x**0.5 # partial psi / partial u with fixed X
    pux= lambda psin,x : (0.5)*x**(-0.5) #partial^2 psi /partial u partial X
    pppx= lambda psin,x : (psin-psie)/(2*x)
    
    if dx ==1 :
        res=lambda psin,x : (-2)*(3/4)*1/(x**3.)\
         -(1.5)*mu0*x**0.5*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)*pu(psin,x)-mu0*x**1.5*dpsi**2.*Fsi(dpsi*psin+psi0,pcoefs,2)*pppx(psin,x)*pu(psin,x)-mu0*x**1.5*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)*pux(psin,x)\
         -(-0.5)*x**(-0.5)*dpsi*Fsi(dpsi*psin+psi0,fcoefs,1)*pu(psin,x)-x**(-0.5)*dpsi**2.*Fsi(dpsi*psin+psi0,fcoefs,2)*pppx(psin,x)*pu(psin,x)-x**(-0.5)*dpsi*Fsi(dpsi*psin+psi0,fcoefs,1)*pux(psin,x)
        
        res= lambda psin,x : (-3-4*x**4.*mu0*dpsi*Fsi(dpsi*psin+psi0,pcoefs,1)-x**2.\
                             *(psin-psie)*dpsi**2.*(x**2.*mu0*Fsi(dpsi*psin+psi0,pcoefs,2)+Fsi(dpsi*psin+psi0,fcoefs,2)))/(2*x**3.)
    return res


"""
Modeling coefficients - it should give zero for constant
"""
def Fsi(psin,cf,dx):
        
    if np.size(psin)>1 :
        if min(psin) == 0. : psin[np.argmin(psin)]=1.e-10

    Ns=len(cf)
    sol=0        
    for ii in range(Ns):
        if dx == 0:
            sol=sol+psin**cf[ii][0]*cf[ii][1]
        if dx == 1:
            sol=sol+cf[ii][0]*psin**(cf[ii][0]-1)*cf[ii][1]
        if dx == 2:
            sol=sol+cf[ii][0]*(cf[ii][0]-1)*psin**(cf[ii][0]-2)*cf[ii][1]            
        if dx == 3:
            sol=sol+cf[ii][0]*(cf[ii][0]-1)*(cf[ii][0]-2)*psin**(cf[ii][0]-3)*cf[ii][1]            

    return sol


"""
Third derivative of Function
""" 
def dFxxx(psie,pcoefs,fcoefs,gefit,fpe):
     
    mu0=4*np.pi*1.e-7
    
    dpsi=gefit["sibry"][0]-gefit["simag"][0]
    psi0=gefit["simag"][0]    
    
    psiv = lambda psin : dpsi*psin+psi0
        
    if fpe == "xxx" :

        tmp1= lambda psin,x : 6*dpsi**2.*Fsi(psiv(psin),fcoefs,2)-6*dpsi**2.*x**2*mu0*Fsi(psiv(psin),pcoefs,2)\
                              -(psin-psie)*(dpsi**3*Fsi(psiv(psin),fcoefs,3)+x**2.*dpsi**3*mu0*Fsi(psiv(psin),pcoefs,3))
        tmp=lambda psin,x : -15*dpsi*Fsi(psiv(psin),fcoefs,1)-3*dpsi*x**2*mu0*Fsi(psiv(psin),pcoefs,1)\
                             +(psin-psie)*tmp1(psin,x)        
        res=lambda psin,x : (1/(8*x**5.5))*(15*x**2*Fsi(psiv(psin),fcoefs,0)+3*x**4*mu0*Fsi(psiv(psin),pcoefs,0)\
                            +(psin-psie)*(-144+x**2*tmp(psin,x)))

    if fpe == "uuu" :
        
        res=lambda psin,x : -x*dpsi**3.*(Fsi(psiv(psin),fcoefs,3)+x**2.*mu0*Fsi(psiv(psin),pcoefs,3))
            
    if fpe == "uux" :
        
        res=lambda psin,x :-1/(2*x**0.5)*(dpsi**2*Fsi(psiv(psin),fcoefs,2)+5*x**2.*dpsi**2*mu0*Fsi(psiv(psin),pcoefs,2)\
                            +(psin-psie)*dpsi**3.*(Fsi(psiv(psin),fcoefs,3)+x**2.*mu0*Fsi(psiv(psin),pcoefs,3)))

    if fpe == "uxx" :

        res=lambda psin,x : -2*mu0*dpsi*Fsi(psiv(psin),pcoefs,1)+(1/(4*x**4.))*(18+x**2*(psin-psie)*\
                            (dpsi**2.*Fsi(psiv(psin),fcoefs,2)-7*x**2.*mu0*dpsi**2.*Fsi(psiv(psin),pcoefs,2)\
                            -(psin-psie)*dpsi**3.*(Fsi(psiv(psin),fcoefs,3)+x**2*mu0*Fsi(psiv(psin),pcoefs,3))))
            
          
    return res 


