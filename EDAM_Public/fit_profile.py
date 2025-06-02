#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:31:43 2022

Profile Fitting Program

"""

from scipy.optimize import curve_fit
import numpy as np

"""
Get Profile data from EFIT 
    input :
        gefit (ff' and pp' is derivative with "flux" itself)
        
    output :
        pcoefs : Pressure profile coefficients
        fcoefs : current profile coeffficients
Comments 
    1. First value = Polynomial order, second = coefficients
    2. Profiles has highest order of p=2 
    3. Coefficients normalized such a way that in ff' and pp' is derivative with "normalized" flux
    4. Everything we play with is normailzed flux - no actual flux should be used  
        
"""
def fit_fpsi(gefit):

    dpsi=gefit["sibry"][0]-gefit["simag"][0]
    
    tmpp=fit_profile(gefit["psi_normal"][0],gefit["pprime"][0]) # FF' in terms of the normalized flux
    tmpf=fit_profile(gefit["psi_normal"][0],gefit["ffprime"][0]) # pp' in terms of the normalized flux

# A fitted profile from EFIT
    
#    pcoe=tmpp/dpsi   # Change this notation !!!!!!!!!!!!!!
#    fcoe=tmpf/dpsi # We need dpsi^2 - but we write dpsi becasue prime has already include derivative with dpsi
# Check this with final result
    pcoe=tmpp*dpsi   # Change this notation !!!!!!!!!!!!!!
    fcoe=tmpf*dpsi # We need dpsi^2 - but we write dpsi becasue prime has already include derivative with dpsi

    pcoefs=np.array([[0,pcoe[2]],[1,pcoe[1]],[2,pcoe[0]]])
    fcoefs=np.array([[0,fcoe[2]],[1,fcoe[1]],[2,fcoe[0]]])
    
    res={}
    res["pre_coef"]=pcoefs
    res["cur_coef"]=fcoefs
    
    return res

def fit_profile(x,y):
    
    params, cov = curve_fit(squre_func, x, y)
    
    return params


def squre_func(x,a,b,c):

    return a*x**2+b*x+c


