#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:18:00 2021

@author: mhwoo
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import broyden1
from functools import partial


"""

 Generates flux surface line for given normalized flux pies
 
   Method : In this function, we calculate
            1. For every given angle theta, we draw a straight line starting from magnetic axis
            2. Along the line, we find a point whose flux is equal to last flux "psie"
            3. When such points are drawn for every theta, these points constitute boundary of our calculation
 
 
   input : Nsize - Number of equally spaced angles
           pie  - Normalized last flux where flux curve should be drawn
           gefit - gefit structure from the geqdsk file

    output : theta - equally spaced angle with starting dtheta (dtheta,2*pi) 
             r - r point of the boundary for given theta
             z - z point of the boundary

"""
def lcfs_trj(Nsize,psie,gefit):
    
    r=gefit['r'][0]
    z=gefit['z'][0]
    r0=gefit['rmaxis'][0]
    z0=gefit['zmaxis'][0]
    psin=(gefit['psirz'][0]-gefit['simag'][0])/(gefit['sibry'][0]-gefit['simag'][0])
    psi_intp=interpolate.RectBivariateSpline(z,r,psin)
    
    theta=np.linspace(2*np.pi/Nsize,2*np.pi,Nsize)
    rs=[]
    zs=[]
    res={}
    
    for thes in theta : # For every theta
        vals=[r0,z0,thes,psie,psi_intp]
        psivp=partial(psiv,vals)
        t0=0. #Initial guess
        tsol=broyden1(psivp,t0,f_tol=1.e-7)    

        [rtemp,ztemp]=stline(r0,z0,thes,tsol)
        rs.append(rtemp)
        zs.append(ztemp)
    res["r"]=rs
    res["z"]=zs
    res["theta"]=theta.tolist()
    res["x0"]=r0
    res["z0"]=z0
    res["comment"]="Flux surface for psi = "+str(psie)
    return res

"""
    Psi difference 
        input : 
                vals - straight line related variables
                x    - iteration parameter
        output :
                psie-psi_inpt - difference between last flux and flux at current point

"""
def psiv(vals,x):
    
    r0=vals[0]
    z0=vals[1]
    theta=vals[2]
    psie=vals[3]
    psi_inpt=vals[4]
    
    [rt,zt]=stline(r0,z0,theta,x)
    
    return psie - psi_inpt(zt,rt)

"""
    Straight line coordinate
    
        input : (r0,z0) - the origin of the straght line
                theta - corresponding angle
                t - length of the straight line
        output : (rt,zt) - coordinate of the straight line
"""
def stline(r0,z0,theta,t):
    
    rt=r0+np.cos(theta)*abs(t)
    zt=z0+np.sin(theta)*abs(t)
    
    return [rt,zt]

    