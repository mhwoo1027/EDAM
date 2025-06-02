#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal derivative calcuation from Boundary Integral Equation               

"""

import numpy as np
import ctypes as C

"""

Core function for getting solution of the boundary integral equation (BIE) for normal flux

    input :
        1. prms - derivative related boundary values
        2. mesh - Mathematica type mesh
        3. psim - normalized flux
        4. Fs   - source related function
        
    output :
        1. Uxp,Uzp - first derivative of U
        2. Psix,Psiz - first derivative in x,z direction


    Comments :
        Integrals are given by simple trapezoidal rule because of good convergence
        for periodic functions. The integral equation is Fredholm equation of second
        type which can be solved to very high accurracy. The main difficulty in
        this equation is the singular integral in 2D. 

"""
def BIE_bdry(prms,mesh,psim,FS):
    
    Xpts=prms["Xpts"]
    Zpts=prms["Zpts"]
    PTX=prms["PTX"]
    PTZ=prms["PTZ"]
    Tx1=prms["Tx1"]
    Tx2=prms["Tx2"]
    Tz1=prms["Tz1"]
    Tz2=prms["Tz2"]
    PNX=prms["PNX"]
    PNZ=prms["PNZ"]
    
    Nsize=np.size(Xpts)
    Gn=np.zeros((Nsize,Nsize),dtype=float)
    for ii in range(Nsize):
        for jj in range(Nsize):
            if ii == jj:
                Gn[ii,jj]=-1/(4*np.pi)*(Tz1[ii]*Tx2[ii]-Tx1[ii]*Tz2[ii])/(Tz1[ii]**2.+Tx1[ii]**2.)**1.5
            else:
                Gn[ii,jj]=1/(2*np.pi)*((Xpts[ii]-Xpts[jj])*PNX[ii]+(Zpts[ii]-Zpts[jj])*PNZ[ii])/((Xpts[ii]-Xpts[jj])**2.+(Zpts[ii]-Zpts[jj])**2.)

    beta=np.zeros((Nsize,Nsize),dtype=float)
    for ii in range(Nsize):
        for jj in range(Nsize):
            beta[ii,jj]=2*np.pi/Nsize*np.sqrt(Tx1[ii]**2.+Tz1[ii]**2.)*Gn[jj,ii]
    
    
    print('Starting the integration process for the first derivative')
    
#  Semi - Analytical Integration Method 
    gammai= C_intgal(mesh,prms,psim,FS)
    coeU=1/2*np.identity(Nsize)+beta
    USo=np.dot(np.transpose(np.linalg.inv(coeU)),gammai)

    Uxp=(PTZ*np.transpose(USo))[0]
    Uzp=(-PTX*np.transpose(USo))[0]
    Psix=np.sqrt(Xpts)*Uxp
    Psiz=np.sqrt(Xpts)*Uzp
    
    sol={}
    sol["USo"]=USo
    sol["Uxp"]=Uxp
    sol["Uzp"]=Uzp
    sol["Psix"]=Psix
    sol["Psiz"]=Psiz
    sol["gammai"]=gammai
    sol["beta"]=beta
      
    return sol





"""
Analytical Integraltion of the Singular integrals

For 2D surface integral, we adopt direct integration method. See the documentation
for details
"""

# *******************************************************************************
# A C library generation complie code is given as

# For Mac
# gcc -shared -Wl,-install_name,adder.so -o adder.so -fPIC add.c
#
# For Linux
# gcc -shared -Wl,-soname,adder -o adder.so -fPIC add.c
#
# Then in python use
# from ctypes import *
# #load the shared object file
# adder = CDLL('./adder.so')
#
# gcc -shared -Wl,-install_name,trig_integral.so -o trig_integral.so -fPIC trig_integral.c
# *******************************************************************************


###############################################################################
# A Direct integraltion of 2D surface integral - C code extensions  
###############################################################################

def C_intgal(mesh,prms,psi,FS):
    
################# Array Initialization from C-library function ################

##### Memory postions of mesh_*** and ***_pointer is synchronized ############
    libc = C.CDLL(C.util.find_library('c'))
    libc.malloc.restype = C.c_void_p
    
#   Make a arrays of mesh elements
    SIZE = len(mesh["MeshElements"])

#   Mesh Element store
    ele_pointer = libc.malloc(6*SIZE * C.sizeof(C.c_int))
    ele_pointer = C.cast(ele_pointer,C.POINTER(C.c_int))
    mesh_ele = np.ctypeslib.as_array(ele_pointer,shape=(6*SIZE,))
    
#   Make arrays of coordinates
    SIZE = len(mesh["Coordinates"])

#   Mesh Element store
    cod_pointer = libc.malloc(2*SIZE * C.sizeof(C.c_double))
    cod_pointer = C.cast(cod_pointer,C.POINTER(C.c_double))
    mesh_cod = np.ctypeslib.as_array(cod_pointer,shape=(2*SIZE,))

#   Make arrays of Fi values
    SIZE = len(mesh["Coordinates"])

#   Mesh Element store
    val_pointer = libc.malloc(2*SIZE * C.sizeof(C.c_double))
    val_pointer = C.cast(val_pointer,C.POINTER(C.c_double))
    mesh_val = np.ctypeslib.as_array(val_pointer,shape=(2*SIZE,))

#   Make arrays of some constants
    SIZE=6
    prm_pointer = libc.malloc(SIZE * C.sizeof(C.c_double))
    prm_pointer = C.cast(prm_pointer,C.POINTER(C.c_double))
    mesh_prm = np.ctypeslib.as_array(prm_pointer,shape=(SIZE,))


#   Make a arrays of temporal mesh elements
    SIZE = len(mesh["MeshElements"])

#   Mesh Element store
    temp_pointer = libc.malloc(SIZE * C.sizeof(C.c_double))
    temp_pointer = C.cast(temp_pointer,C.POINTER(C.c_double))
    mesh_temp = np.ctypeslib.as_array(temp_pointer,shape=(SIZE,))

##########################  End of the initialization    ######################

### Assign Static Values

    tmp=mesh["MeshElements"].flatten()
    for ii in range(len(tmp)):
        mesh_ele[ii]=tmp[ii]
        
    tmp=mesh["Coordinates"].flatten()
    for ii in range(len(tmp)):
        mesh_cod[ii]=tmp[ii]

    Fv=FS["FF"](psi,mesh["Coordinates"][:,0])# Assign F(x,u) values at evaluation points
   
    tmp=Fv.flatten()
    for ii in range(len(tmp)):
        mesh_val[ii]=tmp[ii]

    mesh_prm[4]=len(mesh["MeshElements"])*6 # Overall 6 points
    mesh_prm[5]=len(mesh["Coordinates"])*2 # Overall X,Z coordinates

### Assign Dynamic Values and Call C-function
    tig = C.CDLL('./trig_integral.so')
    tig.trig_intg.restype=C.c_double

    Nb=len(prms["Xpts"])
    djj=int(Nb/10) # Calculate Percent !

    
    gamma=np.zeros((Nb,1))
    # For every boundary points
    for ii in range(Nb): 
        nx=prms["PNX"][ii]
        nz=prms["PNZ"][ii]

        jj=ii+1
        kk=ii-1
        if jj == Nb :
            jj=0
        if kk == -1 :
            kk = Nb-1
        
        nx=(prms["PNX"][ii]+prms["PNX"][ii])/2.
        nz=(prms["PNZ"][ii]+prms["PNZ"][ii])/2.
        
        X0=prms["Xpts"][ii]
        Z0=prms["Zpts"][ii]
        mesh_prm[0]=nx
        mesh_prm[1]=nz
        mesh_prm[2]=X0
        mesh_prm[3]=Z0
        gamma[ii]=tig.trig_intg(ele_pointer,cod_pointer,val_pointer,prm_pointer,temp_pointer) # Core C-function
        if ii%djj==0:
                print("Caclulating index...  "+str(ii/Nb*100.) + " %")
    

    del mesh_ele
    libc.free(ele_pointer)
    del mesh_cod
    libc.free(cod_pointer)
    del mesh_val
    libc.free(val_pointer)
    del mesh_prm
    libc.free(prm_pointer)
    del mesh_temp
    libc.free(temp_pointer)

    return gamma


