#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

           Equillibrium Derivative For Arbitrary Mesh (EDAM) v1.0


   The Main Purpose the Code is to calculate first, second and thrid spacial 
   derivative of the flux for Grad-Shafranov Equation in arbitrary mesh


Input : 
    1. EFIT equillibrium file (gefit file)
    2. Nsize - number of the points along the boundary
    2. kc - a filtering factor for boundary construction
    3. psie - normalized flux value of the boundary for the calculation regime 
       (Boundary should not include X point)
    
Output : 
    1. vtk-format mesh with solutions at each point
    2. Certain plots for visulatization
    
Main Process of the Code 
    1. From the EFIT define calculation regime
    2. Parameterize the boundary 
    3. Solve GS equation for the flux
    4. First spacial derivative of the flux along the boundary
    5. Second & third spacial derivative along the boundary
    6. 1st, 2nd and 3rd derivatives in entire regime

Some comments :
    1. For the spectral theory, the reference is Trefthen, book
    2. For the PDE solver, it is home-made code
    3. Meshes are 2nd order triangle mesh with 6 points (C0-quadratic mesh)
    4. Region integral is given semi-analytically

Installtion construction
    1. Install key packages : "meshpy", "pyevtk", "ctype","geqdsk_dk" from T.Rhee
    2. run   " gcc -shared -Wl,-install_name,trig_integral.so -o trig_integral.so -fPIC trig_integral.c "
           in terminal to compile the .c file and make library
    3. Just run current file to get the result
    4. Entire input is given in this file "only"

                            author: M.Woo, Korean Fusion Energy Institude (KFE)
                            email : mhwoo@kfe.re.kr
                            
"""

import sys
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec


from geqdsk_dk import geqdsk_dk as geqdsk
from lcfs_trj import *
from spectrDiffMh import *
from fit_profile import *
from scipy import interpolate

from gen_mesh import *

from GS_Solver import *
from BIE_bdry import *

from validation_func import *
from driv_solver import *
from PDE_Solver import *
from sourceF import *


if __name__ == '__main__':

##############################################################################
#         Code inputs, initialzation and Loading G-file                      #
##############################################################################

    fn="g018204.005600"    # Gefit gfile name
    Nsize = 2048           # Number of data at boundary (even number )
    psie = 0.85     # Last normailized flux value
    kc = 50           # Cut-off wave number to smooth out boundary
    
    
# Overall Derivative Solutions
    psd={}
# Loading Gfile
    geqdsks = geqdsk(os.getcwd()+"/"+fn)
    gefit=geqdsks.data

##############################################################################
#      1. Get specified flux surface                                         #
##############################################################################

# Boundary Point extraction
    print("Exatracting boundary from EFIT...")
    lcfstrj=lcfs_trj(Nsize,psie,gefit)

# Get Simulation boundary - filterred (X,Z) position
# After filtering process, exact meanings of theta as the geometrical angles are lost
# because (X,Z) values are changed, theta now plays a rule of variable that prameterizes exact boundary
    pts={}
    pts["Xpts"]=higPssTrip(lcfstrj["r"], kc)
    pts["Zpts"]=higPssTrip(lcfstrj["z"], kc)
    pts["theta"]=lcfstrj["theta"]
    pts["x0"]=lcfstrj["x0"]
    pts["z0"]=lcfstrj["z0"]
    DM=spectDiffMh(Nsize, kc)
    prms=deriv_parameters(DM,pts["Xpts"],pts["Zpts"])
    prms["pts"]=pts


##############################################################################
#      2. Generate Mesh with boundary points                                 #
##############################################################################

    print("Generating Mesh")
    
    cwd=os.getcwd()+'/VTK'
    subprocess.run(['rm','-rf',cwd])
    subprocess.run(['mkdir',cwd])

    mx=1/Nsize # Maximimu volume
    py_mesh=MeshGen(prms["Xpts"],prms["Zpts"],max_volume=mx,min_angle=34.) # meshpy-type mesh , DO NOT CHANGE min_angle
    mesh=py2maMesh(py_mesh) # meshpy type mesh to Mathematica type mesh

##############################################################################
#      2. Fit the profile and get Current related functions                  #
##############################################################################

# Get fitting profile
    print("Fitting the Profile from EFIT...")
    coefs=fit_fpsi(gefit) # psi- normalized fitting
    dpsi=gefit["sibry"][0]-gefit["simag"][0]

# Get Fpsi related functions
    Fs=Fb(coefs)

##############################################################################
#      2. Solve GS equation                                                  #
##############################################################################
    
    print("Solving the GS equation ")
    [sigma,tmp_psi]=GS_Solver(mesh,Fs,gefit,psie) #sigma - coefficient befor right GS equation

##############################################################################
#      3. Derivatives at the boundary from BIE                               #
##############################################################################

    Fps = construct_Fpu(psie,coefs,sigma)
# First derivative at boundary from BIE
    psixz=BIE_bdry(prms,mesh,tmp_psi,Fps)
# Boundary mid-point derivatives
    mid_X=mesh["Coordinates"][mesh["BoundaryElements"][:,2]-1][:,0]
    mid_Z=mesh["Coordinates"][mesh["BoundaryElements"][:,2]-1][:,1]
    mid_prms=deriv_parameters(DM,mid_X,mid_Z)
    psixz_mid=BIE_bdry(mid_prms,mesh,tmp_psi,Fps)


##############################################################################
#      4. 2nd and 3rd derivatives at the boundary
##############################################################################

    dpsixz={}
    dpsixz["dUxp"]=np.dot(DM[0],psixz["Uxp"])
    dpsixz["dUzp"]=np.dot(DM[0],psixz["Uzp"])
    dpsixz_mid={}
    dpsixz_mid["dUxp"]=np.dot(DM[0],psixz_mid["Uxp"])
    dpsixz_mid["dUzp"]=np.dot(DM[0],psixz_mid["Uzp"])
    
    dpsixz=filter_drv(DM,prms,psixz,kc) # Filtered parametric derivatives
    dpsixz_mid=filter_drv(DM,mid_prms,psixz_mid,kc) # Filtered parametric derivatives
    
# Second derivative
    sdm=snd_drv_matrix(DM,prms,psie,dpsixz,Fps)
    sdm_mid=snd_drv_matrix(DM,mid_prms,psie,dpsixz_mid,Fps)
# Third derivative
    tdm=thrd_drv_matrix(DM,prms,psie,dpsixz,psixz,sdm,Fps)
    tdm_mid=thrd_drv_matrix(DM,mid_prms,psie,dpsixz_mid,psixz_mid,sdm_mid,Fps)


##############################################################################
#      5. u - Derivatives everywhere
##############################################################################

# First Derivative
    psixzr=fstdrv_region(psixz,psixz_mid,Fps,mesh,tmp_psi)
# Second Derivative
    sdm_merge={}
    Ns=len(sdm["Uxx"])
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(sdm["Uxx"][ii])
        bc_dirich.append(sdm_mid["Uxx"][ii])
    sdm_merge["Uxx"]=bc_dirich
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(sdm["Uzz"][ii])
        bc_dirich.append(sdm_mid["Uzz"][ii])
    sdm_merge["Uzz"]=bc_dirich
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(sdm["Uxz"][ii])
        bc_dirich.append(sdm_mid["Uxz"][ii])
    sdm_merge["Uxz"]=bc_dirich
    psd={}
    psd["mesh"]=mesh
    psd["psi"]=tmp_psi
    psd["ux"]=psixzr["Uxr"]
    psd["uz"]=psixzr["Uzr"]
    psiscn=scndrv_region(sdm_merge,psd,Fps,mesh)

# Third Derivative
    psd["uxx"]=psiscn["Uxxr"]
    psd["uzz"]=psiscn["Uzzr"]
    psd["uxz"]=psiscn["Uxzr"]

    tdm_merge={}
    Ns=len(tdm["Uxxx"])
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(tdm["Uxxx"][ii])
        bc_dirich.append(tdm_mid["Uxxx"][ii])
    tdm_merge["Uxxx"]=bc_dirich
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(tdm["Uzzz"][ii])
        bc_dirich.append(tdm_mid["Uzzz"][ii])
    tdm_merge["Uzzz"]=bc_dirich
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(tdm["Uxzz"][ii])
        bc_dirich.append(tdm_mid["Uxzz"][ii])
    tdm_merge["Uxzz"]=bc_dirich
    bc_dirich=[]
    for ii in range(Ns):
        bc_dirich.append(tdm["Uxxz"][ii])
        bc_dirich.append(tdm_mid["Uxxz"][ii])
    tdm_merge["Uxxz"]=bc_dirich

    psitdn=tdmdrv_region(tdm_merge,psd,Fps,mesh)

##############################################################################
#      6. u  ----------->  psi derivatives 
##############################################################################

    x=mesh["Coordinates"][:,0]
    z=mesh["Coordinates"][:,1]
    u=1/np.sqrt(x)*(tmp_psi-psie)
    
    ux=psixzr["Uxr"]
    uz=psixzr["Uzr"]
    uxx=psiscn["Uxxr"]
    uzz=psiscn["Uzzr"]
    uxz=psiscn["Uxzr"]    
    uxxx=psitdn["Uxxxr"]
    uzzz=psitdn["Uzzzr"]
    uxzz=psitdn["Uxzzr"]
    uxxz=psitdn["Uxxzr"]
    
    psix=(tmp_psi-psie)/(2*x)+np.sqrt(x)*psixzr["Uxr"]
    psiz=np.sqrt(x)*psixzr["Uzr"]
    pxx=1/(4*x**1.5)*(-u+4*x*ux+4*x**2.*uxx)
    pzz=x**0.5*uzz
    pxz=1/(2*x**0.5)*(uz+2*x*uxz)
    pxxx=1/(8*x**2.5)*(3*u-6*x*ux+12*x**2.*uxx+8*x**3.*uxxx)
    pzzz=np.sqrt(x)*uzzz
    pzzx=1/(2*np.sqrt(x))*(uzz+2*x*uxzz)
    pxxz=1/(4*x**1.5)*(-uz+4*x*uxz+4*x**2.*uxxz)
    # Final Solution
    
    sol={}
    sol["psi"]=tmp_psi
    sol["px"]=psix
    sol["pz"]=psiz
    sol["pxx"]=pxx
    sol["pzz"]=pzz
    sol["pxz"]=pxz
    sol["pxxx"]=pxxx
    sol["pzzz"]=pzzz
    sol["pxxz"]=pxxz
    sol["pzzx"]=pzzx

    

##############################################################################
#      7. save to vtk file (Mid-point data are lost )
##############################################################################

    xp=np.array(py_mesh.points)[:,0]
    zp=np.array(py_mesh.points)[:,1]
    
    psid={}
    TMP=intp_solution(mesh,tmp_psi)
    psid["psi"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,psix)
    psid["px"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,psiz)    
    psid["pz"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pxx)
    psid["pxx"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pzz)
    psid["pzz"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pxz)
    psid["pxz"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pxxx)
    psid["pxxx"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pzzz)
    psid["pzzz"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pzzx)
    psid["pzzx"]=array(TMP(xp,zp),dtype=float)
    TMP=intp_solution(mesh,pxxz)
    psid["pxxz"]=array(TMP(xp,zp),dtype=float)

    fname=cwd+'/'+fn+'.'+str(Nsize)
    MeshToVTK(py_mesh,psid,fname)
    
##############################################################################
#      7. Overal Plots                                                       #
##############################################################################

    fig=plt.figure(figsize=(14,5))
    gs=GridSpec(2,3) # 2 rows, 4 columns
    r0=gefit["r"][0][0]
    r1=gefit["r"][0][-1]
    r0=1.2
    r1=2.3
    z0=gefit["z"][0][0]
    z1=gefit["z"][0][-1]
    z0=-0.8
    z1=0.8
    ax1=fig.add_subplot(gs[:,0])
    ax1.set_xlim([r0,r1])
    ax1.set_ylim([z0,z1])
    ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[1,1])
    ax4=fig.add_subplot(gs[:,2])
    ax4.set_xlim([r0,r1])
    ax4.set_ylim([z0,z1])
    #Plot Mesh
    ax1.triplot(mesh["Coordinates"][:,0],mesh["Coordinates"][:,1],mesh["MeshElements"][:,0:3]-1,c="black")
    cpts=mesh["Coordinates"][mesh["MeshElements"][:,3:6]-1]
#    ax.scatter(cpts[:,:,0],cpts[:,:,1],s=8,marker="x",c="blue")
    bcpts=mesh["Coordinates"][mesh["BoundaryElements"][:,2]-1]
    ax1.scatter(bcpts[:,0],bcpts[:,1],s=10,marker="o",c="red")
    ax1.set_title("Meshes for GS Equation",fontsize=12)
    
    #Plot Profiles
    psin=gefit["psi_normal"][0]
    ax2.plot(Fsi(psin,coefs["cur_coef"],0)*1/dpsi)
    ax2.plot(gefit["ffprime"][0],marker="o")
    ax2.set_title("FFprime")
    ax3.plot(Fsi(psin,coefs["pre_coef"],0)*1/dpsi)
    ax3.plot(gefit["pprime"][0],marker="o")
    ax3.set_title("Pprime")
    #Plot Flux contour
    mpts=mesh['Coordinates']
    triangles=mesh["MeshElements"][:,0:3]-1
    x=mesh["Coordinates"][:,0]
    y=mesh["Coordinates"][:,1]
    triang = mtri.Triangulation(x,y,triangles)
    npsi=np.linspace(0,psie,15)
    ax4.tricontour(triang,psd["psi"],npsi) #- Multiline Plot
    ax4.set_title("Flux Contour")


    Nlines = 12
    
    #First derivative plots
    fig=plt.figure(figsize=(9,4.5))
    gs=GridSpec(1,2)
    r0=1.2
    r1=2.3
    z0=-0.8
    z1=0.8

    ax1=fig.add_subplot(gs[0,0])
    ax1.set_xlim([r0,r1])
    ax1.set_ylim([z0,z1])
    tmp=ax1.tricontourf(triang,sol["px"],Nlines,cmap="rainbow")
    ax1.tricontour(triang,sol["px"],Nlines)
    ax1.set_title("psi_x")
    fig.colorbar(tmp)    
    ax2=fig.add_subplot(gs[0,1])
    ax2.set_xlim([r0,r1])
    ax2.set_ylim([z0,z1])
    tmp=ax2.tricontourf(triang,sol["pz"],Nlines,cmap="rainbow")
    ax2.tricontour(triang,sol["pz"],Nlines)
    ax2.set_title("psi_z")
    fig.colorbar(tmp)
    
    
    #Second derivative plots
    fig=plt.figure(figsize=(14,4.5))
    gs=GridSpec(1,3)
    r0=1.2
    r1=2.3
    z0=-0.8
    z1=0.8

    ax1=fig.add_subplot(gs[0,0])
    ax1.set_xlim([r0,r1])
    ax1.set_ylim([z0,z1])
    tmp=ax1.tricontourf(triang,sol["pxx"],Nlines,cmap="rainbow")
    ax1.tricontour(triang,sol["pxx"],Nlines)
    ax1.set_title("psi_xx")
    fig.colorbar(tmp)    
    ax2=fig.add_subplot(gs[0,1])
    ax2.set_xlim([r0,r1])
    ax2.set_ylim([z0,z1])
    tmp=ax2.tricontourf(triang,sol["pzz"],Nlines,cmap="rainbow")
    ax2.tricontour(triang,sol["pzz"],Nlines)
    ax2.set_title("psi_zz")
    fig.colorbar(tmp)
    ax3=fig.add_subplot(gs[0,2])
    ax3.set_xlim([r0,r1])
    ax3.set_ylim([z0,z1])
    tmp=ax3.tricontourf(triang,sol["pxz"],Nlines,cmap="rainbow")
    ax3.tricontour(triang,sol["pxz"],Nlines)
    ax3.set_title("psi_xz")
    fig.colorbar(tmp)


    #Third derivative plots
    fig=plt.figure(figsize=(8,8))
    gs=GridSpec(2,2)
    r0=1.2
    r1=2.3
    z0=-0.8
    z1=0.8

    ax1=fig.add_subplot(gs[0,0])
    ax1.set_xlim([r0,r1])
    ax1.set_ylim([z0,z1])
    tmp=ax1.tricontourf(triang,sol["pxxx"],Nlines,cmap="rainbow")
    ax1.tricontour(triang,sol["pxxx"],Nlines)
    ax1.set_title("psi_xxx")
    fig.colorbar(tmp)
    ax2=fig.add_subplot(gs[0,1])
    ax2.set_xlim([r0,r1])
    ax2.set_ylim([z0,z1])
    tmp=ax2.tricontourf(triang,sol["pzzz"],Nlines,cmap="rainbow")
    ax2.tricontour(triang,sol["pzzz"],Nlines)
    ax2.set_title("psi_zzz")
    fig.colorbar(tmp)
    ax3=fig.add_subplot(gs[1,0])
    ax3.set_xlim([r0,r1])
    ax3.set_ylim([z0,z1])
    tmp=ax3.tricontourf(triang,sol["pxxz"],Nlines,cmap="rainbow")
    ax3.tricontour(triang,sol["pxxz"],Nlines)
    ax3.set_title("psi_xxz")
    fig.colorbar(tmp)
    ax4=fig.add_subplot(gs[1,1])
    ax4.set_xlim([r0,r1])
    ax4.set_ylim([z0,z1])
    tmp=ax4.tricontourf(triang,sol["pzzx"],Nlines,cmap="rainbow")
    ax4.tricontour(triang,sol["pzzx"],Nlines)
    ax4.set_title("psi_zzx")
    fig.colorbar(tmp)



##############################################################################
#      9. Cross-confirm with EFIT results
##############################################################################

# Compare this with interpolated derivatives
    # tmp=psi_didi(gefit,mesh,psie)
    # tmp2=psi_didi2(gefit,mesh,psie,prms["Xpts"],prms["Zpts"])
    # tmp3=psi_didi3(gefit,mesh,psie,prms["Xpts"],prms["Zpts"])
    # uxrd=tmp["Uxrd"][0:1024]
    # uzrd=tmp["Uzrd"][0:1024]

    # x=mesh["Coordinates"][:,0]
    # z=mesh["Coordinates"][:,1]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.plot_trisurf(x,z, psixzr["Uzr"]-tmp["Uzrd"], cmap='jet')        
    
    # x=mesh["Coordinates"][:,0]
    # z=mesh["Coordinates"][:,1]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.plot_trisurf(x,z, psiscn["Uzzr"]-tmp2["Uzzr"], cmap='jet')

    # x=mesh["Coordinates"][:,0]
    # z=mesh["Coordinates"][:,1]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.plot_trisurf(x,z, psitdn["Uxxxr"]-tmp3["Uxxxr"], cmap='jet')  


    






###############################################################################
#     This is validation plot - you do not need it right now
###############################################################################


#     tmp=psi_didi(gefit,mesh,psie)
#    sdm_efit=psi_didi2(gefit,mesh,psie,prms['Xpts'],prms['Zpts'])
#    tdm_efit=psi_didi3(gefit,mesh,psie,prms['Xpts'],prms['Zpts'])
# #   For plots
#     pts=mesh['Coordinates']
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psixzr["Uxr"], cmap='jet')
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psixzr["Uzr"]-tmp["Uzrd"], cmap='jet')    
    
# #   Interpolate Solutions
#     ux=intp_solution(mesh,psixzr["Uxr"])    
#     uz=intp_solution(mesh,psixzr["Uzr"])
    
#     Fgt= (3/4)*(psi-psie)/(pts[:,0])**2.5+1/(pts[:,0])**0.5*Fps["Fpsi"](psih,pts[:,0],0)

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], sdm_efit["Uxxr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psiscn["Uxzr"]-sdm_efit["Uxzr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psiscn["Uxxr"]-sdm_efit["Uxxr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],Fgt, cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],sdm_efit["Uxxr"]+sdm_efit["Uzzr"]-Fgt, cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],psiscn["Uxxr"]+psiscn["Uzzr"]-Fgt, cmap='jet')

    
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psitdm["Uxxxr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],tdm_efit["Uxxxr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psitdm["Uzzzr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],tdm_efit["Uzzzr"], cmap='jet')


#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psitdm["Uxxzr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],tdm_efit["Uxxzr"], cmap='jet')


#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1], psitdm["Uxzzr"], cmap='jet')

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.plot_trisurf(pts[:,0],pts[:,1],tdm_efit["Uxzzr"]-psitdm["Uxzzr"], cmap='jet')








