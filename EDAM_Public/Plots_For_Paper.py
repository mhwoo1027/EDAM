#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:21:35 2023

@author: mhwoo
"""



    mx=None # Maximimu volume
    py_mesh=MeshGen(prms["Xpts"],prms["Zpts"],max_volume=mx,min_angle=34.) # meshpy-type mesh , DO NOT CHANGE min_angle
    mesh_sps=py2maMesh(py_mesh) # meshpy type mesh to Mathematica type mesh

# Plot for mesh and flux values

    fig=plt.figure(figsize=(8,5))
    gs=GridSpec(1,2) # 1 rows, 2 columns
    r0=gefit["r"][0][0]
    r1=gefit["r"][0][-1]
    
    rb=gefit["rbbbs"][0]
    zb=gefit["zbbbs"][0]
    
    ax1=fig.add_subplot(gs[0,0])
    
    ax1.triplot(mesh_sps["Coordinates"][:,0],mesh_sps["Coordinates"][:,1],mesh_sps["MeshElements"][:,0:3]-1,c="black")
    cpts=mesh_sps["Coordinates"][mesh_sps["MeshElements"][:,3:6]-1]

    ax1.set_title("Meshes for GS Equation",fontsize=12)
    ax1.plot(rb,zb,'red')

    rmin=1.3
    rmax=2.3
    
    zmin=-1.
    zmax=0.8

    ax1.set_xlim([rmin,rmax])
    ax2.set_ylim([zmin,zmax])

    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Z(m)")
    ax1.set_aspect("equal")
    xtc=np.linspace(rmin,rmax,5)
    ztc=np.linspace(zmin,zmax,10)
    ax1.set_xticks(xtc)
    ax1.set_yticks(ztc)

    ax2=fig.add_subplot(gs[0,1])
    
    x=mesh["Coordinates"][:,0]
    y=mesh["Coordinates"][:,1]
    triang = mtri.Triangulation(x,y,triangles)
    
    Nlines = 8
    tmp=ax2.tricontourf(triang,tmp_psi,Nlines,cmap="rainbow")
    ax2.tricontour(triang,tmp_psi,Nlines)
    ax2.set_title("Normalized Flux")
    ax2.set_xlabel("X(m)")
    ax2.set_ylabel("Z(m)")
    ax2.set_aspect("equal")
    ax2.set_xticks(xtc)
    ax2.set_yticks(ztc)

    fig.colorbar(tmp)
    
    
    
# Plot for derivatives

    rmin=1.3
    rmax=2.3
    zmin=-1.
    zmax=0.8
    xtc=np.linspace(rmin,rmax,4)
    xtc=np.array([1.3,1.63,1.96,2.3])
    ztc=np.linspace(zmin,zmax,6)
    lsz=8
    Nlines = 12

    fig=plt.figure(figsize=(8,10))
    
    gs=GridSpec(3,3) # 3 rows, 3 columns
    ax1=fig.add_subplot(gs[0,0])
    tmp=ax1.tricontourf(triang,sol["px"],Nlines,cmap="rainbow")
    ax1.tricontour(triang,sol["px"],Nlines)
#    ax1.set_xlabel("X(m)")
#    ax1.set_ylabel("Z(m)")
    ax1.set_aspect("equal")
    ax1.set_xticks(xtc)
    ax1.set_yticks(ztc)
    ax1.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax2=fig.add_subplot(gs[0,1])
    tmp=ax2.tricontourf(triang,sol["pz"],Nlines,cmap="rainbow")
    ax2.tricontour(triang,sol["pz"],Nlines)
#    ax2.set_xlabel("X(m)")
#    ax2.set_ylabel("Z(m)")
    ax2.set_aspect("equal")
    ax2.set_xticks(xtc)
    ax2.set_yticks(ztc)    
    ax2.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax3=fig.add_subplot(gs[0,2])
    tmp=ax3.tricontourf(triang,sol["pxx"],Nlines,cmap="rainbow")
    ax3.tricontour(triang,sol["pxx"],Nlines)
#    ax3.set_xlabel("X(m)")
#    ax3.set_ylabel("Z(m)")
    ax3.set_aspect("equal")
    ax3.set_xticks(xtc)
    ax3.set_yticks(ztc)
    ax3.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax4=fig.add_subplot(gs[1,0])
    tmp=ax4.tricontourf(triang,sol["pxz"],Nlines,cmap="rainbow")
    ax4.tricontour(triang,sol["pxz"],Nlines)
#    ax4.set_xlabel("X(m)")
#    ax4.set_ylabel("Z(m)")
    ax4.set_aspect("equal")
    ax4.set_xticks(xtc)
    ax4.set_yticks(ztc)
    ax4.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax5=fig.add_subplot(gs[1,1])
    tmp=ax5.tricontourf(triang,sol["pzz"],Nlines,cmap="rainbow")
    ax5.tricontour(triang,sol["pzz"],Nlines)
#    ax5.set_xlabel("X(m)")
#    ax5.set_ylabel("Z(m)")
    ax5.set_aspect("equal")
    ax5.set_xticks(xtc)
    ax5.set_yticks(ztc)    
    ax5.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax6=fig.add_subplot(gs[1,2])
    tmp=ax6.tricontourf(triang,sol["pxxx"],Nlines,cmap="rainbow")
    ax6.tricontour(triang,sol["pxxx"],Nlines)
#    ax6.set_xlabel("X(m)")
#    ax6.set_ylabel("Z(m)")
    ax6.set_aspect("equal")
    ax6.set_xticks(xtc)
    ax6.set_yticks(ztc)    
    ax6.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax7=fig.add_subplot(gs[2,0])
    tmp=ax7.tricontourf(triang,sol["pzzz"],Nlines,cmap="rainbow")
    ax7.tricontour(triang,sol["pzzz"],Nlines)
#    ax7.set_xlabel("X(m)")
#    ax7.set_ylabel("Z(m)")
    ax7.set_aspect("equal")
    ax7.set_xticks(xtc)
    ax7.set_yticks(ztc)    
    ax7.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    ax8=fig.add_subplot(gs[2,1])
    tmp=ax8.tricontourf(triang,sol["pxxz"],Nlines,cmap="rainbow")
    ax8.tricontour(triang,sol["pxxz"],Nlines)
#    ax8.set_xlabel("X(m)")
#    ax8.set_ylabel("Z(m)")
    ax8.set_aspect("equal")
    ax8.set_xticks(xtc)
    ax8.set_yticks(ztc)    
    ax8.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
        
    ax9=fig.add_subplot(gs[2,2])
    tmp=ax9.tricontourf(triang,sol["pzzx"],Nlines,cmap="rainbow")
    ax9.tricontour(triang,sol["pzzx"],Nlines)
#    ax9.set_xlabel("X(m)")
#    ax9.set_ylabel("Z(m)")
    ax9.set_aspect("equal")
    ax9.set_xticks(xtc)
    ax9.set_yticks(ztc) 
    ax9.tick_params(axis='both',labelsize=lsz)
    tmc=fig.colorbar(tmp)
    tmc.ax.tick_params(labelsize=lsz)
    
    
    
    ax1.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{x}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax1.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax2.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{z}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax2.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax3.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{xx}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax3.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax4.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{xz}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax4.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax5.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{zz}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax5.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax6.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{xxx}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax6.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax7.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{zzz}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax7.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax8.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{xxz}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax8.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    ax9.text(
    0.98, 0.98,                # x and y (in axes coordinates)
    r"$\psi_{zzx}$",           # LaTeX-style string
    ha="right", va="top",      # Align text to top-right
    transform=ax9.transAxes,   # Use axes-relative coordinates
    fontsize=12               # Adjust size as needed
    )
    
    plt.subplots_adjust(wspace=0.12, hspace=0.2)

    
    
    
    
    
    
    
    