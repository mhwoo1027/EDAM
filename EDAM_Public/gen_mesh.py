#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:00:08 2023

Mesh Generation Algorithm for GS equation (Single Boundary)

   Needs "meshpy","meshio","pyevtk" package
   
   The Mesh is given in terms of the Mathematica-Style , all the original
   codes had used FEM and Mesh style from Mathematica (first index is 1, not zero )

@author: mhwoo
    Last update in EDAM code in 2023.04.14
    
"""

import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from scipy import interpolate
from functools import partial
from copy import deepcopy
import matplotlib.tri as mtri


from pyevtk.hl import unstructuredGridToVTK
from pyevtk import vtk
import meshio

"""

Mesh Generation algorithm using meshpy (Automatic Mesh Generation)
       Input :
           1. (r,z) points of the boundary
           
       Output : 
           1. meshpy type mesh  
           2. vtk file -  visualized by ParaView

"""

def MeshGen(r,z,**kwargs):
    pts_bdry=[]
    pts_bdry.extend((r[ii],z[ii]) for ii in range(len(r)))

    n=len(pts_bdry)
    BD1=np.zeros((n,2))   
    for ii in range(n):        
        BD1[ii,:]=pts_bdry[ii]

    n=len(BD1)
    facets = round_trip_connect(0,len(BD1)-1)
    points=[]
    points.extend((BD1[ii,0],BD1[ii,1]) for ii in range(n) )

    info = triangle.MeshInfo()            
    info.set_points(points)
    info.set_facets(facets)

    mv=None
    mg=34.
    if kwargs!= {}:
        if(list(kwargs.keys())[0]=='max_volume'):
            mv=kwargs["max_volume"]
        if(list(kwargs.keys())[1]=='min_angle'):
            mg=kwargs["min_angle"]
            
    print("Maximum Volume is ",mv)
    print("Minimum angle is",mg)
    
    mesh = triangle.build(info,max_volume=mv,min_angle=mg)
         
    return mesh

###############################################################################
# Save mesh to VTK format - check the mesh using ParaView
#
# # An anoying example from the internet - original code does not work
# import numpy as np
# from pyevtk.hl import unstructuredGridToVTK
# from pyevtk import vtk
# print("Running unstructured...")
# # Define vertices
# x = np.zeros(5)
# y = np.zeros(5)
# z = np.zeros(5)
# x[0], y[0], z[0] = 0.0, 0.0, 0.0
# x[1], y[1], z[1] = 1.0, 0.0, 0.0
# x[2], y[2], z[2] = 0.0, 1.0, 0.0
# x[3], y[3], z[3] = 1.0, 1.0, 0.0
# x[4], y[4], z[4] = 2.0, 1.0, 0.0
# # Define connectivity or vertices that belongs to each element
# conn = np.zeros(9)
# conn[0], conn[1], conn[2] = 0, 1, 2              # first triangle
# conn[3], conn[4], conn[5] = 2, 1, 3              # second triangle
# conn[6], conn[7], conn[8] = 3, 1, 4    # rectangle
# # Define offset of last vertex of each element
# offset = np.zeros(3)
# offset[0] = 3
# offset[1] = 6
# offset[2] = 9
# # Define cell types
# ctype = np.zeros(3)
# ctype[0],ctype[1],ctype[2] = vtk.VtkTriangle.tid, vtk.VtkTriangle.tid,vtk.VtkTriangle.tid    
# cd = np.zeros(3)
# cd[0]=1
# cellData = {"pressure" : cd}    
# pd = np.zeros(5)
# pointData = {"ec" : pd}
# pd[3]=100
# pd[2]=-100
# comments = [ "comment 1", "comment 2" ]
# unstructuredGridToVTK('/Users/mhwoo/example', x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, cellData = cellData, pointData = pointData)
###############################################################################
def MeshToVTK(mesh,psid,fname):
#    fname='/Users/mhwoo/example'
    #points
    pts=np.array(mesh.points)
    x=deepcopy(pts[:,0])
    y=deepcopy(pts[:,1])
    z=np.zeros(len(pts))
    # Connectivity
    cnv=np.array(mesh.elements)
    nm=len(cnv)
    con=cnv.flatten()
    # Offset (Last vertex )
    offset=np.array([ii*3 for ii in range(1,nm+1)])    
    # Cell type
    ctype=np.zeros(nm)
    ctype[:]=vtk.VtkTriangle.tid
    # Cell data 
    mfct=np.array(mesh.facets)+1 # Mathematica notation
    
    md1 = np.zeros(nm)
    md1[0:len(mfct)]=mfct[:,0]
    md2 = np.zeros(nm)
    md2[0:len(mfct)]=mfct[:,1]
  
    
    x0=x[cnv][:,0]
    x1=x[cnv][:,1]
    x2=x[cnv][:,2]
    y0=y[cnv][:,0]
    y1=y[cnv][:,1]
    y2=y[cnv][:,2]

    Ak=(x0*(y1-y2)+x1*(y2-y0)+x2*(y0-y1))/2
    
    cellData={}
    cellData["m_facet1"]= md1
    cellData["m_facet2"]= md2
    cellData["Area"]=Ak
    
    # Point data - Boundary marker
    pd = np.zeros(len(x))
    bde=np.unique(np.array(mesh.facets).flatten())
    pd[bde]=int(1)
    pointData={}
    pointData["BoundaryMarker"] = pd
    pointData["psi"]=psid["psi"]
    pointData["psi_x"]=psid["px"]    
    pointData["psi_z"]=psid["pz"]    
    pointData["psi_xx"]=psid["pxx"]
    pointData["psi_zz"]=psid["pzz"]
    pointData["psi_xz"]=psid["pxz"]
    pointData["psi_xxx"]=psid["pxxx"]
    pointData["psi_zzz"]=psid["pzzz"]
    pointData["psi_zzx"]=psid["pzzx"]
    pointData["psi_xxz"]=psid["pxxz"]
    
    # Final construction of the VTK file
    unstructuredGridToVTK(fname,x,y,z,connectivity=con,offsets=offset,cell_types=ctype,cellData=cellData,pointData=pointData)
    
    return



###############################################################################
#       A new meshio version of writing and reading mesh through vtk file
# import meshio
# # two triangles and one quad
# points = [[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0],[2.0, 0.0],[2.0, 1.0]]
# cells = [("triangle", [[0, 1, 2], [1, 3, 2]]),("quad", [[1, 4, 5, 3]])]
# mesh = meshio.Mesh(
#     points,
#     cells,
#     # Optionally provide extra data on points, cells, etc.
#     point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
#     # Each item in cell data must match the cells array
#     cell_data={"a": [[0.1, 0.2], [0.4]]},
# )
# mesh.write("foo.vtk")
###############################################################################

def MeshRead(filename):

    tmp=meshio.read(filename)
    t1=tmp.points[:,0]
    t2=tmp.points[:,1]
    mesh={}
    mesh["Coordinates"]=np.array(np.transpose([t1,t2]))
    mesh["Connectivity"]=tmp.cells[0].data+1 # Again anoying Mathematica Notation
    mesh["CellData"]=tmp.cell_data
    
    t1=np.array(mesh["CellData"]["m_facet1"]) # This is Mathematica notation
    t2=np.array(mesh["CellData"]["m_facet2"])
    t3=np.array(np.transpose([t1.flatten(),t2.flatten()]))
    didx=np.where(t3[:,0]>0)
    BDE=np.array(np.transpose([t3[didx,0].flatten(),t3[didx,1].flatten()]),dtype=int)
    
    mesh["BoundaryElements"]=BDE
    
    mesh["PointData"]=tmp.point_data
    #Find Boundary mardker & Construct boundary elements
    bdm=mesh["PointData"]["BoundaryMarker"].flatten()
    bidx=np.where(bdm==1)[0]
    tmp=round_trip_connect(bidx[0],bidx[-1])
    mesh["BoundaryNodes"]=np.array(tmp)+1
    mesh["MeshArea"]=np.array(mesh["CellData"]["Area"])
    
    return mesh

#################################################################################
# Conversion from meshpy type mesh to Mathematica-type mesh
#################################################################################

def py2maMesh(pmesh):
        
    mesh={}
    mesh["Coordinates"]=np.array(pmesh.points)
    mesh["Connectivity"]=np.array(pmesh.elements)+1
    mesh["BoundaryElements"]=np.array(pmesh.facets)+1
    res=MeshExtend(mesh)
    
    return res

#################################################################################
# Extend given mesh to second order - adding mid point in triangles
#     The input is Mathematica-type mesh
#     The ouput is conventional mesh used by Woo's Code (still Mathematica type mesh)
#################################################################################
def MeshExtend(mesh):
        
    mpts=mesh["Coordinates"]
    mele=mesh["Connectivity"]-1 # From Mathematica --> python index
    bele=mesh["BoundaryElements"]-1 # From Mathematica --> python index
    
# To calculate meshes of order 2 which needs centrual points
# Mesh Element Generation
    enumb=len(mele)
    lelem=tri2line(mele) # Get line elements from the triangle elements 
    cpts=(mpts[lelem[:,0]]+mpts[lelem[:,1]])/2
    
    mpts_al=np.append(mpts,cpts,axis=0)
    mele_al=np.zeros((enumb,6),int) # Entire mesh elements
    
    for ii in range(enumb):
        mele_al[ii,0]=mele[ii][0]
        mele_al[ii,1]=mele[ii][1]
        mele_al[ii,2]=mele[ii][2]
                
        tline=[min(mele[ii][0],mele[ii][1]),max(mele[ii][0],mele[ii][1])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,3]=get_idx_pos(mpts_al,ctmp)


        tline=[min(mele[ii][1],mele[ii][2]),max(mele[ii][1],mele[ii][2])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,4]=get_idx_pos(mpts_al,ctmp)

        tline=[min(mele[ii][2],mele[ii][0]),max(mele[ii][2],mele[ii][0])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,5]=get_idx_pos(mpts_al,ctmp)

# Boundary Element Generation
    bele_al=np.zeros((len(bele),3),int)
    for ii in range(len(bele)):        
        tmp=(mpts[bele[ii]][0]+mpts[bele[ii]][1])/2
        cidx=get_idx_pos(mpts_al,tmp)
        bele_al[ii,0]=min(bele[ii][0],bele[ii][1])
        bele_al[ii,1]=max(bele[ii][0],bele[ii][1])
        bele_al[ii,2]=cidx    

# Boundary Element with same order of the Xpts
    TMP=bele_al+1
    TMP2=np.transpose(np.array([TMP[:,0],TMP[:,2],TMP[:,1]]))
    TMP2[-1][0]=TMP2[-1][-1] # Last term correction
    res=[]
    for tmps in TMP2 :
        res.extend(tmps[0:2])    
    BDelm=np.array(res)

    res={}
#    v1=mpts_al+np.array([X0,Z0])
    v1=mpts_al
    v2=mele_al+1 # All element has +1 index (again anoying Mathematica Notation! )
    v3=bele_al+1
    v4=BDelm
    res={"Coordinates":v1,"MeshElements" :v2,"BoundaryElements":v3,"BoundaryNodes":v4}
    res["MeshElementCooments"]="First three is triangel point, next three is centrual point"
    res["BoundaryElementCooments"]="First two is boundary point element, final one centrual point, \
         Boundary Nodes are point nodes along the Xpts "
    
    return res

#################################################################################
# This mesh generation algorithm can control denseness of the mesh at some points
#################################################################################
def gen_mesh(pts,cc,**kwargs):

    X0=(min(pts["Xpts"])+max(pts["Xpts"]))/2
    Z0=(min(pts["Zpts"])+max(pts["Zpts"]))/2
    
    pts_bdry=[]
    pts_bdry.extend((pts["Xpts"][ii]-X0,pts["Zpts"][ii]-Z0) for ii in range(len(pts["Xpts"])) )

    n=len(pts_bdry)
    BD1=np.zeros((n,2))   
    for ii in range(n):        
        BD1[ii,:]=pts_bdry[ii]

    n=len(BD1)
    facets = round_trip_connect(0,len(BD1)-1)
    points=[]
    points.extend((BD1[ii,0],BD1[ii,1]) for ii in range(n) )

    info = triangle.MeshInfo()            
    info.set_points(points)
    info.set_facets(facets)


    mv=None
    mg=34.
    if kwargs!= {}:
        if(list(kwargs.keys())[0]=='max_volume'):
            mv=kwargs["max_volume"]
        if(list(kwargs.keys())[1]=='min_angle'):
            mg=kwargs["min_angle"]
            
    print("Maximum Volume is ",mv)
    print("Minimum angle is",mg)

    # Find interpolation funrction for the lcfs 

    rv=np.transpose(np.array([pts["Xpts"]-X0,pts["Zpts"]-Z0]))
    dis_r=np.array([la.norm(r) for r in rv])
    dis_r[-1]=dis_r[0]
    rtht=interpolate.CubicSpline(pts["theta"],dis_r,bc_type='periodic')

    vals=[rtht,[cc[0],cc[1],cc[2]]]
    
    needs_refinement=partial(needs_ref,vals)
    mesh = triangle.build(info,refinement_func=needs_refinement,max_volume=mv,min_angle=mg)
    mpts=np.array(mesh.points)
    mele=np.array(mesh.elements)
    
# By default, we wish to calculate meshes of order 2 which needs centrual points
# Mesh Element Generation
    enumb=len(mele)
    lelem=tri2line(mele) # Get line elements from the triangle elements 
    cpts=(mpts[lelem[:,0]]+mpts[lelem[:,1]])/2
    
    mpts_al=np.append(mpts,cpts,axis=0)
    mele_al=np.zeros((enumb,6),int) # Entire mesh elements
    
    for ii in range(enumb):
        mele_al[ii,0]=mele[ii][0]
        mele_al[ii,1]=mele[ii][1]
        mele_al[ii,2]=mele[ii][2]
                
        tline=[min(mele[ii][0],mele[ii][1]),max(mele[ii][0],mele[ii][1])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,3]=get_idx_pos(mpts_al,ctmp)


        tline=[min(mele[ii][1],mele[ii][2]),max(mele[ii][1],mele[ii][2])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,4]=get_idx_pos(mpts_al,ctmp)

        tline=[min(mele[ii][2],mele[ii][0]),max(mele[ii][2],mele[ii][0])]
        ctmp=cpts[get_idx_pos(lelem,tline)]
        mele_al[ii,5]=get_idx_pos(mpts_al,ctmp)

# Boundary Element Generation
    bele=np.array(mesh.facets) # Boundary line element
    bele_al=np.zeros((len(bele),3),int)
    for ii in range(len(bele)):        
        tmp=(mpts[bele[ii]][0]+mpts[bele[ii]][1])/2
        cidx=get_idx_pos(mpts_al,tmp)
        bele_al[ii,0]=min(bele[ii][0],bele[ii][1])
        bele_al[ii,1]=max(bele[ii][0],bele[ii][1])
        bele_al[ii,2]=cidx    

# Boundary Element with same order of the Xpts
    TMP=bele_al+1
    TMP2=np.transpose(np.array([TMP[:,0],TMP[:,2],TMP[:,1]]))
    TMP2[-1][0]=TMP2[-1][-1] # Last term correction
    res=[]
    for tmps in TMP2 :
        res.extend(tmps[0:2])    
    BDelm=np.array(res)

    res={}
    v1=mpts_al+np.array([X0,Z0])
    v2=mele_al+1 # All element has +1 index (again anoying Mathematica Notation! )
    v3=bele_al+1
    v4=BDelm
    res={"Coordinates":v1,"MeshElements" :v2,"BoundaryElements":v3,"BoundaryNodes":v4}
    res["MeshElementCooments"]="First three is triangel point, next three is centrual point"
    res["BoundaryElementCooments"]="First two is boundary point element, final one centrual point, \
         Boundary Nodes are point nodes along the Xpts "
         
    return res

def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def needs_ref(vals,vertices, area):
   
    rth=vals[0] # Interpolated LCFS vs theta
    cc=vals[1]
    bary=np.sum(np.array(vertices),axis=0)/3 # Generated mesh center
    nv=bary # Shifted center
    ang=np.arctan2(nv[1],nv[0]) # Angle of the mesh center
    dis_r=la.norm(nv) # Distance from center
    dis_l=rth(ang) # Distance to the LCFS
    
    max_area=cc[2] +cc[1]*np.abs(dis_l-dis_r)**cc[0]    

    return bool(area > max_area)

"""
Return the index for given target from srcs
"""
def get_idx_pos(srcs,target):
    tmp=np.argwhere(srcs==target)[:,0]    
    ids=[]            
    for tp in tmp:
        vals=np.where(tmp==tp)
        if(np.size(vals)==2):
            ids.extend(vals[0])
    res=np.unique(tmp[ids])

    try :
        return res[0]
    except:
        return -1
        
"""
Extract Line Elements from the Triangular Elements
"""
def tri2line(mele): 
    enumb=np.shape(mele)[0] # Every triangle index
    midx_ex=np.zeros((enumb,3,2),int)
    for ii in range(enumb):
        midx_ex[ii,0,0]=min(mele[ii][0],mele[ii][1]) # In mesh-line code, smaller components come first
        midx_ex[ii,0,1]=max(mele[ii][0],mele[ii][1]) 
        midx_ex[ii,1,0]=min(mele[ii][1],mele[ii][2])
        midx_ex[ii,1,1]=max(mele[ii][1],mele[ii][2])
        midx_ex[ii,2,0]=min(mele[ii][2],mele[ii][0])
        midx_ex[ii,2,1]=max(mele[ii][2],mele[ii][0])
        
    rsmidx=midx_ex.reshape(-1,np.shape(midx_ex)[-1]) # flattened line element of each triangle

    res=np.unique(rsmidx,axis=0) # delete duplicate 

    return res

"""
Interpolate solutions of the PDE equation to given mesh - linear interpolation
"""
def intp_solution(mesh,sol):
        
    triangles=mesh["MeshElements"][:,0:3]-1
    x=mesh["Coordinates"][:,0]
    y=mesh["Coordinates"][:,1]
    triang = mtri.Triangulation(x,y,triangles)

    res=mtri.LinearTriInterpolator(triang,sol)
    
    
    return res



















