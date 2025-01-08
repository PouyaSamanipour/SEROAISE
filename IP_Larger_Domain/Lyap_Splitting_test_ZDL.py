# this is new type of splitting. In this method first we are obtaining all the  new points that we need to add to 
# our dynamic. The next step is that we have to check that these new points will belong to which regions. As a result
# we will know that for each region which points hould be used to split it. Therefore, we do not need to be worry about 
# neighborhood. 
from copy import deepcopy
from hashlib import new
from random import randint
import numpy as np
import math
# import Origin_detector
# from GetSysData import GetSysData
from scipy.spatial import Delaunay
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from itertools import combinations
from numpy.linalg import norm
#from test_convex import in_hull
def splitting_cell_prep(sol,V,n_r,id,H,A_dyn,cell_info,id_var,neighbor_info,Th,iter):
    print("start splitting...")
    n=np.size(V[0],1)
    new_point=np.zeros(n)
    nvar=np.size(sol)
    # separating the slack variables
    sol_n=sol[nvar-n_r:nvar]
    sol_n=np.array(sol_n)
    # keep the indices of the slack variable(ascending)
    for i in range(id_var.shape[0]):
        if id_var[i][1]-id_var[i][0]<n+1:
            sol=np.insert(sol,(n+1)*(i+1)-1,0)
    sol_org=np.reshape(sol[:np.size(sol)-n_r],(n_r,n+1))
    indices=np.argsort(sol_n)
    cell_info[:]=[cell_info[i] for i in indices]# including info about whether it has the origin or not
    indices_split=[indices[i] for i in indices if sol_n[indices[i]]>Th]
    print("Number of region with slack variables=",len(indices_split))
    new_points,new_point_id,side_inf=new_point_calc(V,H,A_dyn,indices_split,iter)
    splitting_reg=reg_calc(V,new_points,indices_split,neighbor_info,new_point_id,side_inf)
    V_ref=V.copy()
    A_ref=A_dyn.copy()
    H_ref=H.copy()
    V_splitted=[]
    A_splitted=[]
    H_splitted=[]
    cell_info_cp=[]
    ctr=0
    for j,i in enumerate(splitting_reg):
        new_point=[new_points[k] for k in i]
        if len(new_point)!=0:
            V_new,A_new,H_new=splitting_cells(V[j],H[j],A_dyn[j],new_point)
            V_splitted.extend(V_new)
            A_splitted.extend(A_new)
            H_splitted.extend(H_new)
            if bool(i):
                #print(V[j]==V_ref[ctr])
                del V_ref[ctr]
                del H_ref[ctr]
                del A_ref[ctr]
        else:
            ctr=ctr+1       
    V_ref.extend(V_splitted)
    A_ref.extend(A_splitted)
    H_ref.extend(H_splitted)
    # Find all the region with these new points
    return V_ref,A_ref,H_ref
    








def new_point_calc(V,H,A_dyn,indices_split,iter):
    new_point=[]
    list_test=[]
    new_point_id=[]
    buffer=[]
    side_info=[]
    for i in indices_split:
        n_v,n=np.shape(V[i])
        vertex_new=[i for i in V[i].tolist() if i!=[0]*n]            
        der=np.array(np.matmul(A_dyn[i],np.array(vertex_new).T)+np.reshape(H[i],(len(H[i]),1)))
        der=der.T
        if n_v==n+1:
            n_der=len(vertex_new)
            list_ver=[i for i in range(n_der)]
            comb = combinations(list_ver, 2)
            new_points_list,side=newpoints_id(vertex_new,comb,der,iter)
            if list(new_points_list[0]) not in buffer:     
                list_test.extend((new_points_list))
            #new_point.append(new_points)
                new_point_id.extend(len(new_points_list)*[i])
                side_info.extend(side)
                buffer.append(list(new_points_list[0]))            
        else:
            hull=ConvexHull(V[i]).simplices
            new_points_list,side=newpoints_id(vertex_new,hull,der,iter)
            #new_point.append(new_points)
            if list(new_points_list[0]) not in buffer:     
                list_test.extend((new_points_list))
            #new_point.append(new_points)
                new_point_id.extend(len(new_points_list)*[i])
                side_info.extend(side)
                buffer.append(list(new_points_list[0]))  
    return list_test,new_point_id,side_info
def reg_calc(V,list_test,indices_split,neighbor_info,new_point_id,side_inf):
    reg_calc=[]
    reg_calc=[[] for i in range(len(V))]
    for k,j in  enumerate(list_test):
        if len(j)!=0:
            adj_reg=neighbor_info[new_point_id[k]]
            hull=[Delaunay(V[i]) for i in adj_reg]
            sol_stat=[i.find_simplex(j)>=0 for i in hull]
            # # l=sol_stat.index(True)
            [reg_calc[adj_reg[m]].extend([k]) for m,n in enumerate(sol_stat) if n]
            for i in adj_reg:
                if (side_inf[k][0] in V[i].tolist()) and (side_inf[k][1] in V[i].tolist()):
                    reg_calc[i].extend([k])  
                # if in_hull_n(V[i],j):
                #     reg_calc[i].extend([k])   
    return reg_calc
def in_hull_n(points, x):
    n_points = len(points)
    n_dim = len(x)
    #x=np.reshape(x,(n_dim,1))
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    x_bounds = (0, 1)
    lp = linprog(c, A_eq=A, b_eq=b,bounds=[x_bounds]*(n_points))
    sol=lp.x
    #print(sol)
    return lp.success

def splitting_cells(V, H, A_dyn, new_point):
    # Get the number of vertices and dimensions
    n_v, n = np.shape(V)

    # Get unique new points and add to the existing vertices
    new_points = np.unique(np.reshape(new_point, (len(new_point), n)), axis=0)
    V = np.append(V, new_points, axis=0)

    # Get the Delaunay triangulation of the vertices
    tri = Delaunay(V).simplices

    # Get the failed region of the vertices
    V_new = failed_region(V[tri])

    # Create new dynamic and horizon arrays based on the number of vertices
    A_dyn_new = [A_dyn.copy()] * np.size(V_new, 0)
    H_new = [H.copy()] * np.size(V_new, 0)

    # Return the new vertices, dynamic array, and horizon array
    return V_new, A_dyn_new, H_new

def in_hull_LPD(points, x):
    n_points = np.size(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    x_bounds = (0, 1)
    lp = linprog(c, A_eq=A, b_eq=b,bounds=[x_bounds]*(n_dim+1),method='interior-point')
    sol=lp.x
    #print(sol)
    return sol
def failed_region(V_new):
# Create an array of booleans to keep track of which points are valid for the ConvexHull
    keep = np.ones(len(V_new), dtype = bool)
# Loop through each point in the input array
    for cntr, i in enumerate(V_new):
        try:
# Try to form a ConvexHull from the current point
            ConvexHull(V_new[cntr])
        except:
# If an error occurs, set the corresponding index in the keep array to False
            keep[cntr] = False
# Filter the input array to only include the points that formed a valid ConvexHull
    V_new = V_new[keep]
# Convert the filtered array to a list
    V_new = list(V_new)
    return V_new  
# def newpoints_id(vertex_new,col,der,iter):
#     dict={1:-0.2,
#         2:-0.2,
#         3:-1,
#         4:-1,
#         5:-0.5,
#         6:-0.5,
#         7:-0.5,
#         8:-0.5,
#         9:-1,
#         10:-1}
#     Threshold=dict.get(iter,-1)
#     cosine=[]
#     index=[]
#     new_list=[]
#     for i in col:
#         cos=np.dot(der[i[0]],der[i[1]])/(norm(der[i[0]])*norm(der[i[1]]))
#         cosine.append(cos)
#         index.append(list(i))      
#     ind=np.argsort(np.array(cosine))
#     if cosine[ind[0]]<-0.5 :
#         for i in ind:
#             if cosine[i]<-0.5:
#                 new_points_test=(0.67)*(np.array(vertex_new[index[i][0]]))+(0.33)*np.array(vertex_new[index[i][1]])
#                 new_list.append(new_points_test)
#                 new_points_test=(0.33)*(np.array(vertex_new[index[i][0]]))+(0.67)*np.array(vertex_new[index[i][1]])
#                 new_list.append(new_points_test)
#     else:
#         new_points_test=(0.5)*(np.array(vertex_new[index[ind[0]][0]]))+(0.5)*np.array(vertex_new[index[ind[0]][1]])
#         new_list.append(new_points_test)
#     return new_list

def newpoints_id(vertex_new,col,der,iter):
    dict={1:0,
        2:0,
        3:0,
        4:0,
        4:0,
        5:0}
    #Threshold=-1
    # Threshold=0
    Threshold=dict.get(iter,-1)
    cosine=[]
    index=[]
    new_list=[]
    length=[]
    side=[]
    for i in col:
        cos=np.dot(der[i[0]],der[i[1]])/(norm(der[i[0]])*norm(der[i[1]]))
        cosine.append(cos)
        index.append(list(i))
        length.append([norm(der[i[0]]),norm(der[i[1]])])      
    ind=np.argsort(np.array(cosine))
    if cosine[ind[0]]<Threshold :
        for i in ind:
            if cosine[i]<Threshold:
                # new_points_test=(0.67)*(np.array(vertex_new[index[i][0]]))+(0.33)*np.array(vertex_new[index[i][1]])
                # new_list.append(new_points_test)
                # new_points_test=(0.33)*(np.array(vertex_new[index[i][0]]))+(0.67)*np.array(vertex_new[index[i][1]])
                # new_list.append(new_points_test)
                coeff1=1/(1+length[i][0]/length[i][1])
                coeff2=1-coeff1
                new_points_test=(coeff1)*(np.array(vertex_new[index[i][0]]))+(coeff2)*np.array(vertex_new[index[i][1]])
                new_list.append(new_points_test)
                side.append([vertex_new[index[i][0]],vertex_new[index[i][1]]])
    else:
        coeff1=1/(1+length[ind[0]][0]/length[ind[0]][1])
        coeff2=1-coeff1
        new_points_test=(coeff1)*(np.array(vertex_new[index[ind[0]][0]]))+(coeff2)*np.array(vertex_new[index[ind[0]][1]])
        new_list.append(new_points_test)
        side.append([vertex_new[index[ind[0]][0]],vertex_new[index[ind[0]][1]]])
    return new_list,side