import sys
import numba as nb
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import plot_res_BF

# def Refinement(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol,W,c,eps,D,boundary_regions):
#     n_r=len(enumerate_poly)
#     n_h,n=np.shape(all_hyperplanes)
#     W_v=sol[0:n_h]
#     W_v=np.reshape(W_v,(1,n_h))
#     c_v=sol[n_h]
#     hyperplanes=[]
#     bias=[]
#     all_hyperplanes_new=np.copy(all_hyperplanes)
#     all_bias_new=np.copy(all_bias)
#     for i in range(len(boundary_regions)):
#         if slack_var[i]>=eps:
#             reg=boundary_regions[i]
#             x_dot=np.dot(W,np.maximum(np.dot(all_hyperplanes,np.array(enumerate_poly[reg]).T)+all_bias,0))+c
#             A_dyn=W@np.diag(D[reg])@all_hyperplanes
#             B_dyn=W@np.diag(D[reg])@all_bias+c
#             dh=W_v@np.diag(D[reg])@all_hyperplanes@x_dot
#             h=W_v@np.diag(D[reg])@(all_hyperplanes@np.array(enumerate_poly[reg]).T+all_bias)+c_v
#             if np.max(h)*np.min(h)<0:
#                 h_new1=W_v@np.diag(D[reg])@all_hyperplanes
#                 b_new1=W_v@np.diag(D[reg])@all_bias+c_v
#                 if np.all(h_new1!=0):
#                     h_new=h_new1/np.linalg.norm(h_new1)
#                     b_new=b_new1/np.linalg.norm(h_new1)
#                     hyperplanes.extend(h_new)
#                     all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
#                     bias.extend(b_new[0])
#                     all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
#                     # plot_res.plot_polytope([enumerate_poly[reg]],"blue")
#                     # plot_res.plot_hype(h_new[0],b_new[0],3.14)
#                     # print("Refinement Done")

#             elif np.max(dh)*np.min(dh)<0:
#                 h_new1=W_v@np.diag(D[reg])@all_hyperplanes@A_dyn
#                 b_new1=W_v@np.diag(D[reg])@all_hyperplanes@B_dyn+c_v
#                 if np.all(h_new1!=0):
#                     h_new=h_new1/np.linalg.norm(h_new1)
#                     b_new=b_new1/np.linalg.norm(h_new1)

#                     hyperplanes.extend(h_new)
#                     all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
#                     bias.extend(b_new[0])
#                     all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
#                     # plot_res.plot_polytope([enumerate_poly[reg]],"blue")
#                     # plot_res.plot_hype(h_new[0],b_new[0],3.14)
#                     # print("Refinement Done")

#             else:
#                 h_new1=W_v@np.diag(D[reg])@all_hyperplanes@A_dyn
#                 b_new1=W_v@np.diag(D[reg])@all_hyperplanes@B_dyn+c_v-np.mean(dh)
#                 if np.all(h_new1!=0):
#                     h_new=h_new1/np.linalg.norm(h_new1)
#                     b_new=b_new1/np.linalg.norm(h_new1)
#                     hyperplanes.extend(h_new)
#                     all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
#                     bias.extend(b_new[0])
#                     all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
#                     # plot_res.plot_polytope([enumerate_poly[reg]],"blue")
#                     # plot_res.plot_hype(h_new[0],b_new[0],3.14)
#                     # print("Refinement Done")

#     W_append=np.zeros((n,2*len(hyperplanes)))
#     #W=np.append(W,W_append,axis=1)
#     W=np.hstack((W, W_append))
#     print("end of Refinement")
#     return all_hyperplanes_new,all_bias_new,hyperplanes,bias,enumerate_poly,W
def Refinement(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol,W,c,eps,D,boundary_regions,zero_reg,H,iter):
    n_r=len(enumerate_poly)
    n_h,n=np.shape(all_hyperplanes)
    W_v=sol[0:n_h]
    W_v=np.reshape(W_v,(1,n_h))
    c_v=sol[n_h]
    hyperplanes=[]
    bias=[]
    all_hyperplanes_new=np.copy(all_hyperplanes)
    all_bias_new=np.copy(all_bias)
    regs=[]
    # for i in range(len(reg_out)):
    #     if slack_var[i]>=eps:
    #         reg=boundary_regions[i]
    #         x_dot=np.dot(W,np.maximum(np.dot(all_hyperplanes,np.array(enumerate_poly[reg]).T)+all_bias,0))+c
    #         x_dot_m=np.sum(x_dot,axis=1,keepdims=True)/len(x_dot)
    #         mid_point=np.mean(enumerate_poly[reg],axis=0,keepdims=True)
    #         h_new= x_dot_m.T 
    #         b_new=-h_new@mid_point.T 
    #         regs.append(reg)    
    #         # A_dyn=W@np.diag(D[reg])@all_hyperplanes
    #         # B_dyn=W@np.diag(D[reg])@all_bias+c
    #         # dh=W_v@np.diag(D[reg])@all_hyperplanes@x_dot
    #         # h=W_v@np.diag(D[reg])@(all_hyperplanes@np.array(enumerate_poly[reg]).T+all_bias)+c_v
    #         # h_new1=W_v@np.diag(D[reg])@all_hyperplanes
    #         # b_new1=W_v@np.diag(D[reg])@all_bias+c_v-slack_var[i]
    #         # if np.all(h_new1!=0):
    #         #     h_new=h_new1/np.linalg.norm(h_new1)
    #         #     b_new=b_new1/np.linalg.norm(h_new1)
    #         # else:
    #         #     h_new=np.copy(h_new1)
    #         #     b_new=np.copy(b_new1)


            
            
            
    #         # h_new=W_v@np.diag(D[reg])@all_hyperplanes
    #         # b_new=W_v@np.diag(D[reg])@all_bias+c_v-5*slack_var[i]
    #         hyperplanes.extend(h_new)
    #         all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
    #         bias.extend(b_new[0])
    #         all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
    #         plot_polytope([enumerate_poly[reg]],"blue")
    #         plot_hype(h_new[0],b_new[0],3.14)
            #         # print("Refinement Done")
    for i in range(len(boundary_regions)):
        if boundary_regions[i] in zero_reg and iter==0:
            indx=zero_reg.index(boundary_regions[i])
            h_new= np.array([H[indx][:-1]])
            b_new= np.array([[H[indx][-1]]])
            hyperplanes.extend(h_new)
            all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
            bias.extend(b_new[0])
            all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
            # plot_res_BF.plot_polytope([enumerate_poly[boundary_regions[i]]],"blue")
            # plot_res_BF.plot_hype(h_new[0],b_new[0],3.14)
        elif slack_var[i]>=eps:
            reg=boundary_regions[i]
            x_dot=np.dot(W,np.maximum(np.dot(all_hyperplanes,np.array(enumerate_poly[reg]).T)+all_bias,0))+c
            x_dot_m=np.sum(x_dot,axis=1,keepdims=True)/len(x_dot)
            mid_point=np.mean(enumerate_poly[reg],axis=0,keepdims=True)
            h_new= x_dot_m.T 
            b_new=-h_new@mid_point.T 
            regs.append(reg)    
            hyperplanes.extend(h_new)
            all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
            bias.extend(b_new[0])
            all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
            # plot_polytope([enumerate_poly[reg]],"blue")
            # plot_hype(h_new[0],b_new[0],3.14)

            
            # if np.max(h)*np.min(h)<0:
            #     h_new1=W_v@np.diag(D[reg])@all_hyperplanes
            #     b_new1=W_v@np.diag(D[reg])@all_bias+c_v
            #     if np.all(h_new1!=0):
            #         h_new=h_new1/np.linalg.norm(h_new1)
            #         b_new=b_new1/np.linalg.norm(h_new1)
            #         hyperplanes.extend(h_new)
            #         all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
            #         bias.extend(b_new[0])
            #         all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
            #         plot_polytope([enumerate_poly[reg]],"blue")
            #         plot_hype(h_new[0],b_new[0],3.14)
            #         # print("Refinement Done")

            # elif np.max(dh)*np.min(dh)<0:
            #     h_new1=W_v@np.diag(D[reg])@all_hyperplanes@A_dyn
            #     b_new1=W_v@np.diag(D[reg])@all_hyperplanes@B_dyn+c_v
            #     if np.all(h_new1!=0):
            #         h_new=h_new1/np.linalg.norm(h_new1)
            #         b_new=b_new1/np.linalg.norm(h_new1)

            #         hyperplanes.extend(h_new)
            #         all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
            #         bias.extend(b_new[0])
            #         all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
            #         plot_polytope([enumerate_poly[reg]],"blue")
            #         plot_hype(h_new[0],b_new[0],3.14)
            #         # print("Refinement Done")

            # else:
            #     h_new1=W_v@np.diag(D[reg])@all_hyperplanes@A_dyn
            #     b_new1=W_v@np.diag(D[reg])@all_hyperplanes@B_dyn+c_v-np.mean(dh)
            #     if np.all(h_new1!=0):
            #         h_new=h_new1/np.linalg.norm(h_new1)
            #         b_new=b_new1/np.linalg.norm(h_new1)
            #         hyperplanes.extend(h_new)
            #         all_hyperplanes_new=np.vstack((all_hyperplanes_new,h_new,-h_new))
            #         bias.extend(b_new[0])
            #         all_bias_new=np.vstack((all_bias_new,b_new,-b_new))
            #         plot_polytope([enumerate_poly[reg]],"blue")
            #         plot_hype(h_new[0],b_new[0],3.14)

    W_append=np.zeros((n,2*len(hyperplanes)))
    #W=np.append(W,W_append,axis=1)
    W=np.hstack((W, W_append))
    print("end of Refinement")
    return all_hyperplanes_new,all_bias_new,hyperplanes,bias,enumerate_poly,W


def find_cosine(dx):
    cosine_lst=[]
    norm = np.linalg.norm(dx.T, ord=2, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    dx_norm = dx.T/norm
    ans=dx_norm@dx_norm.T
    upper_tri=np.triu(ans, k=0)
    value_upper_tri=(upper_tri[(np.abs(upper_tri)>0)*(np.abs(upper_tri)<1)])
    indices=np.stack(np.where((np.abs(upper_tri)>0)*(np.abs(upper_tri)<1)))
    final=np.vstack((value_upper_tri,indices)).T


    return final
