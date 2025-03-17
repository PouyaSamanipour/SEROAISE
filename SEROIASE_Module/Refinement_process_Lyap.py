import sys
import numba as nb
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def Refinement_Lyap(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol,W,c,eps,D):
    n_r=len(enumerate_poly)
    n_h,n=np.shape(all_hyperplanes)
    W_v=sol[0:n_h]
    W_v=np.reshape(W_v,(1,n_h))
    c_v=sol[n_h]
    hyperplanes=[]
    bias=[]
    all_hyperplanes_new=np.copy(all_hyperplanes)
    all_bias_new=np.copy(all_bias)
    for i in range(n_r):
        if slack_var[i]>=eps:
            x_dot=np.dot(W,np.maximum(np.dot(all_hyperplanes,np.array(enumerate_poly[i]).T)+all_bias,0))+c
            val=np.dot(np.dot(W_v,np.dot(np.diag(D[i]),all_hyperplanes)),x_dot)
            WvDH=np.dot(W_v,np.dot(np.diag(D[i]),all_hyperplanes))
            WDH=np.dot(W,np.dot(np.diag(D[i]),all_hyperplanes))
            WDb=np.dot(W,np.dot(np.diag(D[i]),all_bias))
            if np.min(val)>0:
                #val=np.dot(np.dot(W_v,np.dot(np.diag(D[i]),all_hyperplanes)),np.array(enumerate_poly[i]).T)
                new_h=np.dot(WvDH,WDH)
                new_b=np.dot(WvDH,WDb)-np.mean(val)
                hyperplanes.extend(new_h)
                all_hyperplanes_new=np.vstack((all_hyperplanes_new,new_h,-new_h))
                # all_hyperplanes_new=np.append(all_hyperplanes_new,new_h,axis=0)
                # all_hyperplanes_new=np.append(all_hyperplanes_new,-new_h,axis=0)
                bias.extend(new_b[0])
                all_bias_new=np.vstack((all_bias_new,new_b,-new_b))
                # all_bias_new=np.append(all_bias_new,-new_b)
                # all_bias_new=np.append(all_bias_new,new_b)
                # all_bias_new=np.append(all_bias_new,new_b)
            else:
                if [0]*n not in enumerate_poly[i]:
                    new_h=np.dot(WvDH,WDH)
                    new_b=np.dot(WvDH,WDb)
                    hyperplanes.extend(new_h)
                    all_hyperplanes_new=np.vstack((all_hyperplanes_new,new_h,-new_h))
                    # all_hyperplanes_new=np.append(all_hyperplanes_new,new_h,axis=0)
                    # all_hyperplanes_new=np.append(all_hyperplanes_new,-new_h,axis=0)
                    bias.extend(new_b[0])
                    all_bias_new=np.vstack((all_bias_new,new_b,-new_b))
                    # all_bias_new=np.append(all_bias_new,-new_b)
                    # all_bias_new=np.append(all_bias_new,new_b)
                else:
                    new_h=np.dot(WvDH,WDH)
                    new_b=np.array([0])
                    hyperplanes.extend(new_h)
                    all_hyperplanes_new=np.vstack((all_hyperplanes_new,new_h,-new_h))
                    # all_hyperplanes_new=np.append(all_hyperplanes_new,new_h,axis=0)
                    # all_hyperplanes_new=np.append(all_hyperplanes_new,-new_h,axis=0)
                    bias.extend(new_b)
                    all_bias_new=np.vstack((all_bias_new,new_b,-new_b))
                    # all_bias_new=np.append(all_bias_new,-new_b)
                    # all_bias_new=np.append(all_bias_new,new_b)
                
                
                #Plotting Hyperplanes
                # hull_n = ConvexHull(enumerate_poly[i])
                # for simplex in hull_n.simplices:
                #     plt.plot(np.array(enumerate_poly[i])[simplex,0],np.array(enumerate_poly[i])[simplex,1] ,'g-',linewidth=1.5) 
                # x11=np.array([-3.14])
                # x21=(-new_b-new_h[0,0]*x11)/new_h[0,1]
                # x12=np.array([3.14])
                # x22=(-new_b-new_h[0,0]*x12)/new_h[0,1]
                # plt.plot([x11,x12],[x21[0],x22[0]],'r')

    W_append=np.zeros((n,2*len(hyperplanes)))
    #W=np.append(W,W_append,axis=1)
    W=np.hstack((W, W_append))
    print("end of Refinement")
    return all_hyperplanes_new,all_bias_new,hyperplanes,bias,enumerate_poly,W
