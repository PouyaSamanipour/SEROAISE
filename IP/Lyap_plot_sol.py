from cProfile import label
from dataclasses import field
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from splitting_cells import in_hull
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.spatial import Delaunay
sample_time=0.01
def plot2d(A_dyn,sol,n_r,H,ls,list_points,levset_data,V,color,legend):
    for j in range(np.size(ls)):
        name="levelset_"+str(j)
        sol_n=np.reshape(sol,(n_r,-1))
        levset_pts=levset_data[name]
        LV=zip(*levset_pts)
        LV_f=tuple(LV)
        for i in range(n_r):
          # hull = ConvexHull(V[i])
          # for simplex in hull.simplices:
          #     plt.plot(V[i][simplex][:,0],V[i][simplex][:,1] ,'g-',linewidth=1.5) 
          plt.xlabel('$X_1$',fontweight='bold', fontsize=20,style='italic')
          plt.ylabel('$X_2$',fontweight='bold', fontsize=20,style='italic')
          plt.grid()
          plt.title("Level sets and flows", fontsize=10,style='italic')
          #plt.show()
    # x_old=np.array([1.43,1.49])
    # x_new=np.zeros(2)
    # for iter in range(1000):
    #   for i in range(n_r):
    #     dum=in_hull(V[i],x_old)
    #     if dum[0]:
    #       dx1=A_dyn[i][0,0]*x_old[0]+A_dyn[i][0,1]*x_old[1]+H[i][0]
    #       dx2=A_dyn[i][1,0]*x_old[0]+A_dyn[i][1,1]*x_old[1]+H[i][1]
    #       x_new[0]=x_old[0]+dx1*sample_time
    #       x_new[1]=x_old[1]+dx2*sample_time
    #       plt.plot([x_old[0],x_new[0]],[x_old[1],x_new[1]],'c',linestyle="--")
    #       x_old=x_new
    #       break
    ns=100
    list_max=[]
    list_min=[]
    for i in V:
        list_max.append(np.max(i,axis=0))
        list_min.append(np.min(i,axis=0))
    max=np.max(list_max,axis=0)
    min=np.min(list_min,axis=0)
    x1 = np.linspace(min[0],max[0],ns)
    x2 = np.linspace(min[1],max[1],ns)
    # Creating 2-D grid of features
    [X, Y] = np.meshgrid(x1, x2)  
    u,v,Z=model_PWA(A_dyn,H,sol_n,X,Y,V,n_r)
    plt.contour(X, Y, Z,ls,colors=color)
    n = 10
    color_array = np.sqrt(((v-n)/2)**2 + ((u-n)/2)**2)
    widths = 0.3
    # qv1=plt.streamplot(X,Y,u,v,color=('blue'), linewidth=1,
    #                     density=1, arrowstyle='-|>', arrowsize=1.5)
    plt.legend([legend],loc='upper right')      
    red_line = mlines.Line2D([], [], color='red', marker="_",label=legend,
                          markersize=15)                     
    arrow = plt.scatter( 0,0, c='blue',marker=r'$\longrightarrow$',s=40, label='Flows' )
    plt.legend(handles=[red_line],loc='upper right')
    plt.plot(0,0,color='red', marker="o",)
    # plt.show()

def model_PWA(A_dyn,H,sol_n,X,Y,V,n_r):
  hull=[Delaunay(i) for i in V]
  u=np.zeros((np.size(X,0),np.size(X,0)))
  v=np.zeros((np.size(X,0),np.size(X,0)))
  Z=np.zeros((np.size(X,0),np.size(X,0)))
  for i in range(np.size(X,0)):
    for j in range(np.size(X,1)):
        X_new=np.array([X[i,j],Y[i,j]])
        sol_stat=[i.find_simplex(X_new)>=0 for i in hull]
        try:
          k=sol_stat.index(True)
          Z[i,j]=np.matmul(sol_n[k,0:2],X_new)+sol_n[k,2]
          u[i,j]=A_dyn[k][0,0]*X[i,j]+A_dyn[k][0,1]*Y[i,j]+H[k][0]
          v[i,j]=A_dyn[k][1,0]*X[i,j]+A_dyn[k][1,1]*Y[i,j]+H[k][1]
        except:
          Z[i,j]=100000
          u[i,j]=0
          v[i,j]=0
        # for k in range(n_r):
        #   dum=in_hull(V[k],X_new)
        #   if dum[0]:
        #     u[i,j]=A_dyn[k][0,0]*X[i,j]+A_dyn[k][0,1]*Y[i,j]+H[k][0]
        #     v[i,j]=A_dyn[k][1,0]*X[i,j]+A_dyn[k][1,1]*Y[i,j]+H[k][1]
        #     Z[i,j]=np.matmul(sol_n[k,0:2],X_new)+sol_n[k,2]
        #     break
  return u,v,Z
