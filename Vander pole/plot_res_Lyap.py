import numba as nb
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import time
import matplotlib.lines as mlines
import os
from numba import prange
import torch
def plot_hyperplanes_and_vertices(hyperplanes, vertices):
    # Check if the data is in 2D or 3D
    dim = hyperplanes[0][0:-1].size

    if dim == 2:
        # 2D Plot
        for E in hyperplanes:
            x = np.linspace(-10, 10, 5)
            if E[1]!=0:
                y = (-E[2] - E[0] * x) / E[1]
            else:
                x=np.array([-E[2]/E[0]]*5)
                y=np.array([0]*5)                
            plt.plot(x, y, label=f'{E[0]}x + {E[1]}y >= {E[2]}')
        
        for vertex in vertices:
            plt.scatter(*vertex, color='red')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        #plt.legend()
        plt.xlim(-0.9,0.9)
        plt.ylim(-0.9,0.9)
        plt.show()
    elif dim == 3:
        # 3D Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for E in hyperplanes:
            x, y = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
            z = (E[3] - E[0] * x - E[1] * y) / E[2]
            ax.plot_surface(x, y, z, alpha=0.5)
        
        vertices = np.array(vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', marker='o', s=100, label='Vertices')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

def plotting_results_lyap(TH,all_hyperplanes,all_bias,c_v,W_v,W,c,enumerate_poly,name):
    x1=np.linspace(-TH,TH,200)
    x2=np.linspace(-TH,TH,200)
    X1,X2=np.meshgrid(x1,x2)
    data=tuple(zip(np.ravel(X1),np.ravel(X2)))
    LV=np.zeros(np.size(X1,0)*np.size(X1,0))
    dX1=np.zeros(np.size(X1,0)*np.size(X1,0))
    dX2=np.zeros(np.size(X1,0)*np.size(X1,0))
    LV=np.dot(W_v,np.maximum(np.dot(all_hyperplanes,np.array(data).T)+all_bias,0))+c_v
    dX1,dX2=np.dot(W,np.maximum(np.dot(all_hyperplanes,np.array(data).T)+all_bias,0))+np.reshape(c,(2,1))
    Z=LV.reshape(len(X1),len(X2))
    dX1=dX1.reshape(len(X1),len(X2))
    dX2=dX2.reshape(len(X1),len(X2))
    plt.xlabel('$X_1$',fontweight='bold', fontsize=20,style='italic')
    plt.ylabel('$X_2$',fontweight='bold', fontsize=20,style='italic')
    levels=np.linspace(np.min(LV),0.8*np.max(LV),20)
    plt.contour(X1, X2, Z,[21],colors='red')
    plt.streamplot(X1,X2,dX1,dX2,color=('blue'), linewidth=1,
                        density=1, arrowstyle='-|>', arrowsize=1.5)
    # plt.quiver(X1,X2,dX1,dX2,color=('blue'))
    #Plotting Regions
    # for k in range(len(enumerate_poly)):
    #     hull_n = ConvexHull(enumerate_poly[k])
    #     for simplex in hull_n.simplices:
    #             plt.plot(np.array(enumerate_poly[k])[simplex,0],np.array(enumerate_poly[k])[simplex,1] ,'g-',linewidth=1.5) 
    plt.xlabel('$X_1$',fontweight='bold', fontsize=20,style='italic')
    plt.ylabel('$X_2$',fontweight='bold', fontsize=20,style='italic')
    plt.legend(['Level sets','Flows'],loc='upper right')
    red_line = mlines.Line2D([], [], color='red', marker="_",label="Level sets",
                          markersize=15) 
    arrow = plt.scatter( 0,0, c='blue',marker=r'$\longrightarrow$',s=40, label='Flows' )
    plt.legend(handles=[red_line,arrow],loc='upper right')
    plt.title(name)
    cntr=0
    name_new=name+"_"+str(cntr)+".png"
    while os.path.exists(os.getcwd()+"/Figures/"+name_new):
      cntr=cntr+1
      name_new=name+"_"+str(cntr)+".png"
    plt.savefig(os.getcwd()+'/Figures/'+name_new)
    plt.figure("3D")
    ax = plt.axes(projection='3d')
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('Lyapunov')
    # plt.show()



def plot_polytope(enumerate_poly, name):
    for i in enumerate_poly:
        hull = ConvexHull(i)
        for simplex in hull.simplices:
            plt.plot(i[simplex, 0], i[simplex, 1], 'g--')
        # plt.plot(i[hull.vertices,0], i[hull.vertices,1], 'r--', lw=2)
        # plt.plot(i[hull.vertices[0],0], i[hull.vertices[0],1], 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title(name)
    # plt.show()


def plot_polytope_2D(NN_model,TH):
    import torch
    # device = torch.device('cuda' if torch.cuda.is_available())
    model = torch.jit.load(NN_model)
    # Load the .pth file

# Print available keys
    # model.load_state_dict(torch.load(NN_model))
    model=model.to('cpu')
    model.eval()
    x1=np.linspace(-TH[0],TH[0],700)
    x2=np.linspace(-TH[1],TH[1],700)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    with torch.no_grad():  # No need to track gradients for evaluation
        output = model(torch.FloatTensor(data))
    plt.streamplot(X1,X2,np.array(output[:,0]).reshape(len(X1),len(X2)),np.array(output[:,1]).reshape(len(X1),len(X2)),color=('blue'),linewidth=1)



def plot_invariant_set_multiple(NN_model,zeros,TH,alph,color):
    # import torch
    model = torch.jit.load(NN_model)

    model=model.to('cpu')
    model.eval()
    x1=np.linspace(-TH[0],TH[0],100)
    x2=np.linspace(-TH[1],TH[1],100)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    with torch.no_grad():  # No need to track gradients for evaluation
        if len(zeros)==0:
            output = model(torch.FloatTensor(data))
        else:
            val=model(torch.FloatTensor(zeros))
            output = model(torch.FloatTensor(data))-torch.max(val)
    Z=np.asarray(output)
    Z=np.reshape(Z,(len(x1),len(x2)))
    # contour1=plt.contour(X1, X2,Z,levels=[0],colors=color,label=f'a = {alph}')
    # # contour1.collections[0].set_label('$\alpha$='+str(alph))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    return X1,X2,Z

def plot_invariant_set_weights(W,c,h,b,TH):
    # import torch
    # model = torch.jit.load(NN_model)

    # model=model.to('cpu')
    # model.eval()
    x1=np.linspace(-TH,TH,200)
    x2=np.linspace(-TH,TH,200)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    # No need to track gradients for evaluation
    output =W@np.maximum(np.dot(h[0],np.array(data).T)+b[0].reshape((len(b[0]),1)),0)+c 
    Z=np.asarray(output)
    Z=np.reshape(Z,(len(x1),len(x2)))
    plt.contour(X1, X2,Z,levels=[0],colors='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def plot_hype(A,b,TH):
    x1=np.linspace(-TH,TH,5)
    if A[1]!=0:
        # x2=np.array([b/A[0]]*5)
        x2=(-A[0]*x1-b)/A[1]
    else:
        x2=np.linspace(-TH,TH,5)
        x1=-b/A[0]
    plt.xlim(-TH,TH)
    plt.ylim(-TH,TH)
    plt.plot(x1,x2,'c')
    # plt.show()

def plot_level_set(model,TH,color,level):
    # import torch
    # model = torch.jit.load(NN_model)
    # # plt.figure("Comparison")
    # model=model.to('cpu')
    # model.eval()
    x1=np.linspace(-TH[0],TH[0],100)
    x2=np.linspace(-TH[1],TH[1],100)
    # x1=np.linspace(-TH,TH,500)
    # x2=np.linspace(-TH,TH,500)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    with torch.no_grad():  # No need to track gradients for evaluation
        output = model(torch.FloatTensor(data))+torch.unsqueeze(torch.norm(torch.FloatTensor(data)@(model.R.T@model.R+0.01*torch.eye(2)),p=1,dim=1),dim=1)
    Z=np.asarray(output)
    Z=np.reshape(Z,(len(x1),len(x2)))
    plt.contour(X1, X2,Z,[5],colors=color)
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.show()

def plot_invariant_set_single(NN_model,TH,zeros,color):
    # import torch
    model = torch.jit.load(NN_model)

    model=model.to('cpu')
    model.eval()
    x1=np.linspace(-TH[0],TH[0],200)
    x2=np.linspace(-TH[0],TH[0],200)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    with torch.no_grad():  # No need to track gradients for evaluation
        # output = model(torch.FloatTensor(data))
        if len(zeros)==0:
            output = model(torch.FloatTensor(data))
        else:
            val=model(torch.FloatTensor(zeros))
            output = model(torch.FloatTensor(data))-torch.max(val)
    Z=np.asarray(output)
    Z=np.reshape(Z,(len(x1),len(x2)))
    # img=plt.pcolormesh(X1, X2, Z, cmap='coolwarm', shading='auto')  # 'RdBu' colormap for red-blue
    # cbar = plt.colorbar(img)
    plt.contour(X1, X2,Z,levels=[0],colors=color)
    plt.xlim(-TH[0],TH[0])
    plt.ylim(-TH[1],TH[0])
    # # contour1.collections[0].set_label('$\alpha$='+str(alph))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # return X1,X2,Z    

def plot_heatmap(NN_model,TH,zeros):
    # import torch
    model = torch.jit.load(NN_model)

    model=model.to('cpu')
    model.eval()
    x1=np.linspace(-TH[0],TH[0],200)
    x2=np.linspace(-TH[0],TH[0],200)
    X1,X2=np.meshgrid(x1,x2)
    data=np.array(tuple(zip(np.ravel(X1),np.ravel(X2))))
    with torch.no_grad():  # No need to track gradients for evaluation
        # output = model(torch.FloatTensor(data))
        if len(zeros)==0:
            output = model(torch.FloatTensor(data))
        else:
            val=model(torch.FloatTensor(zeros))
            output = model(torch.FloatTensor(data))-torch.max(val)
    Z=np.asarray(output)
    Z=np.reshape(Z,(len(x1),len(x2)))
    img=plt.pcolormesh(X1, X2, Z, cmap='coolwarm', shading='auto')  # 'RdBu' colormap for red-blue
    cbar = plt.colorbar(img)
    plt.xlim(-TH[0],TH[0])
    plt.ylim(-TH[0],TH[0])

