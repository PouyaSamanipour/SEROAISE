import numpy as np
from scipy.optimize import linprog
import itertools
from scipy.spatial import ConvexHull
#levset_pts=[]
levset_data=dict()
def sol_Process(sol,A_PD,id_var,n,V,n_r):
    nvar=np.size(sol)
    ns=20
    sol_test=np.matmul(A_PD,sol)
    print(sol_test)
    ls=np.linspace(0.8*np.min(sol_test),0.93*np.mean(sol_test),ns)
    ls = [*set(ls)]
    ls=np.sort(ls)
    sol=sol[0:nvar-n_r]
    for i in range(id_var.shape[0]):
        if id_var[i][1]-id_var[i][0]<n+1:
            sol=np.insert(sol,(n+1)*(i+1)-1,0)
# finding max and min of state space
    for k in range(np.size(ls)):
            n_var_cnt=0
            levset_out=[]
            levset_pts=[]
            for i in range(n_r):
                levset_pts=level_set(V[i],sol[n_var_cnt:n_var_cnt+n+1],ls[k],levset_out)
                n_var_cnt=n_var_cnt+n+1
#                print(i)
#                print(len(levset_pts))
            name=str("levelset_")+str(k)
            levset_data[name]=levset_pts
    
    list_points=[]
    for i in range(n_r):
        ns_q=2
        nv=np.size(V[i],0)
        rand_coeff=np.random.rand(nv,ns_q)
        rand_points=rand_coeff/np.sum(rand_coeff,axis=0,keepdims=True)
        rand_points=np.insert(rand_points,ns_q,np.eye(nv),axis=1)
        new_points=np.matmul(np.transpose(V[i]),rand_points)
        #new_points=np.sort(new_points)
        list_points.append(new_points)
    max_val=np.zeros((n_r,n))
    min_val=np.zeros((n_r,n))
    for i in range(n_r):
        max_val[i]=np.max(V[i],0)
        min_val[i]=np.min(V[i],0)
    return min_val,max_val,ls,sol,list_points,levset_data


def check_status(sol,n_r,Threshold):
    nvar=np.size(sol)
    print("max Error is:\n",max(sol[nvar-n_r:nvar]))
    if max(sol[nvar-n_r:nvar])>Threshold:
        status=True
        print("Sum of the errors is:",np.sum(sol[nvar-n_r:nvar]))
    else:
        status=False 
    return status
def level_set(V,sol,ls,levset_out):
    level_set_dum=[]
    ctr=0
    nv,n=np.shape(V)
    sol=np.reshape(sol,(1,n+1))
    n_points = 2
    n_dim = n
    hull=ConvexHull(V)
    sides=hull.simplices
    sol_new=np.delete(sol,n)
    sol_new=np.reshape(sol_new,(1,n_dim))
    for i in range(np.size(sides,0)):
        points=np.array([V[sides[i][0]],V[sides[i][1]]])
        A_eq=np.matmul(sol_new,points.T)
        A = np.r_[A_eq,np.ones((1,n_points))]
        b = np.r_[ls-sol[0][2], np.ones(1)]
        c = np.zeros(n_points)
        #x_bounds = (0, 1)
        #lp = linprog(c, A_eq=A, b_eq=b,bounds=[x_bounds],method='interior-point')
        lp = linprog(c, A_eq=A, b_eq=b)
        if lp.success:
            sol_lv=np.matmul(lp.x,points)
            sol_lv=sol_lv.round(decimals=6)
            level_set_dum.append(sol_lv)
            ctr=ctr+1
    if ctr<2:
            level_set_dum.append(np.zeros(n_dim))
            level_set_dum.append(np.zeros(n_dim))
    elif ctr>2:
        level_set_dum = list(np.unique(level_set_dum,axis=0))
    levset_out.extend(level_set_dum)
    return levset_out

