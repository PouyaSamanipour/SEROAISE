import gurobipy as gb
from gurobipy import GRB
import numpy as np
import torch
def preprocessing_Lyap_test(enumerate_poly,D,W,c,hyperplanes,b,eps1,eps2,h):
    model=torch.jit.load(h)
    model.eval()
    W_h=model.out.weight
    n_h,n=np.shape(hyperplanes)
    n_r=len(enumerate_poly)
    n_var=n_h+1+n_r
    m=gb.Model("linear") 
    x={}
    for i in range(n_var):
        if i <n_var-n_r:
            x[i] = m.addVar(lb=-float('inf'),name=f"x[{i}]")
        else:
            x[i] = m.addVar(lb=1e-12,name=f"Slack[{i}]")
    V=[]
    index_list=[]
    [(index_list.extend([i]*len(j)),V.extend(j.tolist())) for i,j in enumerate(enumerate_poly)]
    #[V.extend(i) for i in enumerate_poly]
    var_w=[x[i] for i in range(n_h)]
    var_w=np.reshape(var_w,(-1,n_h))
    buffer=[]
    ReLU_val=np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0)
    # dot_x=np.dot(W,np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0))+np.reshape(c,(len(c),1))
    for j,i in enumerate(V):
        if i not in buffer:
            if (i!=[0]*n):
                # eq=np.dot(var_w,ReLU_val[:,j:j+1])+x[n_h]
                eq=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+x[n_h]
                m.addConstr(eq[0][0]>=eps1)
                # m.addConstr(eq[0][0]<=4.08)
            else:
                eq=np.dot(var_w,ReLU_val[:,j:j+1])+x[n_h]
                eq=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+x[n_h]

                m.addConstr(eq[0][0]<=1e-6)
                m.addConstr(eq[0][0]>=-1e-6)
        if (i!=[0]*n):
            dot_x_test=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
            # eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x[:,j:j+1]))-x[n_h+1+index_list[j]]
            eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x_test))-x[n_h+1+index_list[j]]
            m.addConstr(eq[0][0]<=-eps2)
        buffer.append(i)
        val_test=model(torch.FloatTensor(np.array(i)).cuda())
        if val_test<=1e-8:
            val_new=-W_h@torch.FloatTensor(np.dot(np.dot(np.diag(D[index_list[j],0:146]),hyperplanes[0:146,:]),dot_x_test)).cuda()
            if val_new>-eps2:
                print('error')
        
    param=[x[i] for i in range (n_h+1,n_var)]
    #m.addConstrs(x[i]>=1e-12 for i in range (0,n_list[0]-2*n_r))
    m.setObjective(gb.quicksum(param), GRB.MINIMIZE)
    m.setParam('BarHomogeneous', 1)
    m.optimize()
    sol = m.getAttr('X')
    # m.write('model_o.rlp')
    # for constr in m.getConstrs():
    #     print(constr)
    #print(sol)
    return sol,n_h,n_r,n,m.objVal