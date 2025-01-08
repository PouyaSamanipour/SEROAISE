import gurobipy as gb
from gurobipy import GRB
import numpy as np
import numba
from numba import njit
def preprocessing_BF(enumerate_poly,D,W,c,hyperplanes,b,eps1,eps2,TH,alpha):
    reg_out=[]
    n_h,n=np.shape(hyperplanes)
    n_r=len(enumerate_poly)
    boundary_regions=finding_boundary_Regions(enumerate_poly,TH,hyperplanes,b,c,W)
    
    n_b=len(boundary_regions) # number of boundary regions
    n_var=n_h+2+n_b
    m=gb.Model("linear") 
    x={}
    for i in range(n_var):
        if i <n_var-(n_b+1):
            x[i] = m.addVar(lb=-float('inf'),name=f"x[{i}]")
        else:
            x[i] = m.addVar(lb=1e-12,name=f"Slack[{i}]")
    # V=[]
    # index_list=[]
    # [(index_list.extend([i]*len(j)),V.extend(j.tolist())) for i,j in enumerate(enumerate_poly)]
    index_list,V=create_index_and_vertices_lists(enumerate_poly)
    #[V.extend(i) for i in enumerate_poly]
    var_w=[x[i] for i in range(n_h)]
    var_w=np.reshape(var_w,(-1,n_h))
    var_c=x[n_h]
    buffer=[]
    # ReLU_val=np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0)
    # dot_x=np.dot(W,np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0))+np.reshape(c,(len(c),1))
    for j,i in enumerate(V):
        h=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+var_c
        if i.tolist() not in buffer:
            buffer.append(i.tolist())
            if (np.max(np.abs(i))<TH-1e-6):
                eq=h+x[n_var-1]
            # if (i!=[0]*n):
                # eq=np.dot(var_w,ReLU_val[:,j:j+1])+x[n_h]
                # h=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+var_c
                m.addConstr(eq[0][0]>=eps1,name=f"PI")
                # m.addConstr(eq[0][0]<=10,name=f"PI")

            elif (np.max(np.abs(i))>=TH-1e-6):
                xdot=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
                id=np.where(boundary_regions==index_list[j])[0][0]
                eq=h-x[n_h+1+id]
                m.addConstr(eq[0][0]<=-eps1,name=f"NB")
                m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
                # if i@xdot>1e-8:
                #     id=np.where(boundary_regions==index_list[j])[0][0]
                #     eq=h-x[n_h+1+id]
                #     m.addConstr(eq[0][0]<=-eps1,name=f"NB")
                #     # m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
                # else:
                #     eq=h+x[n_var-1]
                #     m.addConstr(eq[0][0]>=eps1,name=f"PI")

                
            # else:
            #     eq=np.dot(var_w,ReLU_val[:,j:j+1])+x[n_h]
            #     eq=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+x[n_h]

            #     m.addConstr(eq[0][0]<=1e-6)
            #     m.addConstr(eq[0][0]>=-1e-6)
        # else:
        #     print("repeated")
        # if (i!=[0]*n):
        if i.tolist()==[0]*n:
            dot_x_test=np.array([0]*n)
        else:
            dot_x_test=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
            # eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x[:,j:j+1]))-x[n_h+1+index_list[j]]
        eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x_test))+alpha*(h)
        m.addConstr(eq[0][0]>=eps2,name=f"PD")

    param=[x[i] for i in range (n_h+1,n_var)]
    #m.addConstrs(x[i]>=1e-12 for i in range (0,n_list[0]-2*n_r))
    m.setObjective(gb.quicksum(param), GRB.MINIMIZE)
    m.setParam('BarHomogeneous', 1)
    m.optimize()
    sol = m.getAttr('X')
    # m.write('model_o.rlp')
    W_v=np.array([sol[i] for i in range(n_h)])
    c_v=sol[n_h]
    # for constr in m.getConstrs():
    #     print(constr)
    #print(sol)
    return sol,n_h,n_r,n,m.objVal,boundary_regions,reg_out




@njit
def finding_boundary_Regions(enumerate_poly, TH,hyperplanes,b,c,W):
    regions = []  # Use a list to collect regions
    for k, i in enumerate(enumerate_poly):
        for j in i:
            if np.max(np.abs(j)) >= TH - 1e-6:
                # xdot=np.dot(W,np.maximum(np.dot(hyperplanes,j)+b.T,0).T)+np.reshape(c,(len(c),1))
                # if j@xdot>1e-8:
                    regions.append(k)  # Append k to the list
    return np.unique(np.array(regions))  # Convert to array and return unique values


@njit
def create_index_and_vertices_lists(enumerate_poly):
    n=len(enumerate_poly[0][0])  # Get the number of elements in the first list of enumerate_poly
    V = np.zeros((0,n), dtype=np.float64)  # Pre-allocate an empty array for V
    index_list = []  # Pre-allocate an empty array for index_list
    
    for i, j in enumerate(enumerate_poly):
        # Extend index_list with the current index 'i' repeated 'len(j)' times
        index_list.extend([i]*len(j)) # Create an array of repeated indices
        # Extend V with elements from j
        V = np.vstack((V, j))  # Concatenate the values

    return index_list, V