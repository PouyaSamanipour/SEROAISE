import gurobipy as gb
from gurobipy import GRB
import numpy as np
import numba
from numba import njit
def preprocessing_BF(enumerate_poly,D,W,c,hyperplanes,b,eps1,eps2,TH,alpha):
    n_h,n=np.shape(hyperplanes)
    n_r=len(enumerate_poly)
    boundary_regions,index_list,V,zero_reg,H,zero_point=finding_boundary_Regions(enumerate_poly,TH,hyperplanes,b,c,W,D)
    n_b=len(boundary_regions) # number of boundary regions
    n_var=n_h+2+n_b
    m=gb.Model("linear") 
    x={}
    for i in range(n_var):
        if i <n_var-(n_b+1):
            # x[i] = m.addVar(lb=-float('inf'),name=f"x[{i}]")
            x[i] = m.addVar(lb=-100.0,ub=100.0,name=f"Weight[{i}]")    
        else:
            x[i] = m.addVar(lb=1e-12,name=f"Slack[{i}]")
    # V=[]
    # index_list=[]
    # [(index_list.extend([i]*len(j)),V.extend(j.tolist())) for i,j in enumerate(enumerate_poly)]
    # index_list,V=create_index_and_vertices_lists(enumerate_poly)
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
            if np.all(TH-i>1e-6) and np.all(i+TH>1e-6):
            # if (np.max(np.abs(i))<TH-1e-6):
                eq=h+x[n_var-1]
            # if (i!=[0]*n):
                # eq=np.dot(var_w,ReLU_val[:,j:j+1])+x[n_h]
                # h=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+var_c
                m.addConstr(eq[0][0]>=eps1,name=f"PI")
                # m.addConstr(eq[0][0]<=10,name=f"PI")

            elif np.any(TH-i<=1e-6) or np.any(i+TH<=1e-6):
                # xdot=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
                id=np.where(boundary_regions==index_list[j])[0][0]
                eq=h-x[n_h+1+id]
                m.addConstr(eq[0][0]<=-eps1,name=f"NB")
                # m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
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
        # if np.all(TH-i>1e-6) and np.all(i+TH>1e-6):
        #     eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x_test))+alpha*(h)
        # else:
        #     eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x_test))+alpha*(h)
        eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x_test))+alpha*(h)
        m.addConstr(eq[0][0]>=eps2,name=f"PD")

    param=[x[i] for i in range (n_h+1,n_var)]
    quadterm=gb.QuadExpr()
    quadterm=gb.quicksum([x[i]*x[i] for i in range (0,n_h+1)])

    #m.addConstrs(x[i]>=1e-12 for i in range (0,n_list[0]-2*n_r))
    # m.setObjective(gb.quicksum(param)+0.1*quadterm, GRB.MINIMIZE)
    m.setObjective(gb.quicksum(param), GRB.MINIMIZE)
    # m.setParam('BarHomogeneous', 1)
    m.optimize()
    sol = m.getAttr('X')
    # m.write('model_o.rlp')
    W_v=np.array([sol[i] for i in range(n_h)])
    c_v=sol[n_h]
    # for constr in m.getConstrs():
    #     print(constr)
    #print(sol)
    return sol,n_h,n_r,n,m.objVal,boundary_regions,zero_reg,H,zero_point




@njit
def finding_boundary_Regions(enumerate_poly, TH,hyperplanes,b,c,W,D):
    n=len(enumerate_poly[0][0])  # Get the number of elements in the first list of enumerate_poly
    V = np.zeros((0,n), dtype=np.float64)  # Pre-allocate an empty array for V
    index_list = []
    zero_point=[] 
    zero_reg = []
    H=[]
    regions = []  # Use a list to collect regions
    h=np.vstack((np.eye(n),-np.eye(n)))
    # np.array([i for i in ran])
    b_b=np.zeros((0,1))
    for f in range(n):
        for k in range(2):
            b_b=np.vstack((b_b,np.array([[TH[f]]])))
    # for f in range(n):
    #     bb=bb+[TH[f]]*n
    
    # b_b=np.array([[TH]]*(2*n))
        # vert=np.reshape(j,(len(j),1))
    for k, i in enumerate(enumerate_poly):
        index_list.extend([k]*len(i)) # Create an array of repeated indices
        # Extend V with elements from j
        V = np.vstack((V, i))
        # for j in range(len(i)):
        #     vert=np.copy(i[j:j+1,:].T)
            # if not np.any(np.array([0.0]*n)-i[j]):
            #     xdot=np.array([[0.0]]*n)
            # else:        
            #     xdot=np.dot(W,np.maximum(np.dot(hyperplanes,vert)+b,0))+np.reshape(c,(len(c),1))
        xdot=np.dot(W,np.maximum(np.dot(hyperplanes,i.T)+b,0))+np.reshape(c,(len(c),1))
        x=h@i.T+b_b
        y=1-np.sign(np.maximum(np.abs(x),1e-6)-1e-6)
            # y_prime=np.copy(y[:,0])
        # y_prime=y[:,0]
        if np.max(y)==1:
            regions.append(k)
            for j in range(np.shape(y)[0]):
                val=h[j,:]@xdot@np.diag(y[j,:])
                val_new=val[y[j,:]==1]
                for f in range(len(val)):
                    if np.abs(val[f])>=1e-10 and np.abs(val[f])<=2e-4:
                            zero_point.append(i[f])
                # if len(val_new)>0 and np.min(np.abs(val_new))<=2e-4:
                #     zero_point.append([i[f] for f in range(len(val)) if np.abs(val[f])>=1e-10 and np.abs(val[f])<=2e-4])                     
                if np.max(val)*np.min(val)<=-1e-6:
                    zero_reg.append(k)
                    h_new=h[j,:]@(W@np.diag(D[k])@hyperplanes)
                    b_new=h[j,:]@(W@np.diag(D[k])@b+c)-1e-4
                    H.append(np.hstack((h_new,b_new)))
            # val_prime=np.copy(val[:,0])
            # val_prime=val[:,0]
            # if np.min(np.abs(val_prime[y_prime==1]))<1e-8:
            #     zero_point.append(i[j])
            # if np.max(np.abs(j)) >= TH - 1e-6:
            #     # xdot=np.dot(W,np.maximum(np.dot(hyperplanes,j)+b.T,0).T)+np.reshape(c,(len(c),1))
            #     # if j@xdot>1e-8:
            #         regions.append(k)  # Append k to the list
    return np.unique(np.array(regions)),index_list,V,zero_reg,H,zero_point  # Convert to array and return unique values


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