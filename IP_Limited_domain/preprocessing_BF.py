import gurobipy as gb
from gurobipy import GRB
import numpy as np
import numba
from numba import njit
def preprocessing_BF(enumerate_poly,D,W,c,hyperplanes,b,eps1,eps2,TH,alpha):
    n_h,n=np.shape(hyperplanes)
    n_r=len(enumerate_poly)
    boundary_vertices=[]
    # boundary_regions=[]
    boundary_regions_org,index_list,V,Xdot,states,reg_out=finding_boundary_Regions(enumerate_poly,TH,hyperplanes,b,c,W)
    #### To DO: number of var= n_h+3  one for boundary reg and one for interior
    #### To DO: for boundary cell use 1-|sgn(x)| for boundary cells
    n_b=len(boundary_regions_org) # number of boundary regions
    n_o=len(reg_out)
    n_var=n_h+1+n_b+1+n_o
    m=gb.Model("linear") 
    x={}
    for i in range(n_var):
        if i <n_h+1:
            x[i] = m.addVar(lb=-float('inf'),name=f"x[{i}]")
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
    tau_int=x[n_h+n_b+n_o+1]
    var_bi=[x[i] for i in range(n_h+1,n_h+1+n_b)]
    var_bo=[x[i] for i in range(n_h+1+n_b,n_h+1+n_b+n_o)]
    # tau_b=x[n_h+2]
    buffer=[]
    # ReLU_val=np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0)
    # dot_x=np.dot(W,np.maximum(np.dot(hyperplanes,np.array(V).T)+b,0))+np.reshape(c,(len(c),1))
    for j,i in enumerate(V):
        h=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+var_c
        xdot=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
        xdot_test=Xdot[j,:]
        if i.tolist()==[0]*n:
            xdot=np.array([[0.0]]*n)
        # state=Finding_valid_boundary(TH,n,i,xdot)
        if i.tolist() not in buffer:
            buffer.append(i.tolist())
            if (states[j]=="int") and (index_list[j] not in boundary_regions_org):
                eq=h+tau_int
                m.addConstr(eq[0][0]>=eps1,name=f"PI")    
            elif states[j]=="bO" :
                id=np.where(reg_out==index_list[j])[0][0]
                eq=h-var_bo[id]
                # eq=h+tau_int
                m.addConstr(eq[0][0]<=-eps1,name=f"NB")
            else:
                id=np.where(boundary_regions_org==index_list[j])[0][0]
                eq=h+var_bi[id]
                # eq=h+tau_int
                m.addConstr(eq[0][0]>=eps1,name=f"bI")


                # id=np.where(boundary_regions==index_list[j])[0][0]
                # eq=h-x[n_h+1+id]
                # m.addConstr(eq[0][0]<=-eps1,name=f"NB")
                # m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
                # if i@xdot>1e-8:

                # m.addConstr(eq[0][0]<=-eps1,name=f"NB")
                # m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
                # boundary_vertices.append(i)
                # boundary_regions.append(index_list[j])
                
                    # m.addConstr(eq[0][0]>=-1000*eps1,name=f"NB")
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

        # else:
            # dot_x_test=np.dot(W,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+np.reshape(c,(len(c),1))
            # eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),dot_x[:,j:j+1]))-x[n_h+1+index_list[j]]
        eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),xdot))+alpha*(h)
        m.addConstr(eq[0][0]>=eps2,name=f"PD")

    param=[x[i] for i in range (n_h+1,n_var)]
    #m.addConstrs(x[i]>=1e-12 for i in range (0,n_list[0]-2*n_r))
    m.setObjective(10*len(boundary_regions_org)*gb.quicksum(var_bo)+gb.quicksum(var_bi)+tau_int, GRB.MINIMIZE)
    # m.setObjective(1000*tau_b+tau_int, GRB.MINIMIZE)
    m.setParam('BarHomogeneous', 1)
    m.optimize()
    sol = m.getAttr('X')
    # m.write('model_o.rlp')
    W_v=np.array([sol[i] for i in range(n_h)])
    c_v=sol[n_h]
    # for constr in m.getConstrs():
    #     print(constr)
    #print(sol)
    return sol,n_h,n_r,n,boundary_vertices,boundary_regions_org,reg_out




# @njit
def finding_boundary_Regions(enumerate_poly, TH,hyperplanes,b,c,W):
    n=len(enumerate_poly[0][0])  # Get the number of elements in the first list of enumerate_poly
    V = np.zeros((0,n), dtype=np.float64)  # Pre-allocate an empty array for V
    Xdot=np.zeros((0,n), dtype=np.float64)
    index_list = []  # Pre-allocate an empty array for index_list
    regions = []  # Use a list to collect regions
    reg_out=[]
    states=[]
    for k, i in enumerate(enumerate_poly):
        index_list.extend([k]*len(i)) # Create an array of repeated indices
        V = np.vstack((V, i))
        xdot=np.dot(W,np.maximum(np.dot(hyperplanes,i.T)+b,0))+np.reshape(c,(len(c),1))
        Xdot = np.vstack((Xdot, xdot.T))
        # state="int"
        h=np.vstack((np.eye(n),-np.eye(n)))
        b_b=np.array([[TH]]*(2*n))
        # vert=np.reshape(j,(len(j),1))
        x=h@i.T+b_b
        y=1-np.sign(np.maximum(np.abs(x),1e-6)-1e-6)
        # if np.max(y)==1:
        state=1
        for j in range(np.shape(y)[1]):
            if np.max(y[:,j])<1:
                state=1*state
            if np.min(np.diag(y[:,j])@h@xdot)>-1e-8:
                state=1*state
            else:
                state=0*state
                reg_out.append(k)
        regions.append(k)  # Append k to the list
        if np.min(np.diag(y[:,0])@h@xdot)>-1e-8:
            state="bI"
        else:
            state="bO"
            reg_out.append(k) 
        # for j in range(len(i)):
        #     vert=np.copy(i[j:j+1,:].T)
        #     if not np.any(np.array([0.0]*n)-i[j]):
        #         xdot=np.array([[0.0]]*n)
        #     else:        
        #         xdot=np.dot(W,np.maximum(np.dot(hyperplanes,vert)+b,0))+np.reshape(c,(len(c),1))
        #         # xdot=np.dot(hyperplanes,vert)
        #     # Extend V with elements from j
        #     # Concatenate the values
        #     Xdot = np.vstack((Xdot, xdot.T))
        #     state="int"
        #     h=np.vstack((np.eye(n),-np.eye(n)))
        #     b_b=np.array([[TH]]*(2*n))
        #     # vert=np.reshape(j,(len(j),1))
        #     x=h@vert+b_b
        #     y=1-np.sign(np.maximum(np.abs(x),1e-6)-1e-6)

        #     if np.max(y)==1:
        #         regions.append(k)  # Append k to the list
        #         if np.min(np.diag(y[:,0])@h@xdot)>-1e-8:
        #             state="bI"
        #         else:
        #             state="bO"
        #             reg_out.append(k)
        #     # if not state:
        #     #     regions.append(k)  # Append k to the list
        #     states.append(state)    
                # for j in i:
                #     if np.max(np.abs(j)) >= TH - 1e-6:
                #     # xdot=np.dot(W,np.maximum(np.dot(hyperplanes,j)+b.T,0).T)+np.reshape(c,(len(c),1))
                #     # if j@xdot>1e-8:
                #         regions.append(k)  # Append k to the list
    return np.unique(np.array(regions)),index_list,V,Xdot,states,np.unique(np.array(reg_out))  # Convert to array and return unique values


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
# @njit
def Finding_valid_boundary(TH,n,vertex,dotx):
    # I am checking if the vector filed is pointing inside the polytope on the boundary
    # IF state is true then the vector field is pointing inside the polytope
    state=True
    h=np.vstack((np.eye(n),-np.eye(n)))
    b=np.array([[TH]]*(2*n))
    vert=np.reshape(vertex,(len(vertex),1))
    x=h@vert+b
    y=1-np.sign(np.maximum(np.abs(x),1e-6)-1e-6)
    if np.max(y)==1:
        if np.min(np.diag(y[:,0])@h@dotx)>-1e-8:
            state=True
        else:
            state=False
    return state
    # elif np.min(np.diag(y[:,0])@h@dotx)<0:
    #     print("can not be inside")


