from copy import copy
import numpy as np
# from equality_constraints import   equlaity_constraints
def preprocess(V,A_dyn,n,H):
    # number of region
    print("Preprocessing...")
    n_r=len(V)
    n_list=[]
    #number of variable
    id_var,cell_info,nvar=find_idvar(n_r,n,V)
    # adding slack variables        
    n_list.append(nvar+n_r)
    V_new=V.copy()
    #  creating Empty matrix of inequalites
    for i in range(n_r):
        if (id_var[i][1]-id_var[i,0]==n+1):
            V_new[i]=np.insert(V_new[i],n,1,axis=1)
        # Equality 
    
    #neighbors_list,index_list,vert_list,neighbor_info=equlaity_constraints(V)
    A_sub_new,A_val_new,n_list,A_PD,neighbor_infon=forming_constraints(V,V_new,n_list,id_var,A_dyn,H,n,nvar,n_r)
    qsubi,qsubj,qval=Cost_function(n_list)
   
    return A_sub_new,A_val_new,qsubi,qsubj,qval,n_list,id_var,A_PD,n_r,cell_info,neighbor_infon
    



def find_idvar(n_r,n,V):    
    # Initialize variables
    cell_info = []
    nvar = 0
    id_var = np.zeros((n_r,2))
    
    # Loop through the rows of the matrix V
    for i in range(n_r):
        # Check if the row is equal to a row of zeros
        log_origin = list(np.zeros(n)) in V[i].tolist()
        
        # If the row is equal to a row of zeros, update nvar and cell_info
        if (log_origin == True):
            id_var[i] = [nvar, nvar + n]
            nvar = nvar + n
            cell_info.append((1))
        
        # If the row is not equal to a row of zeros, update nvar and cell_info
        else:
            id_var[i] = [nvar, nvar + n + 1]
            nvar = nvar + n + 1
            cell_info.append((0))
    
    # Return the results
    return id_var, cell_info, nvar



def Cost_function(n_list):
    qsubi = np.arange(n_list[0])
    qsubj = np.arange(n_list[0])
    qval = np.zeros(n_list[0], dtype=int)
    return qsubi,qsubj,qval

def finding_sparse_id(ls1,ls2):
    ls1_n=[ls1[i] for i, e in enumerate(ls2) if e != 0]
    ls2_n=[e for i, e in enumerate(ls2) if e != 0]
    return ls1_n,ls2_n

def forming_constraint_info(A_sub_row, A_val_row, nvar):
    # Initialize two empty lists with nvar number of sublists
    a_sub_new = [[] for i in range(nvar)]
    a_val_new = [[] for i in range(nvar)]
    
    # Iterate over all sublists in A_sub_row
    for j in range(len(A_sub_row)):
        # Iterate over all elements in each sublist in A_sub_row
        for i in A_sub_row[j]:
            # Append the index j to the sublist in a_sub_new at index i
            a_sub_new[int(i)].append(j)
            # Append the value at the index (j, i) in A_val_row to the sublist in a_val_new at index i
            a_val_new[int(i)].append(A_val_row[j][A_sub_row[j].index(i)])
    
    # Return the two lists a_sub_new and a_val_new
    return a_sub_new, a_val_new

def forming_constraints(V,V_new,n_list,id_var,A_dyn,H,n,nvar,n_r):
        V_list=[]
        V_new_list=[]
        index_list=[]
        [(index_list.extend([i]*len(j)),V_list.extend(j.tolist())) for i,j in enumerate(V)]
        #[V_list.extend(i.tolist()) for i in V]
        [V_new_list.extend(i.tolist()) for i in V_new]
        # We used buffer to save the duplicate vetices. buffer_index is for their indices. 
        buffer=[]
        buffer_index=[]
        buffer_index_h=[]
        A_PD=[]
        A_sub_PD=[]
        A_sub_ND=[]
        A_val_ND=[]
        A_val_PD=[]
        A_sub_EQ=[]
        A_val_EQ=[]
        neighbor_infon=[]
        [neighbor_infon.append([i]) for i in range(n_r)]
        for index,i in enumerate(V_list):
            if i!=[0]*n:
                
                # dum_ND=np.zeros(n_list[0])
                # dum_eq=np.zeros(n_list[0])
                if (i not in buffer):
                    if n==2:
                        dum_PD=np.zeros(n_list[0])
                        dum_PD[int(id_var[index_list[index]][0]):int(id_var[index_list[index]][1])]=V_new_list[index]
                        A_PD.append((dum_PD))
                    buffer.append((i))
                    buffer_index.append((index))
                    buffer_index_h.append([index])
                    r=np.arange(id_var[index_list[index]][0],id_var[index_list[index]][1],dtype=int)
                    l=[*r]
                    a_sub=l
                    a_val=V_new_list[index]
                    a_sub,a_val=finding_sparse_id(a_sub,a_val)
                    A_sub_PD.append((a_sub))
                    A_val_PD.append((a_val))
                else:
                    dum=buffer.index(i)
                    first_reg=index_list[buffer_index[dum]]
                    #buffer_index_h[dum].extend(([index]))
                    # dum_eq[int(id_var[first_reg][0]):int(id_var[first_reg][1])]=V_new_list[buffer_index[dum]]
                    # dum_eq[int(id_var[index_list[index]][0]):int(id_var[index_list[index]][1])]=np.negative(V_new_list[index])
                    r1=np.arange(id_var[first_reg][0],id_var[first_reg][1],dtype=int)
                    r2=np.arange(id_var[index_list[index]][0],id_var[index_list[index]][1],dtype=int)
                    l1=[*r1]
                    l2=[*r2]
                    a_sub=l1+l2
                    r1=V_new_list[buffer_index[dum]]
                    r2=np.negative(V_new_list[index])
                    a_val=r1+[*r2]
                    a_sub,a_val=finding_sparse_id(a_sub,a_val)
                    A_sub_EQ.append((a_sub))
                    A_val_EQ.append((a_val))
                    sec_reg=index_list[index]
                    list_neighbor=[]
                    # if sec_reg not in neighbor_infon[first_reg]:
                    #     neighbor_infon[first_reg].append((sec_reg))
                    # if first_reg not in neighbor_infon[sec_reg]: 
                    #     neighbor_infon[sec_reg].append((first_reg))
                    for k in buffer_index_h[dum]:
                        first=index_list[k]
                        list_neighbor.append((first))
                        if sec_reg not in neighbor_infon[first]:
                            neighbor_infon[first].append((sec_reg)) 
                    neighbor_infon[sec_reg].extend((list_neighbor))
                    neighbor_infon[sec_reg]=list(set(neighbor_infon[sec_reg]))
                    list(set(neighbor_infon[sec_reg]))
                    buffer_index_h[dum].extend(([index]))
                # dum_ND[int(id_var[index_list[index]][0]):int(id_var[index_list[index]][0])+n]=np.matmul(A_dyn[index_list[index]],np.transpose(i))+H[index_list[index]]
                # dum_ND[int(index_list[index]+nvar)]=-1
                r1=np.arange(id_var[index_list[index]][0],id_var[index_list[index]][0]+n,dtype=int)
                l1=[*r1]+[index_list[index]+nvar]
                a_sub=l1
                l2=np.matmul(A_dyn[index_list[index]],np.transpose(i))+H[index_list[index]]
                a_val=l2.tolist()+[-1.0]
                a_sub,a_val=finding_sparse_id(a_sub,a_val)
                A_sub_ND.append((a_sub))
                A_val_ND.append((a_val))
        A_sub_row=A_sub_PD+A_sub_ND
        A_val_row=A_val_PD+A_val_ND
        n_list.append(len(A_sub_PD))
        n_list.append(len(A_sub_ND))
        del A_sub_ND,A_val_ND,A_sub_PD,A_val_PD
        if n_r>1:
            A_sub_row=A_sub_row+A_sub_EQ
            A_val_row=A_val_row+A_val_EQ
            n_list.append(len(A_sub_EQ))
        A_sub_new,A_val_new=forming_constraint_info(A_sub_row,A_val_row,nvar+n_r)

        return A_sub_new,A_val_new,n_list,A_PD,neighbor_infon
        



