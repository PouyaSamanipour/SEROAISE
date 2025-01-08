import gurobipy as gb
from gurobipy import GRB
import numpy as np
import numba
from numba import njit
import torch


def SIS_opt(enumerate_poly,D,eps1,eps2,TH,alpha,h,dx,AGIS_points,AGIS_region,NAGIS_points,NAGIS_region,index_list,V,boundary_regions,coeff):
    with torch.no_grad():
        dx_model=torch.jit.load(dx)
        dx_model.eval()
        h_model=torch.jit.load(h)
        h_model.eval()
        W_h=(h_model.fc1.weight).cpu().numpy()
        c_h=(h_model.fc1.bias).cpu().numpy()
        hyperplanes=(h_model.fc1.weight).cpu().numpy()
        b=(h_model.fc1.bias).cpu().numpy()
        W=(dx_model.out.weight).cpu().numpy()
        c=(dx_model.out.bias).cpu().numpy()
    n_h,n=np.shape(hyperplanes)
    # boundary_regions,index_list,V,zero_reg,H,zero_point=finding_boundary_Regions(enumerate_poly,TH,hyperplanes,b,c,W,D)
    n_AGIS=len(AGIS_points)
    n_b=len(boundary_regions) # number of boundary regions
    if len(AGIS_points)!=0:
        n_NAGIS=1
    else:
        n_NAGIS=0
    n_var=n_h+1+n_b+n_AGIS+n_NAGIS+2
    m=gb.Model("linear") 
    x={}
    for i in range(n_var):
        if i <n_h+1:
            x[i] = m.addVar(lb=-float('inf'),name=f"x[{i}]")
        else:
            x[i] = m.addVar(lb=1e-12,name=f"Slack[{i}]")
    #var=[s,t,tau_b,tau_AGIS,tau_NAGIS,tau_NAGIS,tau_int,tau_uc]
    var_w=[x[i] for i in range(n_h)]
    var_w=np.reshape(var_w,(-1,n_h))
    var_c=x[n_h]
    buffer=[]
    tau_b=[x[i] for i in range(n_h+1,n_h+1+n_b)]
    tau_AGIS=[x[i] for i in range(n_h+1+n_b,n_h+1+n_b+n_AGIS)]
    tau_NAGIS=[x[i] for i in range(n_h+1+n_b+n_AGIS,n_h+1+n_b+n_AGIS+n_NAGIS)]
    tau_int=[x[i] for i in range(n_h+1+n_b+n_AGIS+n_NAGIS,n_h+1+n_b+n_AGIS+n_NAGIS+1)]
    tau_uc=[x[i] for i in range(n_h+1+n_b+n_AGIS+n_NAGIS+1,n_h+1+n_b+n_AGIS+n_NAGIS+2)]
    cnt_AGIS=0
    cnt_NAGIS=0
    for j,i in enumerate(V):
        xdot=W@np.maximum(np.dot(hyperplanes,i)+b.T,0).T+c
        h=np.dot(var_w,np.maximum(np.dot(hyperplanes,i)+b.T,0).T)+var_c
        h_val=W_h@np.maximum(np.dot(hyperplanes,i)+b.T,0).T+c_h
        #Check if the point is inside or outside the invariant set
        # with torch.no_grad():
        #     h_val=coeff*(h_model(torch.FloatTensor(i).cuda())).cpu().numpy()
        if np.max(np.abs(i))>=TH-1e-6:
            pos="boundary"
        elif h_val>=-1e-5:
            pos="int"
        else:
            pos="uc"
        if pos=="int":
            if finding_similar_vertex(i,AGIS_points):
                st,id=finding_similar_vertex_with_index(i,AGIS_points)
                if i.tolist() not in buffer:
                    buffer.append(i.tolist())
                    eq=h+tau_AGIS[cnt_AGIS]
                    m.addConstr(eq[0]>=1*eps1,name=f"AGIS")
                    for k in AGIS_region[cnt_AGIS]:
                        eq=np.dot(var_w,np.dot(np.dot(np.diag(D[k]),hyperplanes),xdot))+1*alpha*(h)
                        m.addConstr(eq[0]>=0,name=f"PD")
                    cnt_AGIS+=1
            elif i.tolist() in np.array(NAGIS_points).tolist():
                if i.tolist() not in buffer:
                    buffer.append(i.tolist())
                    eq=h+tau_NAGIS
                    m.addConstr(eq[0]>=0,name=f"NAGIS")
                    for k in NAGIS_region[cnt_NAGIS]:
                        eq=np.dot(var_w,np.dot(np.dot(np.diag(D[k]),hyperplanes),xdot))+1*alpha*(h)
                        m.addConstr(eq[0]>=0,name=f"PD")
                    cnt_NAGIS+=1
            else:
                if i.tolist() not in buffer:
                    eq=h+tau_int
                    m.addConstr(eq[0]>=eps1,name=f"INT")
                    buffer.append(i.tolist())
                eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),xdot))+1*alpha*(h)
                m.addConstr(eq[0]>=eps2,name=f"PD")
        elif pos=="uc":
            eq=np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),xdot))+1*alpha*(h)
            m.addConstr(eq[0]>=eps2,name=f"PD")
            if i.tolist() not in buffer:
                buffer.append(i.tolist())
                eq=h+tau_uc
                m.addConstr(eq[0]>=eps1,name=f"UC")
        else:
            eq=var_w@np.diag(D[index_list[j]])@hyperplanes@xdot+1*alpha*(h)
            # np.dot(var_w,np.dot(np.dot(np.diag(D[index_list[j]]),hyperplanes),xdot))+0.001*alpha*(h)
            m.addConstr(eq[0]>=eps2,name=f"PD")
            if i.tolist() not in buffer:
                buffer.append(i.tolist())
                eq=h-tau_b[(np.where(boundary_regions==index_list[j]))[0][0]]
                m.addConstr(eq[0]<=-eps1,name=f"NB")



    param=[x[i] for i in range (n_h+1,n_var)]
    #m.addConstrs(x[i]>=1e-12 for i in range (0,n_list[0]-2*n_r))
    m.setObjective(gb.quicksum(param), GRB.MINIMIZE)
    m.setParam('BarHomogeneous', 1)
    m.optimize()
    sol = m.getAttr('X')
    return sol,n_h,n_b,n_AGIS





    
