from Optimization_SISE_test import check_sol_AGIS
from Updating_Zero_LevelSets import updating_BF_LV,finding_boundary_vertice,updating_BF_LV_n,polishing_regions,cells_ordering,finding_all_info
import numpy as np
from numba.typed import List
from utils_Enumeration import Enumerator_rapid
from utils_n_old import Finding_Indicator_mat
from Enum_module_BF import Finding_Barrier, Finding_Lyap_Invariant,updating_NN,updating_NN_Original
from Optimization_SISE_test import SIS_opt,SIS_opt_new
from Refinement_SISE import Refinement
import pandas as pd
import torch
import time
import os
import csv
from Enum_module_BF import generate_hypercube_vertices
def SISE_algorithm(NN,h,enumerate_poly,D,all_hyperplanes,all_bias,W_h,c_h,W_x,c_x,TH,border_hype,border_bias,zeros,eps1,eps2,alpha,parallel,NN_file,iter,bound):
    # enumerate_poly,all_hyperplanes_x,all_bias_x,W_x,c_x,border_hype,border_bias,D_x=initial_Enum(NN_file_original,TH,mode,parallel)
    state=True
    while state:
        n=np.shape(enumerate_poly[0])[1]
        index,V=cells_ordering(enumerate_poly)
        _,_,_,_,points_all,region_all=updating_BF_LV(NN,h,enumerate_poly,D,all_hyperplanes,all_bias,W_h,c_h,border_hype,border_bias,TH)
        points=np.zeros((0,n))
        _,_,b_reg,lab=finding_all_info(index,V,points,all_hyperplanes,all_bias,W_h,D,c_h,TH)
        AGIS_PN,AGIS_RN,NAGIS_PN,NAGIS_RN=polishing_regions(points_all,region_all,W_h,c_h,NN,D,enumerate_poly,eps2,TH,all_hyperplanes)
        sol_iter,n_h,n_b,n_AGIS,n_NAGIS=SIS_opt_new(enumerate_poly,D,eps1,eps2,TH,alpha[0],h,NN,AGIS_PN,AGIS_RN,NAGIS_PN,NAGIS_RN,index,V,b_reg,all_hyperplanes,all_bias,W_h,c_h,iter)
        if len(AGIS_PN)==0:
            raise ValueError("No more improvement is possible with the current Epsilon value")
        state=check_sol_AGIS(sol_iter,n_h,n_b,n_AGIS,n_NAGIS,eps1)
        W_v=sol_iter[0:n_h]
        W_v=np.reshape(W_v,(1,n_h))
        c_v=sol_iter[n_h]
        slack_var=sol_iter[n_h+1:n_h+1+n_b+n_AGIS+n_NAGIS+1]
        if not state:
            # print("Accumulative enumeration time=\n",enumeration_time)
            print("Number of hyperplanes:\n",n_h)
            print("Number of cells:\n",len(enumerate_poly))
            print('Solution is found')
            # print("Seacrching for the Barrier function:\n",end_process-start_process)
            a=0.1
            name="_BF_updated"
            # val=W_v@np.maximum(all_hyperplanes@np.array(zeros).T+all_bias,0)+c_v
            # c_v=c_v-0.5*np.max(val)
            BF_NN,_,_,_,_=updating_NN(NN_file,n,all_hyperplanes,all_bias,W_v,c_v,name)
            NN,_,_,_,_=updating_NN_Original(NN,all_hyperplanes,all_bias,n,W_x,c_x)
        else:
            tau_b=sol_iter[n_h+1:n_h+1+n_b]
            tau_agis=sol_iter[n_h+1+n_b:n_h+1+n_b+n_AGIS]
            tau_nagis=sol_iter[n_h+1+n_b+n_AGIS:n_h+1+n_b+n_AGIS+n_NAGIS]
            all_hyperplanes,all_bias,new_hype,new_bias,enumerate_poly,W_x=Refinement(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol_iter,W_x,c_x,eps1,D,b_reg,tau_b,tau_agis,AGIS_RN,tau_nagis,NAGIS_RN)
            enumerate_poly,border_hype,border_bias=Enumerator_rapid(new_hype,new_bias,enumerate_poly,TH,[border_hype],[border_bias],parallel)
            print("number of the cells:",len(enumerate_poly))
            D=Finding_Indicator_mat(List(enumerate_poly), all_hyperplanes, all_bias)
            D[D>0]=1
            D[D<0]=0
            # D_x=D
            W_app=np.zeros((1,2*len(new_hype)))
            W_h=np.hstack((W_h, W_app))
            NN,_,_,_,_=updating_NN_Original(NN,all_hyperplanes,all_bias,n,W_x,c_x)
            name="_BF_updated"
            h,_,_,_,_=updating_NN(NN_file,n,all_hyperplanes,all_bias,W_h,c_h,name)



    # state=True

    # while state:
    #     new_hype,new_bias,all_hype,all_b,points_all,region_all=updating_BF_LV(NN,h,enumerate_poly,D,all_hyperplanes,all_bias,W_h,c_h,border_hype,border_bias)
    #     enumerate_poly,border_hype,border_bias=Enumerator_rapid(new_hype,new_bias,enumerate_poly,TH,[border_hype],[border_bias],parallel)
    #     D_new=Finding_Indicator_mat(List(enumerate_poly),all_hype,all_b)
    #     D_new[D_new>0]=1
    #     D_new[D_new<0]=0
    #     # D_test=csr_matrix(D_new)
    #     n=np.shape(enumerate_poly[0])[1]
    #     W_h=np.hstack((W_h,np.zeros((1,len(new_hype)*2))))
    #     name="_BF_SISE"
    #     h_n,_,_,_,_=updating_NN(NN_file,n,all_hype,all_b,W_h,c_h,name)
    #     W_x=np.hstack(((W_x),np.zeros((2,len(new_hype)*2))))
    #     Original_NN,_,_,_,_=updating_NN_Original(NN_file,all_hype,all_b,n,W_x,c_x)
    #     # updating_BF_LV_n(NN,h_n,polytope,D_new,new_hype)
    #     AGIS_points,AGIS_region,NAGIS_points,NAGIS_region,index,V,b_reg,label=finding_boundary_vertice(enumerate_poly,h_n,Original_NN,W_h,c_h,all_hype,all_b,D_new,TH,eps2)
    #     if len(AGIS_points)==0:
    #         raise ValueError("No more improvement is possible with the current Epsilon value")
    #     sol_iter,n_h,n_b,n_AGIS,n_NAGIS=SIS_opt(enumerate_poly,D_new,eps1,eps2,TH,alpha[0],h_n,Original_NN,AGIS_points,AGIS_region,NAGIS_points,NAGIS_region,index,V,b_reg,label)
    #     # sol,n_h,n_r,n,obj_function,boundary_regions,zero_reg,H,zero_point=preprocessing_BF(polytope,D_new,W_x,c_x,all_hype,all_b,eps1,eps2,TH,alpha[0])
    #     state=check_sol_AGIS(sol_iter,n_h,n_b,n_AGIS,n_NAGIS,eps1)
    #     W_v=sol_iter[0:n_h]
    #     W_v=np.reshape(W_v,(1,n_h))
    #     c_v=sol_iter[n_h]
    #     slack_var=sol_iter[n_h+1:n_h+1+n_b+n_AGIS+n_NAGIS+1]
    #     if not state:
    #         # print("Accumulative enumeration time=\n",enumeration_time)
    #         print("Number of hyperplanes:\n",n_h)
    #         print("Number of cells:\n",len(enumerate_poly))
    #         print('Solution is found')
    #         # print("Seacrching for the Barrier function:\n",end_process-start_process)
    #         a=0.1
    #         name="_BF_updated"
    #         BF_NN,_,_,_,_=updating_NN(NN_file,n,all_hype,all_b,W_v,c_v,name)
    #     else:
    #         tau_b=sol_iter[n_h+1:n_h+1+n_b]
    #         tau_agis=sol_iter[n_h+1+n_b:n_h+1+n_b+n_AGIS]
    #         all_hyperplanes,all_bias,new_hype,new_bias,enumerate_poly,W=Refinement(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol_iter,W_x,c_x,eps1,D,b_reg,tau_b,tau_agis,AGIS_region)
    return NN,h,all_hyperplanes,all_bias,W_x,c_x,enumerate_poly ,D ,border_hype,border_bias,W_v,c_v
    # return NN,h



def initial_Enum(NN_file,TH,mode,parallel):
    if NN_file[-4:]=="xlsx":
            hyperplanes=np.array(pd.read_excel(NN_file,sheet_name='1'))
            n=np.shape(hyperplanes)[1]
            b=np.array(pd.read_excel(NN_file,sheet_name='2'))
            W=np.array(pd.read_excel(NN_file,sheet_name='3'))
            c=np.array(pd.read_excel(NN_file,sheet_name='4'))
            h_append=np.array([[1.0]*n])
            b_append=np.array([0.0]*n)
            W_append=np.zeros((n,1))
            h_append=np.eye(n)
            b_append=np.array(b_append)
            W_append=np.zeros((n,len(h_append)))
            hyperplanes=np.append(hyperplanes,np.array(h_append),axis=0)
            b=np.append(b,b_append)
            W=np.append(W,W_append,axis=1) 
    else:
        model = torch.jit.load(NN_file)
    #knowing number of neurons in each layer
        cntr=0
        params=[]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name, param in model.named_parameters():
            with torch.no_grad():
                if device.type=='cuda':
                    param=param.cpu()
                    param=param.numpy()
                    params.append(param)
                else:
                    params.append(param.numpy())
            cntr=cntr+1
        num_hidden_layers = ((cntr-4)/2)+1
        print(num_hidden_layers)
                

        hyperplanes=[]
        b=[]
        W=[]
        c=[]
        nn=[]
        for i in range(len(params)-2):
                if i%2==0:
                    hyperplanes.extend(params[i])
                    nn.append(np.shape(params[i])[0])
                else:
                    b.extend(params[i])
        hyperplanes=np.array(hyperplanes)
        b=np.array(b)
        W=params[-2]
        c=params[-1]
        c=np.reshape(c,(len(c),1))

    # c=np.array([[0],[0]])
    n_h,n=np.shape(hyperplanes)
    original_polytope_test=np.array([generate_hypercube_vertices(n,TH,-TH)])
    cwd=os.getcwd()
    print(cwd)
    if mode=="Low_Ram":
        csv_file=cwd+'\Results'+'\Enumerate_poly_'+name+'.csv'
        with open (csv_file,'w',newline='') as f:
            wtr = csv.writer(f)
            wtr.writerows(original_polytope_test)
    border_hyperplane=np.vstack((np.eye(n),-np.eye(n)))
    border_bias=[-TH]*np.shape(border_hyperplane)[0]
    border_hype_org=np.copy(border_hyperplane)
    
    all_hyperplanes=np.append(hyperplanes,-hyperplanes,axis=0)
    all_hyperplanes=np.append(all_hyperplanes,border_hyperplane,axis=0)
    all_bias=np.append(b,-b)
    all_bias=np.reshape(np.append(all_bias,np.array([TH]*(2*n))),(len(all_hyperplanes),1))
    W_append=np.zeros((n,len(hyperplanes)+len(border_hyperplane)))
    W=np.append(W,W_append,axis=1)
    status=True
    enumeration_time=0
    # alpha=0.0025
    iter=0
    start_process=time.time()
    st_enum=time.time()
    enumerate_poly,border_hyperplane,border_bias=Enumerator_rapid(hyperplanes,b,original_polytope_test,TH,[border_hyperplane],[border_bias],parallel)
    end_enum=time.time()
    enumeration_time=enumeration_time+(end_enum-st_enum)
    # plot_res.plot_polytope(enumerate_poly,"blue")
    print("number of the cells:",len(enumerate_poly))
    # st=time.time()
    D=Finding_Indicator_mat(List(enumerate_poly),all_hyperplanes,all_bias)
    D[D>0]=1
    D[D<0]=0
    
    return enumerate_poly,all_hyperplanes,all_bias,W,c,border_hyperplane,border_bias,D
