import os
import sys
import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from utils_n_old import Finding_Indicator_mat
from utils_n_old import checking_sloution
from numba import prange
from utils_n_old import Enumerator_rapid
from preprocessing_BF import preprocessing_BF
from plot_res_BF import plotting_results_BF
from Refinement_process_BF import Refinement
import torch
from training_process import training
import torch.nn.functional as F
import plot_res_BF
from preprocessing_LF import preprocessing_Lyap
from Refinement_process_Lyap import Refinement_Lyap
from plot_res_Lyap import plotting_results_lyap,plot_invariant_set_multiple,plot_level_set,plot_polytope_2D,plot_polytope
from Preprocessing_Lyap_test import preprocessing_Lyap_test
import preprocessing_BF_new

def Finding_Barrier(NN_file,name_fig,eps1,eps2,TH,mode,parallel,alph):
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
    while status:
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
        
        sol,n_h,n_r,n,obj_function,boundary_regions,zero_reg,H,zero_point=preprocessing_BF(enumerate_poly,D,W,c,all_hyperplanes,all_bias,eps1,eps2,TH,alph)
        W_v=sol[0:n_h]
        W_v=np.reshape(W_v,(1,n_h))
        c_v=sol[n_h]
        slack_var=sol[n_h+1:-1]
        status=checking_sloution(slack_var,eps1)
        if not status:
            end_process=time.time()
            print("Accumulative enumeration time=\n",enumeration_time)
            print("Number of hyperplanes:\n",n_h)
            print("Number of cells:\n",len(enumerate_poly))
            print('Solution is found')
            print("Seacrching for the Barrier function:\n",end_process-start_process)
            a=0.1
            name="_BF_updated"
            BF_NN,_,_,_,_=updating_NN(NN_file,n,all_hyperplanes,all_bias,W_v,c_v,name)
            Original_NN,_,_,_,_=updating_NN_Original(NN_file,all_hyperplanes,all_bias,n,W,c)
            # preprocessing_BF_new.preprocessing_BF(enumerate_poly,D,W,c,all_hyperplanes,all_bias,eps1,eps2,TH,alph)

            # model_h,alpha,state,Violations=training(enumerate_poly,D,NN_file,TH,a,W_v,c_v,all_hyperplanes,all_bias)
            # saving_results(W_v,all_hyperplanes,all_bias,c_v,name,eps1,eps2,n_r)
            # if n==2:

            #     plotting_results_BF(TH,all_hyperplanes,all_bias,c_v,W_v,W,c,enumerate_poly,name_fig)
                # plot_res.plot 
        else:
            # pass
            all_hyperplanes,all_bias,hyperplanes,b,original_polytope_test,W=Refinement(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol,W,c,eps1,D,boundary_regions,zero_reg,H,iter)
            # Original_NN,_,_,_,_=updating_NN_Original(NN_file,[hyperplanes],[b])
        iter=iter+1
    return Original_NN,BF_NN,all_hyperplanes,all_bias,W,c,enumerate_poly,D,border_hyperplane,border_bias,np.unique(zero_point,axis=0),W_v,c_v



def generate_hypercube_vertices(dimensions, lower_bound, upper_bound):
    if dimensions == 0:
        return [[]]

    vertices = []

    for vertex in generate_hypercube_vertices(dimensions - 1, lower_bound, upper_bound):
        for value in [upper_bound, lower_bound]:
            vertices.append([value] + vertex)

    return vertices







def updating_NN(NN_file,n,hype,b,W_v,c_v,name):
    import torch.nn as nn
    # # import torch
    # model = torch.jit.load(NN_file)
    # existing_layer = model.fc1
    # outer_layer = 1
    # n=len(new_hype[0][0])
    # new_weight = torch.nn.Parameter(torch.cat([existing_layer.weight, torch.FloatTensor(np.array(new_hype[0])).cuda()], dim=0))
    # new_bias = torch.nn.Parameter(torch.cat([existing_layer.bias, torch.FloatTensor(new_bias[0]).cuda()], dim=0))
    # outer_weight = torch.nn.Parameter(torch.cat([outer_layer.weight, torch.FloatTensor(np.zeros((n,len(new_hype[0])))).cuda()], dim=1))
    # outer_bias=outer_layer.bias
    # n_h=new_weight.size(0)
    # n_o=outer_weight.size(0)
    # # new_bias = torch.nn.Parameter(torch.cat([existing_layer.bias, torch.FloatTensor(new_bias[0]).cuda()], dim=0)) 
    # num_new_neurons = len(new_hype[0])
    # # Create a new linear layer with the modified weight and bias
    # new_layer = nn.Linear(existing_layer.weight.size(1), existing_layer.weight.size(0) + num_new_neurons)
    # new_layer.weight = new_weight
    # new_layer.bias = new_bias
    class Model(nn.Module):
        def __init__(self,in_features,h1,out_features):
            # 2 feature-->h1 N-->h2 N-->output (1)
            super().__init__()
            self.fc1=nn.Linear(in_features,h1)
            self.out=nn.Linear(h1,out_features)

        def forward(self,x):
            x=F.relu(self.fc1(x))
            x=self.out(x)
            return x
    n_h=len(hype)
    n_o=1
    updated_model=Model(n,n_h,n_o)
    if not torch.is_tensor(hype):
        new_weight=torch.FloatTensor(hype).cuda()
        new_bias=torch.FloatTensor(b).cuda()
        new_bias=torch.squeeze(new_bias)
        outer_weight=torch.FloatTensor(W_v).cuda()
        outer_bias=torch.FloatTensor(np.array([c_v])).cuda()
    else:
        new_weight=hype.cuda()
        new_bias=b.cuda()
        new_bias=torch.squeeze(new_bias)
        outer_weight=W_v.cuda()
        outer_bias=c_v.cuda()
    with torch.no_grad():
        updated_model.fc1.weight.copy_(new_weight)
        updated_model.fc1.bias.copy_(new_bias)
        updated_model.out.weight.copy_(outer_weight)
        updated_model.out.bias.copy_(outer_bias)  
    # extended_model = ExtendedModel(model, new_layer)
    # Save the extended model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MGPU=  updated_model.to(device)
    if NN_file[-3-len(name):-3]==name:
        status=True
    else:
        status=False
    if status:
        torch.jit.save(torch.jit.script(MGPU), NN_file)
    else:
        NN_file=NN_file[:-3]+name+".pt"
        torch.jit.save(torch.jit.script(MGPU), NN_file)

    return NN_file,[new_weight.cpu().detach().numpy()],[new_bias.cpu().detach().numpy()],outer_weight.cpu().detach().numpy(),outer_bias.cpu().detach().numpy()



def updating_NN_Original(NN_file,new_hype,new_bias,n,W,c):
    import torch.nn as nn
    # import torch
    # model = torch.jit.load(NN_file)
    # existing_layer = model.fc1
    # outer_layer = model.out
    # n=len(new_hype[0][0])
    # new_weight = torch.nn.Parameter(torch.cat([existing_layer.weight, torch.FloatTensor(np.array(new_hype[0])).cuda()], dim=0))
    # new_bias = torch.nn.Parameter(torch.cat([existing_layer.bias, torch.FloatTensor(new_bias[0]).cuda()], dim=0))
    # outer_weight = torch.nn.Parameter(torch.cat([outer_layer.weight, torch.FloatTensor(np.zeros((n,len(new_hype[0])))).cuda()], dim=1))
    # outer_bias=outer_layer.bias
    # n_h=new_weight.size(0)
    # n_o=outer_weight.size(0)
    # # new_bias = torch.nn.Parameter(torch.cat([existing_layer.bias, torch.FloatTensor(new_bias[0]).cuda()], dim=0)) 
    # num_new_neurons = len(new_hype[0])
    # # Create a new linear layer with the modified weight and bias
    # new_layer = nn.Linear(existing_layer.weight.size(1), existing_layer.weight.size(0) + num_new_neurons)
    # new_layer.weight = new_weight
    # new_layer.bias = new_bias
    class Model(nn.Module):
        def __init__(self,in_features,h1,out_features):
            # 2 feature-->h1 N-->h2 N-->output (1)
            super().__init__()
            self.fc1=nn.Linear(in_features,h1)
            self.out=nn.Linear(h1,out_features)

        def forward(self,x):
            x=F.relu(self.fc1(x))
            x=self.out(x)
            return x
    n_h=len(new_hype)
    n_o=n
    updated_model=Model(n,n_h,n_o)
    if not torch.is_tensor(new_hype):
        new_weight=torch.FloatTensor(new_hype).cuda()
        new_bias=torch.FloatTensor(new_bias).cuda()
        new_bias=torch.squeeze(new_bias)
        outer_weight=torch.FloatTensor(W).cuda()
        outer_bias=torch.FloatTensor(c).cuda()
        outer_bias=torch.squeeze(outer_bias)
    else:
        new_weight=new_hype.cuda()
        new_bias=new_bias.cuda()
        new_bias=torch.squeeze(new_bias)
        outer_weight=W.cuda()
        outer_bias=c.cuda()
    with torch.no_grad():
        updated_model.fc1.weight.copy_(new_weight)
        updated_model.fc1.bias.copy_(new_bias)
        updated_model.out.weight.copy_(outer_weight)
        updated_model.out.bias.copy_(outer_bias)  
    # updated_model=Model(n,n_h,n_o) 
    # with torch.no_grad():
    #     updated_model.fc1.weight.copy_(new_weight)
    #     updated_model.fc1.bias.copy_(new_bias)
    #     updated_model.out.weight.copy_(outer_weight)
    #     updated_model.out.bias.copy_(outer_bias)  
    # extended_model = ExtendedModel(model, new_layer)
    # Save the extended model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MGPU=  updated_model.to(device)
    if NN_file[-10:-3]=="updated":
        status=True
    else:
        status=False
    if status:
        torch.jit.save(torch.jit.script(MGPU), NN_file)
    else:
        NN_file=NN_file[:-3]+"_updated.pt"
        torch.jit.save(torch.jit.script(MGPU), NN_file)

    return NN_file,[new_weight.cpu().detach().numpy()],[new_bias.cpu().detach().numpy()],outer_weight.cpu().detach().numpy(),outer_bias.cpu().detach().numpy()





def Finding_Lyap_Invariant(NN_file,h,original_polytope_test,all_hyperplanes,all_bias,border_hyperplane,border_bias,hyperplanes,b,W,c,eps1,eps2,TH,parallel):
    enumeration_time=0
    n=np.shape(hyperplanes)[1]
    all_hyperplanes=np.append(all_hyperplanes,np.append(hyperplanes,-hyperplanes,axis=0),axis=0)
    # all_hyperplanes=np.append(all_hyperplanes,border_hyperplane,axis=0)
    # all_bias=np.append(b,-b)
    all_bias=np.reshape(np.append(all_bias,np.append(b,-b)),(len(all_hyperplanes),1))
    W_append=np.zeros((n,2*len(hyperplanes)))
    W=np.append(W,W_append,axis=1)
    status=True
    while status:
        st_enum=time.time()
        enumerate_poly,border_hyperplane,border_bias=Enumerator_rapid(hyperplanes,b,original_polytope_test,TH,[border_hyperplane],[border_bias],parallel)
        end_enum=time.time()
        enumeration_time=enumeration_time+(end_enum-st_enum)
        # plot_invariant_set(h,TH,'cyan')
        # plot_polytope(enumerate_poly,"test")
        # enumerate_poly=trimiming_polytope(enumerate_poly,h)

        

        #print(len(enumerate_poly))
        # st=time.time()
        D=Finding_Indicator_mat(List(enumerate_poly),all_hyperplanes,all_bias)
        D[D>0]=1
        D[D<0]=0
        sol,n_h,n_r,n,obj_function=preprocessing_Lyap_test(enumerate_poly,D,W,c,all_hyperplanes,all_bias,eps1,eps2,h)
        W_v=sol[0:n_h]
        W_v=np.reshape(W_v,(1,n_h))
        c_v=sol[n_h]
        slack_var=sol[n_h+1:]
        status=checking_sloution(slack_var,eps2)





        # st=time.time()
        
        # df1=pd.DataFrame(A_dyn)
        # # df1.to_excel("Path_following_70.xlsx",sheet_name="Sheet1")
        # df2=pd.DataFrame(B_dyn)
        # # df2.to_excel("Path_following_70.xlsx",sheet_name="Sheet2")
        # df3=pd.DataFrame(Vert)
        # # df3.to_excel("Path_following_70.xlsx",sheet_name="Sheet3")
        # with pd.ExcelWriter("Path_following_20.xlsx") as writer:
        #     df1.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        #     df2.to_excel(writer, sheet_name='Sheet2', index=False, header=False)
        #     df3.to_excel(writer, sheet_name='Sheet3', index=False, header=False)

        
        # status=False
        if not status:
            end_process=time.time()
            print("Accumulative enumeration time=\n",enumeration_time)
            print("Number of hyperplanes:\n",n_h)
            print("Number of cells:\n",len(enumerate_poly))
            print('Solution is found')
            # print("Seacrching for the Lyapuov function:\n",end_process-start_process)
            a=1.0
            name="_Lyap_BF_updated"
            NN_file,_,_,_,_=updating_NN(NN_file,n,all_hyperplanes,all_bias,W_v,c_v,name)
            # model_h,alpha,state,Violations=training(enumerate_poly,D,NN_file,TH,a,W_v,c_v,all_hyperplanes,all_bias)
            # saving_results(W_v,all_hyperplanes,all_bias,c_v,name,eps1,eps2,n_r)
            # if n==2:
            #     plotting_results_lyap(TH,all_hyperplanes,all_bias,c_v,W_v,W,c,enumerate_poly,name_figure) 
        else:
            # pass
            all_hyperplanes,all_bias,hyperplanes,b,original_polytope_test,W=Refinement_Lyap(enumerate_poly,all_hyperplanes,all_bias,slack_var,sol,W,c,eps2,D)

    return NN_file,all_hyperplanes,all_bias,W_v,c_v,enumerate_poly,D,border_hyperplane,border_bias


def trimiming_polytope(enumerate_poly,h):
    model=torch.jit.load(h)
    model.eval()
    enumerate_poly_new=[]
    for i in range(len(enumerate_poly)):
        vertices=enumerate_poly[i]
        vertices=torch.FloatTensor(vertices).cuda()
        h_val=model(vertices)
        if torch.min(h_val)>-1e-6:
            enumerate_poly_new.append(enumerate_poly[i])

    return enumerate_poly_new