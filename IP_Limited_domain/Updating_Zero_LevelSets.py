import numpy as np
import torch
from Enum_module_BF import updating_NN
from utils_n_old import finding_side
from numba import njit

def updating_BF_LV(NN,h,enumerate_poly,D):
    enumerate_poly_new=[]
    model=torch.jit.load(h)
    model.eval()
    model_original=torch.jit.load(NN)
    model_original.eval()
    hyperplanes=[]
    biases=[]
    hyperplanes=model.fc1.weight
    # all_hyperplanes=model.fc1.weight
    # all_bias=model.fc1.bias
    biases=model.fc1.bias
    W=model.out.weight
    weights=model_original.out.weight
    boundary_hyperplanes=[((model_original.fc1.weight).detach().cpu().numpy()).astype(np.float32)]
    border_bias=[((model_original.fc1.bias).detach().cpu().numpy()).astype(np.float32)]
    c=model.out.bias
    new_hype=[] 
    new_bias=[]
    A_dyn=[]
    B_dyn=[]
    all_hyperplanes=[]
    all_bias=[]
    with torch.no_grad():
        for j,i in enumerate(enumerate_poly):
            vertices=i
            vertices=torch.FloatTensor(vertices).cuda()
            h_val=model(vertices)
            if torch.max(h_val)>1e-8 and torch.min(h_val)<-1e-8:
                enum=enumerate_poly[j].astype(np.float32)
                sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                enumerate_poly_new.append(i)
                new_hype.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                new_bias.extend((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
            elif torch.min(h_val)>-1e-10:
                enum=enumerate_poly[j].astype(np.float32)
                sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                enumerate_poly_new.append(i)
                A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())

                # all_hyperplanes=torch.cat([all_hyperplanes,new_hype],0)
                # all_bias=torch.cat([all_bias,new_bias],0)
                # weights=torch.cat([weights,torch.FloatTensor(np.zeros((2,len(new_hype)))).cuda()],1)


    # NN,_,_,_,_=updating_NN(NN,all_hyperplanes.size()[1],all_hyperplanes,all_bias,weights,c)
    return enumerate_poly_new,np.array(new_hype).squeeze(1),np.array(new_bias),A_dyn,B_dyn,np.array(all_hyperplanes).squeeze(1),np.array(all_bias)


    
            



# @njit
def finding_side_new(boundary_hyperplanes,enumerate_poly,border_bias):
    # side=list()
    # hyp_f=List()
    # side=List()
    side=[]
    hyp_f=[]
    n=len(boundary_hyperplanes[0])
    # test=np.reshape(border_bias,(len(border_bias),1))
    # test=border_bias.reshape((len(border_bias),-1))
    dum=np.dot(boundary_hyperplanes,enumerate_poly.T)+border_bias.reshape((len(border_bias),-1))
    # dum=np.dot(boundary_hyperplanes,(np.array(enumerate_poly)).T)+test
    for j,i in enumerate(dum):
        res=[k for k,l in enumerate(i) if np.abs(l)<1e-6]
        if len(res)>=n:
            # if res not in side:
            side.append(((res)))
            hyp_f.append((np.append(boundary_hyperplanes[j],border_bias[j])))
                # vertices=(dum[j])[dum[j]<1e-10 and dum[j]>-1e-10]
    if len(hyp_f)!=2*len(enumerate_poly):
        print("Error in finding the side")
    return side,hyp_f