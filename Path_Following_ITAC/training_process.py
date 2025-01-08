import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self,input_dim,hidden_units,output_units):
        # 2 feature-->h1 N-->h2 N-->output (1)
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_units)
        # self.fc3=nn.Linear(h2,h3)

        self.out=nn.Linear(hidden_units,output_units)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.out(x)
        return x
def Invariant_loss(h,inputs,a,S,W,hype,TH,W_v,c_v,bias_test):
    # dy_dx = torch.autograd.grad(outputs=y_pred, inputs=inputs,
    #                             grad_outputs=torch.ones_like(y_pred),
    #                             create_graph=True)[0]
    # # dy/dx=
    W_v=torch.FloatTensor(W_v).cuda()
    c_v=torch.FloatTensor([c_v]).cuda()
    n=len(hype[0])
    S=torch.FloatTensor(S).cuda()
    val=torch.FloatTensor([1.0]*(len(S[0]))).cuda()
    # S_prime=val-S
    # S_boundary=torch.ones(S_prime.size()[0],2*n).cuda()
    # S_new=torch.hstack((S_boundary,S,S_prime))
    # bias_test=bias_test.unsqueeze(1)
    indices=inputs[0][:,-1].long()
    D=torch.diag_embed(S[indices])
    # h_new=torch.zeros((len(inputs[0]),1)).cuda()
    # dh_test=torch.zeros((len(inputs[0]),n)).cuda()
    # for i in range(len(h_new)):
    #     h_new[i]=W_v@D[i]@(hype@inputs[0][i,0:n].T+bias_test).T+c_v
    #     dh_test[i]=W_v@D[i]@hype

    # # h_new=W_v@D@(hype@inputs[0][:,0:n].T+bias_test).T+c_v
    # dh_dx_test=W_v@D@hype
    dh_dx=W@D@hype
    dx_dt=inputs[0][:,n:2*n]
    dx_dt=dx_dt.unsqueeze(2)
    id=torch.reshape(inputs[0][:,-2],(len(inputs[0]),1))

    # constraint=id*torch.relu(h)
    #Constraint Lyaunov function
    constraint=(1-id)*torch.relu(-h-1e-5)
    constraint_origin=id*(torch.abs(h)-1e-5)
    # constraint[constraint==1e-6]=0
    violation_1=inputs[0][torch.where(constraint>0)[0]]
    constraint1 = (1-id)*(dh_dx@dx_dt-1e-5) # Lyapunov derivative
    # constraint1[constraint1>=-1e-6]=0
    violation_2=inputs[0][torch.where(constraint1>0)[0]]
    penalty = torch.max(torch.relu(constraint1))+torch.max(constraint)+torch.max(constraint_origin)  # Penalize if the constraint is violated
    # constraint_test=(1-id)*torch.relu(-h_new-1e-5)
    # constraint_origin_test=id*(torch.abs(h_new)-1e-5)
    # constraint1_test = (1-id)*(dh_dx_test@dx_dt-1e-5) # Lyapunov derivative
    # penalty_test=torch.max(torch.relu(constraint1_test))+torch.max(constraint_test)+torch.max(constraint_origin_test)
    # if penalty_test!=0:
    #     print("Violation")


    penalty1=torch.max(torch.relu(constraint1))+torch.max(constraint)+torch.max(constraint_origin)
    
    
    
    return penalty,penalty1,violation_1,violation_2

def creating_training_samples(polytope,NN,TH):
    n=len(polytope[0][0])
    model = torch.jit.load(NN)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        # W=W.cuda()
        # h=h.cuda()
        # b=b.cuda()
        # c=c.cuda()
        training_sample=torch.zeros((0,2*n+2)).cuda()
        for j,i in enumerate(polytope):
            # D=torch.tensor(S[j]).float().cuda()
            vertices=torch.tensor(i).float().cuda()
            id=torch.zeros(len(vertices),1).cuda()
            dx=torch.zeros(len(vertices),n).cuda()
            for k in range(len(vertices)):
                # if TH<=torch.max(torch.abs(vertices[k]))+1e-6:
                if torch.all(vertices[k]==0.0):
                    id[k]=1
                else:
                    dx[k]=model(vertices[k])   

            # dx=(W@torch.diag(D)@(h@vertices.T+torch.reshape(b,(len(b),1)))).T+c
            # dx=model(vertices)
            # dx[torch.abs(dx)<0.05]=0
            training_sample=torch.vstack((training_sample,torch.hstack((vertices,dx,id,j*torch.ones((len(vertices),1)).cuda()))))

    X=training_sample.cpu()
    X=X.detach().numpy()
    return X



def training(polytope,S,NN_file,TH,a,W_v,c_v,all_hype,all_bias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n=len(polytope[0][0])
    model = torch.jit.load(NN_file)
    hyperplanes=model.fc1.weight.data
    bias=model.fc1.bias.data
    W=model.out.weight.data
    c=model.out.bias.data
    X_train=creating_training_samples(polytope,NN_file,TH)
    # model = torch.jit.load(NN_file)
    # hyperplanes=model.fc1.weight.data
    # bias=model.fc1.bias.data
    boundary_hype=torch.vstack((torch.eye(n),-torch.eye(n))).cuda()
    boundary_bias=TH*torch.ones((2*n)).cuda()
    input_dim = n
    hidden_units = 2*len(hyperplanes)+2*n
    output_units = 1
    model1=Model(input_dim,hidden_units,output_units)
    # model1.fc1.weight.data = torch.vstack((hyperplanes,-hyperplanes,boundary_hype))
    # model1.fc1.bias.data = torch.hstack((bias,-bias,boundary_bias))
    model1.fc1.weight.data = torch.FloatTensor(all_hype).cuda()
    model1.fc1.bias.data = torch.FloatTensor(all_bias).squeeze().cuda()
    # model1.out.weight.data = torch.FloatTensor(W_v).cuda()
    model1.out.bias.data = torch.FloatTensor([c_v])


    model1.fc1.weight.requires_grad = False
    model1.fc1.bias.requires_grad = False
    model1.out.bias.requires_grad = False
    model1.to(device)
    batch_size =1024  # Adjust batch size as needed
    dataset = TensorDataset(torch.FloatTensor(X_train).cuda())
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # beta =torch.FloatTensor([a]).to(device).requires_grad_(True)
    # beta =torch.FloatTensor([a]).to(device)  
    # a.clone().detach().requires_grad_(True) 
    alpha = torch.FloatTensor([0]).to(device)
    num_epochs = 10000
    optimizer = torch.optim.Adam(list(model1.parameters()), lr=5e-2)
    loss=0
    status=True
    Violation=torch.zeros((0,2*n+2)).to(device)
    num_batch=(len(X_train)//batch_size)+1
    for epoch in range(num_epochs):
        for i, (batch_X) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred=model1.forward(batch_X[0][:,0:n])
            W1 = model1.out.weight
            hype=model1.fc1.weight
            bias_test=model1.fc1.bias
            if (epoch==num_epochs-1):
                Loss,loss1,v1,v2=Invariant_loss(y_pred,batch_X, alpha,S,W1,hype,TH,W_v,c_v,bias_test)
                
                if len(v1)>0:
                    Violation=torch.vstack((Violation,v1))
                if len(v2)>0:
                    Violation=torch.vstack((Violation,v2))                
            else:
                Loss,loss1,v1,v2=Invariant_loss(y_pred,batch_X, alpha,S,W1,hype,TH,W_v,c_v,bias_test)
            loss1.backward()
            optimizer.step()
            loss=loss+loss1.item()
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss}")
        if loss<1e-5:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss}")
            status=False
            break
        loss=0
    
    if not status:
        print("Training is successful")
    else:
        print("Refinement is required")
    
    return model1,alpha,status,Violation[1:,:]














# class Model(nn.Module):
#     def __init__(self,in_features=3,h1=hidden_units,out_features=3):
#         # 2 feature-->h1 N-->h2 N-->output (1)
#         super().__init__()
#         self.fc1=nn.Linear(in_features,h1)
#         # self.fc3=nn.Linear(h2,h3)

#         self.out=nn.Linear(h1,out_features)

#     def forward(self,x):
#         x=F.relu(self.fc1(x))
#         # x=F.relu(self.fc2(x))
#         # x=F.relu(self.fc3(x))


#         x=self.out(x)
#         return x




        