import numpy as np
import torch
from Enum_module_BF import updating_NN
from utils_n_old import finding_side
from numba import njit
from plot_res_Lyap import plot_polytope,plot_invariant_set_single,plot_hype
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from utils_n_old import Intersection_polytope_hype
from Optimization_SISE_test import finding_similar_vertex_with_index


def updating_BF_LV(NN,h,enumerate_poly,D,all_hyperplanes,all_bias,W_v,c_v,border_hype,border_b):
    with torch.no_grad():
        # enumerate_poly_new=[]
        model=torch.jit.load(h)
        model.eval()
        # model_original=torch.jit.load(NN)
        # model_original.eval()
        # hyperplanes=[]
        # biases=[]
        # hyperplanes=model.fc1.weight
        # all_hyperplanes=model.fc1.weight
        # all_bias=model.fc1.bias
        # biases=model.fc1.bias
        # W=model.out.weight
        # weights=model_original.out.weight
        # boundary_hyperplanes=[((model_original.fc1.weight).detach().cpu().numpy()).astype(np.float32)]
        # border_bias=[((model_original.fc1.bias).detach().cpu().numpy()).astype(np.float32)]
        # c=model.out.bias
        new_hype=np.zeros((0,2)) 
        new_bias=np.zeros((0))
        # A_dyn=[]
        # B_dyn=[]
        points_all=np.zeros((0,np.shape(enumerate_poly[0])[1]))
        region_all=[]
        all_hyperplanes_n=np.copy(all_hyperplanes)
        all_bias_n=np.copy(all_bias)
        with torch.no_grad():
            for j,i in enumerate(enumerate_poly):
                vertices=i
                vertices=torch.FloatTensor(vertices).cuda()
                h_val1=W_v@np.diag(D[j])@(all_hyperplanes@i.T+all_bias)+c_v
                h_val=model(vertices).cpu().numpy()
                if np.max(h_val1)>1e-6 and np.min(h_val1)<=-1e-6:
                    if not (np.max(h_val)>1e-6 and np.min(h_val)<=-1e-6):
                        print("check")


                    # enum=enumerate_poly[j].astype(np.float32)
                    # sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                    # enumerate_poly_new.append(i)
                    hype=W_v@np.diag(D[j])@(all_hyperplanes)
                    bias=(W_v@np.diag(D[j])@all_bias+c_v)
                    points_intersect=Intersection_polytope_hype([border_hype],hype,bias[0,0],[border_b],h_val1[0],enumerate_poly[j],3.14,parallel=False)
                    for point in points_intersect:
                        st,id=finding_similar_vertex_with_index(np.array(point),points_all)
                        if st:
                            region_all[id].append(j)
                        else:
                            points_all=np.vstack((points_all,point))
                            region_all.append([j])
                        
                    # points_all.extend(points_intersect)
                    bias=bias.reshape(-1,1)
                    all_hyperplanes_n=np.vstack((all_hyperplanes_n,hype,-hype))
                    # bias.extend(bias[0])
                    all_bias_n=np.vstack((all_bias_n,bias,-bias))
                    new_hype=np.vstack((new_hype,hype))
                    new_bias=np.hstack((new_bias,bias[0]))
                    # plot_polytope([enumerate_poly[j]], "test")
                    # plot_hype(hype[0],bias[0],3.14)
                else:
                    pass
                    # plot_polytope([enumerate_poly[j]], "test")
                    # all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                    # all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                    # A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                    # B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                # elif torch.min(h_val)>-1e-10:
                #     enum=enumerate_poly[j].astype(np.float32)
                #     sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                #     enumerate_poly_new.append(i)
                #     A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                #     B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                #     all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                #     all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())

                    # all_hyperplanes=torch.cat([all_hyperplanes,new_hype],0)
                    # all_bias=torch.cat([all_bias,new_bias],0)
                    # weights=torch.cat([weights,torch.FloatTensor(np.zeros((2,len(new_hype)))).cuda()],1)


        # NN,_,_,_,_=updating_NN(NN,all_hyperplanes.size()[1],all_hyperplanes,all_bias,weights,c)
    return new_hype,new_bias,all_hyperplanes_n,all_bias_n,points_all,region_all


    
            



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




def finding_boundary_vertice(enumerate_poly,h_n,Original_NN,W_h,c_h,hyperplanes,biases,D,TH,eps):
    index_list,V=cells_ordering(enumerate_poly)
    # with torch.no_grad():
    #     model=torch.jit.load(h_n)
    #     model.eval()
    #     h_val=model(torch.FloatTensor(V).cuda()).cpu().numpy()
    #     if torch.max(model(torch.FloatTensor(V).cuda()))<5:
    #         coeff=(5.0/torch.max(model(torch.FloatTensor(V).cuda()))).cpu().numpy()
    #     elif torch.min(model(torch.FloatTensor(V).cuda()))>-5:
    #         coeff=(-5.0/torch.min(model(torch.FloatTensor(V).cuda()))).cpu().numpy()
    #     else:
    #         coeff=1.0
        # for i in V:
        #     plt.plot(i[0],i[1],'ko')
    n=np.shape(enumerate_poly[0])[1]
        # W_v=(model.out.weight).cpu().numpy()
        # c=model.out.bias.cpu().numpy()
        # hyperplanes=model.fc1.weight.cpu().numpy()
        # biases=np.reshape(model.fc1.bias.cpu().numpy(),(len(model.fc1.bias.cpu().numpy()),1))
    points=np.zeros((0,n))
        # region=[]
        # h_val=model(torch.FloatTensor(V).cuda()).cpu().numpy()
        # D_new=np.copy(D)
        # D_new=D_new.astype(np.float32)
    points,region,b_reg,label=finding_all_info(index_list,V,points,hyperplanes,biases,W_h,D,c_h,TH)
        
        # for i in points:
        #     plt.plot(i[0],i[1],'ro')
    AGIS_Points,AGIS_Region,NAGIS_Points,NAGIS_Region=polishing_regions(points,region,h_n,Original_NN,D,enumerate_poly,eps,TH)

    return AGIS_Points,AGIS_Region,NAGIS_Points,NAGIS_Region,index_list,V,b_reg,label

def updating_BF_LV_n(NN,h,enumerate_poly,D,new_hype):
    with torch.no_grad():
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
        # new_hype=np.zeros((0,2)) 
        new_bias=np.zeros((0))
        A_dyn=[]
        B_dyn=[]
        all_hyperplanes=np.zeros((0,2)) 
        all_bias=np.zeros((0))
        with torch.no_grad():
            for j,i in enumerate(enumerate_poly):
                vertices=i
                vertices=torch.FloatTensor(vertices).cuda()
                h_val=model(vertices)
                if torch.max(h_val)>=1e-5 and torch.min(h_val)<=-1e-5:
                    print("check")
                    plot_polytope([enumerate_poly[j]], "test")
                    # enum=enumerate_poly[j].astype(np.float32)
                    # sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                    # enumerate_poly_new.append(i)
                    hype=(W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy()
                    bias=(W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy()
                    # all_hyperplanes=np.vstack((all_hyperplanes,hype,-hype))
                    # # bias.extend(bias[0])
                    # all_bias=np.hstack((all_bias,bias,-bias))
                    # new_hype=np.vstack((new_hype,hype))
                    # new_bias=np.hstack((new_bias,bias))
                    # all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                    # all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                    # A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                    # B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                # elif torch.min(h_val)>-1e-10:
                #     enum=enumerate_poly[j].astype(np.float32)
                #     sides,hyp_f=finding_side_new(boundary_hyperplanes[0],enum,border_bias[0])
                #     enumerate_poly_new.append(i)
                #     A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                #     B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                #     all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                #     all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())

                    # all_hyperplanes=torch.cat([all_hyperplanes,new_hype],0)
                    # all_bias=torch.cat([all_bias,new_bias],0)
                    # weights=torch.cat([weights,torch.FloatTensor(np.zeros((2,len(new_hype)))).cuda()],1)


        # NN,_,_,_,_=updating_NN(NN,all_hyperplanes.size()[1],all_hyperplanes,all_bias,weights,c)
    return new_hype,new_bias,all_hyperplanes,all_bias



@njit
def cells_ordering(enumerate_poly):
    n=len(enumerate_poly[0][0])  # Get the number of elements in the first list of enumerate_poly
    V = np.zeros((0,n), dtype=np.float64)  # Pre-allocate an empty array for V
    index_list = []
    # vert=np.reshape(j,(len(j),1))
    for k, i in enumerate(enumerate_poly):
        index_list.extend([k]*len(i)) # Create an array of repeated indices
        # Extend V with elements from j
        V = np.vstack((V, i))
    return index_list,V

@njit
def finding_all_info(index_list,V,points,hyperplanes,biases,W_h,D,c_h,TH):
    region=[]
    b_reg=[]
    # boundary_points=[]
    # rest=[]
    # interior_points=[]
    # uc_points=[]
    # int_reg=[]
    # uc_reg=[]
    label=[]
    for j,i in enumerate(V):
        stat=False
        vertex=np.reshape(np.copy(i),(1,len(i)))
        h=(W_h@np.diag(D[index_list[j]])@(hyperplanes@vertex.T+biases)+c_h)
        h_new=W_h@np.maximum(hyperplanes@vertex.T+biases,0)+c_h
        # if h-h_new>1e-5:
        #     print("check")
        if np.any(TH-i>=-1e-6) or np.any(i+TH>=-1e-6):
            # boundary_points.append(i)
            b_reg.append(index_list[j])
            label.append("bd")
        elif np.min(np.abs(h))<=1e-5:
            label.append("N-AGIS")
            # plt.plot(i[0],i[1],'go')
            if len(points)==0:
                points=np.vstack((points,vertex))
                region.append([index_list[j]])
            else:
                val_new=(W_h@np.diag(D[index_list[j]])@(hyperplanes@points.T+biases)+c_h)
                if np.min(np.abs(val_new))>=1e-5:
                    points=np.vstack((points,vertex))
                    region.append([index_list[j]])
                else:
                    id_new=np.where(np.abs(val_new[0])<=1e-5)
                    for f in id_new[0]:
                        if np.all(np.abs(points[f]-i)<=1e-12):
                            stat=True
                            region[f].append(index_list[j])
                            break
                    if not stat:
                        points=np.vstack((points,vertex))
                        region.append([index_list[j]])    
        elif h>1e-5:
            label.append("int")
            # interior_points.append(i)
            # int_reg.append(index_list[j])
        else:
            label.append("uc")
            # uc_points.append(i)
            # uc_reg.append(index_list[j])

            # rest.append(i)
        #     plt.plot(i[0],i[1],'yo')
    return points,region,np.unique(np.array(b_reg)),label

def polishing_regions(points,region,W_h,c_h,X,D,enumerate_poly,eps,TH,hype):
    #In this function, we need to keep cells with h(x)<=0 and remove the rest
    AGIS_region=[]
    Hdot=[]
    AGIS_points=[]
    NAGIS_region=[]
    NAGIS_points=[]
    with torch.no_grad():
        # model=torch.jit.load(h)
        Xdot=torch.jit.load(X)
        # W=Xdot.out.weight
        # c=Xdot.out.bias
        # hype=Xdot.fc1.weight
        # bias=X.fc1.bias
        # W_h=model.out.weight
        # c_h=model.out.bias
        # model.eval()
        for k,i in enumerate(region):
            if np.all(TH-points[k])>=0.05 and np.all(points[k]+TH)>=0.05:
            # if np.max(np.abs(points[k]))<=TH-0.05:
                xdot=Xdot(torch.FloatTensor(points[k]).cuda())
                for j in i:
                    dh=(W_h@np.diag(D[j]))@hype@(xdot.cpu().numpy())
                    Hdot.append(dh)
                if np.min(Hdot)>=3*eps:
                    AGIS_region.append(region[k])
                    AGIS_points.append(points[k])
                    # plt.plot(points[k][0],points[k][1],'go')
                    # DH.append(Hdot)
                else:
                    NAGIS_region.append(region[k])
                    NAGIS_points.append(points[k])
            else:
                NAGIS_region.append(region[k])
                NAGIS_points.append(points[k])

                        
                Hdot=[]
    return AGIS_points,AGIS_region,NAGIS_points,NAGIS_region
        




