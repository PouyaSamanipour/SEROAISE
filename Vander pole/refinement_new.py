import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull
from plot_res_BF import plot_polytope,plot_polytope_2D,plot_hype
import matplotlib.pyplot as plt
import torch
def refinement_preocess_ZDL(polytope,splitting_reg,NN_file):
    n=len(polytope[0][0])
    cosine_lst=[]
    model=torch.jit.load(NN_file)
    model.eval()
    H_new=np.zeros((0,n))
    B_new=np.zeros((0))
    for i in splitting_reg:
        vertices=polytope[int(i)]
        # stat=origin_boundary(n,vertices)
        # if np.all(stat==0):
        with torch.no_grad():
            dx=model(torch.FloatTensor(vertices).cuda())
            # dx[torch.abs(dx)<0.05]=0
        # else:
        #     with torch.no_grad():
        #         dx=model(torch.FloatTensor(vertices).cuda())
        #     index=np.where(stat!=0)
        #     for k in index:
        #         dx[k]=torch.FloatTensor(-vertices[k]).cuda()
        norm = torch.norm(dx, p=2, dim=1, keepdim=True)
        norm[norm == 0] = 1.0
        dx_norm = dx/norm
        ans=dx_norm@dx_norm.T
        upper_tri=torch.triu(ans, diagonal=1)
        value_upper_tri=(upper_tri[torch.abs(upper_tri)>0])
        indices=torch.stack(torch.where(torch.abs(upper_tri)>0))
        final=torch.vstack((value_upper_tri,torch.FloatTensor([i]*len(value_upper_tri)).cuda(),indices)).T
        cosine_lst.extend(final.cpu().detach().numpy())
        candidate=final[torch.argsort(final[:,0])][0:n,:]
        points=np.zeros((0,n))
        candidate_n=candidate.cpu().detach().numpy()
        for j in range(n):
            dx1=dx[int(candidate[j,-2])]
            dx2=dx[int(candidate[j,-1])]
            if torch.norm(dx1)<=1e-7 or torch.norm(dx2)<=1e-7:
                alpha=torch.FloatTensor([0.5]).cuda()
            else:
                alpha=1/(1+torch.norm(dx1)/torch.norm(dx2))
            beta=1-alpha
            alpha=alpha.cpu().detach().numpy()
            beta=beta.cpu().detach().numpy()
            points=np.vstack((points,alpha*vertices[int(candidate_n[j,-2])]+beta*vertices[int(candidate_n[j,-1])]))
        # for j in range(len(vertices)):
        #     if j not in [int(candidate[-2]),int(candidate[-1])]:
        #         points=np.vstack((points,vertices[j]))
        #         if len(points)==n:
        #             break
        A=np.ones((n,n+1))
        A[:n,0]=np.array(points)[:,0]
        B=-np.array(points)[:,1]
        A_new=A[:n,:n]
        # A=np.array([[points[0][1:],1],[points[0][1:],1]])
        # B=np.array([-points[0][0],-points[0][0]])
        sol=np.linalg.solve(A_new,B)
        h_new=np.array([sol[0],1])
        b_new=sol[1]
        # plot_polytope(polytope,'b-')
        # plot_hype(h_new,b_new,3.14)
        # plot_polytope([polytope[i]],'g*')
        H_new=np.vstack((H_new,h_new))
        B_new=np.hstack((B_new,np.array(b_new)))

        
        
        # print("check")












    # # plot_polytope(polytope,'b-')
    # # for i in cosine_lst:
    # #     if i[0]<GA:
    # #         # plot_polytope([polytope[int(i[1])]],'r-')
    # # plt.show()
    # cosine_new=np.array(cosine_lst)
    # cosine_new=cosine_new[cosine_new[:,0].argsort()]
    # # list1=cosine_new[cosine_new[:,0]<GA]
    # regions=[]
    # for i in list1:
    #     vertices=polytope[int(i[1])]
    #     # mid_point=np.sum(vertices,axis=0)/len(vertices)
    #     # if np.max(np.abs(vertices))>0.10001:
    #     regions=[int(i[1])]
    #     if len(regions)!=0:
    #         break
    # # for i in list1:
    # #     if i[1] not in regions:
    # #         regions.append(int(i[1]))
    # H_new=[]
    # B_new=[]
    # if len(regions)!=0:
    #     # plot_polytope([polytope[regions[0]]],'g*')

    #     for i in regions:
    #         info_reg=cosine_new[cosine_new[:,1]==i]
    #         candidates=info_reg[info_reg[:,0].argsort()][0:n]
    #         vertices=polytope[i]
    #         info=opt_sample[opt_sample[:,-1]==i]
    #         # plot_polytope([polytope[i]],'g*')
    #         # if len(candidates)%n==0:
    #         #     pass
    #         # else:
    #         #     candidates=info_reg[0:np.shape(candidates)[0]+1]
    #         points=[]
    #         for j in range(int(len(candidates))):
    #             alpha=1/(1+np.linalg.norm(info[int(info_reg[j,2]),n:2*n])/np.linalg.norm(info[int(info_reg[j,3]),n:2*n]))
    #             beta=1-alpha
    #             point=alpha*info[int(info_reg[j,2]),0:n]+beta*info[int(info_reg[j,3]),0:n]
    #             points.append(alpha*info[int(info_reg[j,2]),0:n]+beta*info[int(info_reg[j,3]),0:n])
    #             model=torch.jit.load(NN_file)
    #             model.eval()
    #             dx=(model(torch.FloatTensor(point).cuda())).cpu().detach().numpy()
    #             # cos1=np.dot(dx,info[int(info_reg[j,2]),n:2*n])/(np.linalg.norm(dx)*np.linalg.norm(info[int(info_reg[j,2]),n:2*n]))
    #             # cos2=np.dot(dx,info[int(info_reg[j,3]),n:2*n])/(np.linalg.norm(dx)*np.linalg.norm(info[int(info_reg[j,3]),n:2*n]))




    #             if len(points)==n:
    #                 # A = np.hstack((points, np.ones((np.array(points).shape[0], 1))))

    #                 # # Create a vector of zeros (right-hand side of the equation system)
    #                 # b = np.zeros(np.array(points).shape[0])

    #                 # # Solve the system using least squares
    #                 # solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    #                 # # Extract the coefficients H and b
    #                 # H_p = solution[:-1]
    #                 # b_p = solution[-1]
    #                 A=np.ones((n,n+1))
    #                 A[:n,0]=np.array(points)[:,0]
    #                 B=-np.array(points)[:,1]
    #                 A_new=A[:n,:n]
    #                 # A=np.array([[points[0][1:],1],[points[0][1:],1]])
    #                 # B=np.array([-points[0][0],-points[0][0]])
    #                 sol=np.linalg.solve(A_new,B)
    #                 h_new=np.array([sol[0],1])
    #                 b_new=sol[1]
    #                 # plot_polytope(polytope,'b-')
    #                 # plot_hype(h_new,b_new,3.14)
    #                 # plt.show()

    #                 H_new.append(h_new)
    #                 B_new.append(b_new)
    #                 points=[]
    return [H_new],[B_new]    
def origin_boundary(n,vertices):
    H_b=np.vstack((np.eye(n),-np.eye(n)))
    b=0.1*np.ones((2*n,1))
    val=H_b@vertices.T+b
    val=np.maximum(val,0)
    return np.min(val,axis=0)
