import cProfile
# from scipy.sparse import csr_matrix
# from preprocessing_BF import preprocessing_BF
import numpy as np
# from numba.typed import List
import random
from Enum_module_BF import Finding_Barrier, Finding_Lyap_Invariant,updating_NN,updating_NN_Original
# from preprocessing_LF import preprocessing_Lyap
# from utils_n_old import checking_sloution
import time
from plot_res_Lyap import plot_invariant_set_multiple,plot_polytope_2D,plot_polytope,plot_invariant_set_single,plot_heatmap
import matplotlib.pyplot as plt
import time
import torch
from SISE_Algorithm import SISE_algorithm
# from Optimization_SISE import SIS_opt
mode="Rapid_mode" 
parallel=False
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        NN_file="NN_files/model_2d_IP_8.pt"
        # NN_file="NN_files/model_IP_Pedram_n copy 2.pt"
        # NN_file="NN_files/Inverted_Penduluem20.xlsx"
        # NN_file="NN_files/model_2d_simple_3.pt"
        # NN_file="NN_files/Path_following_20.xlsx"
        # NN_file="NN_files/model_2d_Pedram3.pt"
        # eps1=0.01
        # eps2=0.01
        # name="IP_Lyap"
        # TH=3.14
        # V=Finding_Lyapunov_function(NN_file,name,eps1,eps2,TH,mode,parallel)
        eps1=1e-02
        eps2=1e-04
        name="IP_BF"
        def random_color():
            return (random.random(), random.random(), random.random())
        bound=[(-3.14,3.14),(-3.14,3.14)]
        TH=np.array([3.14,3.14])
        
        # plot_polytope_2D(NN_file,TH)
        alpha=[0.04]
        X=[]
        Y=[]
        Z=[]
        fig, ax = plt.subplots()
        time_start=time.time()
        for alph in alpha:
            NN,h,all_hyperplanes,all_bias,W_x,c_x,enumerate_poly ,D ,border_hype,border_bias,zeros,W_h,c_h=Finding_Barrier(NN_file,name,eps1,eps2,TH,mode,parallel,alph,bound)
            zeros1=[]
            x,y,z=plot_invariant_set_multiple(h,zeros1,TH,alph,color=[random_color()])
            X.append(x)
            Y.append(y)
            Z.append(z)


        lines=[]
        labels=[]
        Z_new=np.max(np.array(Z),axis=0)
        # ax.pcolormesh(X[0], Y[0], Z_new, cmap='seismic', shading='auto')  # 'RdBu' colormap for red-blue

        ax.contour(X[0],Y[0],Z_new,levels=[0],colors='red',linestyles='solid')
        plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='-'),plt.Rectangle((0,0),1,2,color='black',fill=False,linewidth = 2),plt.Rectangle((0,0),1,2,color='green',fill=False,linewidth = 2,linestyle='-'),plt.Rectangle((0,0),1,2,color='yellow',fill=False,linewidth = 2,linestyle='-')]\
           ,["Invariant set[23]","Iteration 1","Iteration 2","Iteration 3"],loc='upper right',fontsize=14)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        alpha_p=alpha
        colors=['black','green','yellow','brown']
        iteration=3
        for iter in range(iteration):
            alpha[0]=alpha[0]-0.1*alpha[0]
            NN,h,all_hyperplanes,all_bias,W_x,c_x,enumerate_poly ,D ,border_hype,border_bias,W_h,c_h=SISE_algorithm(NN,h,enumerate_poly,D,all_hyperplanes,all_bias,W_h,c_h,W_x,c_x,TH,border_hype,border_bias,zeros,eps1,eps2,alpha,parallel,NN_file,iter,bound)
            plot_polytope_2D(NN_file,TH)
            if iter<iteration-1:
                zeros1=[]
            else:
                zeros1=zeros
            # zeros=[]
            plot_invariant_set_single(h,TH,zeros1,colors[iter])
        time_end=time.time()
        print("Time for the whole process=",time_end-time_start)
        # x,y,z=plot_invariant_set_multiple(BF_NN,zeros,TH,alph,color=[random_color()])
        # X.append(x)
        # Y.append(y)
        # Z.append(z)
        # ax.contour(X[1],Y[1],Z[1],levels=[0],colors='red',linestyles='solid')
        plot_heatmap(h,TH,zeros)
        plt.show()

        # x,y,z=plot_invariant_set(h_n,[],TH,alph,color=[random_color()])
        # X.append(x)
        # Y.append(y)
        # Z.append(z)
        # CS=ax.contour(X[0],Y[0],Z[alpha.index(alph)],levels=[0],colors=[random_color()],linestyles=':')
        # plt.clabel(CS, inline=True, fmt={0: fr'$\alpha={alph:.3f}$'}, fontsize=8) 
        # ax.contour(X[0],Y[0],Z[1],levels=[0],colors='red',linestyles='solid')
        # plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='solid')]\
        #    ,[fr'UIS Invariant Set'],loc='upper right',fontsize=14)
        # plt.plot(points[:,0].detach().cpu().numpy(),points[:,1].detach().cpu().numpy(),'ro')
        # plot_polytope(enumerate_poly, name)
        # plt.show()

        # h_sol=np.hstack((all_hype,all_b.reshape((len(all_b),1))))
        # h_sol=h_sol.reshape(-1)
        # Refined_polytope,A_new,B_new=finding_PWA_Invariat_set(h_sol,Refined_polytope,A_dyn,B_dyn,2)
        # # plot_invariant_set(h,TH,'cyan')
        # # plot_polytope(Refined_polytope, name)
        # eps1=0.01
        # eps2=0.01
        # name="IP_BF_Lyap"
        # TH=3.14
        # n=2
        # # V_lyp1,A_lyap1,H_lyap1,sol_Lyap1,A_PD2,id_var2=finding_Lyapunov(Refined_polytope,A_new,n,B_new,eps1,eps2,Threshold=0.099)

        # V_final,_,_,_,_,_,_,_,_=Finding_Lyap_Invariant(NN_file,h,Refined_polytope,all_hyperplanes,all_bias,border_hype,border_bias,new_hype,new_bias,W,c,eps1,eps2,TH,parallel)

        # # plot_level_set(V,TH,'green',[20])
        # # min_val,max_val,ls,sol_n2,list_points,levset_pts=Lyap_PostProcess.sol_Process(sol_Lyap1,A_PD2,id_var2,n,V_lyp1,len(V_lyp1))
        # # plot2d(A_lyap1,sol_n2,len(V_lyp1),H_lyap1,1,list_points,levset_pts,V_lyp1,'red',"level set after finding the safe set")
        # plot_level_set(V_final,TH,'red',[1,2,2.5,3])
        # plot_invariant_set(h,TH,'cyan')
        # plt.show()





