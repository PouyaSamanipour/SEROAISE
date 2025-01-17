import numpy as np
import Lyap_optimization_process
import Lyap_PostProcess
import Lyap_Splitting_test_ZDL
from Lyap_preprocessing_01 import preprocess
def finding_Lyapunov(V,A_dyn,n,H,epsilon1,epsilon2,Threshold):
    Threshold=1e-7
    status=True
    while(status):
        # finding number of optimization variables
        # finding the identifier vector
        [asub,aval,qsubi,qsubj,qval,n_list,id_var,A_PD,n_r,cell_info,neighbor_info]=preprocess(V,A_dyn,n,H)    
        # num_reg.append(n_r)
        c=np.zeros((1,int(n_list[0])))
        for i in range (n_r):
            c[0][-1-i]=1
        # st1=time.time()
        sol=Lyap_optimization_process.qp_optimization(asub,aval,qsubi,qsubj,qval,n_list,c,n_r,epsilon1,epsilon2)
        # st2=time.time()
        # opt_time.append(st2-st1)
        status=Lyap_PostProcess.check_status(sol,n_r,Threshold)
        print("number of region:\n",n_r)
        # if status==False:
        #     # I need to add a part for saving all the output data
        #     SYSDATA=final_Info.final_sysData(V,A_dyn,H,sol,"IP",epsilon1,epsilon2,Th,n_r,id_var)
        #     et=time.time()
        #     break
        Th=1e-7
        V,A_dyn,H=Lyap_Splitting_test_ZDL.splitting_cell_prep(sol,V,n_r,id,H,A_dyn,cell_info,id_var,neighbor_info,Th,iter)
    return V,A_dyn,H,sol,A_PD,id_var