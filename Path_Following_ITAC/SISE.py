import cProfile
import pstats
import numpy as np
import random
from Updating_Zero_LevelSets import updating_BF_LV
from Enum_module_Lyap import Finding_Lyapunov_function 
from Enum_module_BF import Finding_Barrier, Finding_Lyap_Invariant
from preprocessing_LF import preprocessing_Lyap
from utils_n_old import checking_sloution
import time
from plot_res_Lyap import plotting_results_lyap,plot_invariant_set,plot_level_set,plot_polytope_2D,plot_polytope
import matplotlib.pyplot as plt
from invarint_PWA import finding_PWA_Invariat_set
from Finding_Lyapunov import finding_Lyapunov
import Lyap_PostProcess
from Lyap_plot_sol import plot2d
mode="Rapid_mode" 
parallel=False
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        NN_file="NN_files/model_2d_IP_8.pt"
        # NN_file="NN_files/Inverted_Penduluem20.xlsx"
        # NN_file="NN_files/model_2d_simple_5.pt"
        # NN_file="NN_files/Path_following_20.xlsx"
        # NN_file="NN_files/model_2d_Pedram3.pt"
        # eps1=0.01
        # eps2=0.01
        # name="IP_Lyap"
        # TH=3.14
        # V=Finding_Lyapunov_function(NN_file,name,eps1,eps2,TH,mode,parallel)
        # plot_level_set(V,TH,'green',[21])
        eps1=0.01
        eps2=0.01
        name="IP_BF"
        def random_color():
            return (random.random(), random.random(), random.random())
        TH=3.14
        # alpha=[0.0025,.003,0.004,0.005,0.01,0.05,0.08,0.2,0.3,0.5]
        # alpha=[0.0009,0.001,0.0028,0.0025,0.003,0.004,0.005,0.01,0.05,0.08,0.2,0.3]
        # alpha=[0.0025,0.003,0.0035,0.0038,0.004,0.005,0.01,0.05,0.08,0.2,0.3]
        alpha=[0.022]
        X=[]
        Y=[]
        Z=[]
        for alph in alpha:
            NN,h,all_hyperplanes,all_bias,W,c,enumerate_poly,D,border_hype,border_bias=Finding_Barrier(NN_file,name,eps1,eps2,TH,mode,parallel,alph)
            x,y,z=plot_invariant_set(h,TH,alph,color=[random_color()])
            X.append(x)
            Y.append(y)
            Z.append(z)
        # plt.legend([f"$\alpha={a}" for a in alpha])
        fig, ax = plt.subplots()
        lines=[]
        labels=[]
        ax.contour(X[0],Y[0],Z[0],levels=[0],colors='red',linestyles='dashed')
        # ax.contour(X[1],Y[1],Z[1],levels=[0],colors='green',linestyles='dashed')
        # ax.contour(X[2],Y[2],Z[2],levels=[0],colors='black',linestyles='dashed')
            # ax.clabel(CS1, inline=1, fontsize=10)
            # lines.append(CS1.collections[0])
            # labels.append(f"alpha={alpha[i]}")
            # contour.collections[0].set_label(f'a = {alph}')  # Set the label on the contour line
        LV1=np.array([[-1.8754645710519053, 3.1320293398533003],
[-1.130117293964199, 2.765281173594132],
[-0.637117999878452, 2.4425427872860634],
[-0.07524297942563773, 2.163814180929095],
[0.3718812381081493, 1.8557457212713935],
[1.1056440415694668, 1.3716381418092913],
[1.747881088510428, 1.1515892420537899],
[2.3899638628001885, 0.7701711491442547],
[3.0319554759778775, 0.29339853300733454],
[3.1344977022724825, -0.4474327628361845],
[3.1221418653901, -1.3716381418092896],
[3.0867994034790813, -2.3398533007334947],
[2.983184281019321, -2.72127139364303],
[2.615678776290631, -3.1320293398532986],
[1.5262193383105735, -2.7066014669926632],
[0.8268312864936629, -2.266503667481661],
[0.16213354401679236, -1.5403422982885084],
[-0.4110244171532491, -1.0635696821515888],
[-1.431509955260931, -0.4914425427872855],
[-2.1999560556690523, -0.28606356968215074],
[-3.1172612396976254, 0.2127139364303181],
[-3.1397569971997172, 0.682151589242054],
[-3.070516626384363, 1.1075794621026898],
[-3.1390206959099043, 1.4523227383863109],
[-3.0122716881635476, 2.0317848410757944],
[-2.90846021869317, 2.6185819070904643],
[-3.0573613766730396, 2.8679706601466988],
[-2.4261478118995634, 3.1173594132029336],
[-1.886922821600041, 3.146699266503667]])
        LV2=np.array([[1.9847036328871894, -3.1320293398532986],
[1.1131753519052667, -2.7506112469437642],
[0.6432328067805155, -2.3105134474327613],
[0.012762555690088995, -1.7823960880195582],
[-0.9390295782746709, -1.3569682151589229],
[-2.005446292080203, -0.8288508557457206],
[-2.83107144693722, -0.4327628361858178],
[-3.1176539337188585, -0.19804400977995051],
[-3.1176539337188585, -0.19804400977995051],
[-3.197440943961628, 0.3447432762836189],
[-3.208310153477913, 0.975550122249389],
[-3.1617478623887947, 1.6797066014669926],
[-3.081063265811778, 2.0757946210268945],
[-3.138074022823002, 2.4425427872860634],
[-3.1375691305099878, 2.970660146699266],
[-3.1375691305099878, 2.970660146699266],
[-2.7588718461761417, 3.0880195599022002],
[-2.426133787113091, 3.1320293398533003],
[-1.829561444927001, 3.146699266503667],
[-1.48539318488876, 3.146699266503667],
[-1.1300050956724181, 2.882640586797066],
[-0.637117999878452, 2.4425427872860634],
[-0.07521492985269251, 2.1931540342298286],
[0.6242011715371625, 1.78239608801956],
[1.3924228753617225, 1.3422982885085575],
[2.069160897025343, 1.2102689486552567],
[2.619367295132931, 0.7261613691931545],
[2.9517126601747483, 0.3594132029339856],
[3.1236705671156155, 0.22738386308068526],
[3.1346800244966264, -0.25672371638141733],
[3.1447918955433902, -1.6797066014669921],
[3.132969000546966, -2.0464547677261598],
[3.0865469573225743, -2.6039119804400963],
[2.994362035838005, -3.0293398533007325],
[2.7762906309751436, -3.1320293398532986],
[2.2715806401847534, -3.058679706601466],
[1.99618993300827, -3.117359413202932]])
        plt.plot(LV1[:,0],LV1[:,1],color='green',linestyle='--')
        plt.plot(LV2[:,0],LV2[:,1],color='black',linestyle='--')
        plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='--'),plt.Rectangle((0,0),1,2,color='g',fill=False,linewidth = 2),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2,linestyle='-')]\
           ,[r'$\alpha=0.0023$',r'$\alpha=0.0025$',r'$\alpha=0.0027$'],loc='upper right',fontsize=14)
        # plt.legend(lines, labels)
        plot_polytope_2D(NN_file,TH)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()
