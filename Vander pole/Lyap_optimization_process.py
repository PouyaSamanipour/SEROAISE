import sys, os, mosek
# Since the actual value of Infinity is ignored, we define it solely
# for symbolic purposes:
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def qp_optimization(asub,aval,qsubi,qsubj,qval,n,c,n_r,epsilon1,epsilon2):
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    print("Start Optimization ...")
    with mosek.Env() as env:
        nvar=n[0]
        if len(n)>3:
            npd=n[1]
            nnd=n[2]
            ne=n[3]
        else:
            npd=n[1]
            nnd=n[2]
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)
        # Create a task
        with env.Task() as task:
            #task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients

            bkc_PD = [mosek.boundkey.lo]*(int(npd))
            bkc_ND=[mosek.boundkey.up]*(int(nnd))
            bkc=bkc_PD+bkc_ND
            list1=[epsilon1]*(int(npd))
            list2=[-inf]*(int(nnd))
            #
            blc = list1+list2
            buc1 = [inf]*(int(npd))
            buc2=[-epsilon2]*(int(nnd))
            buc=buc1+buc2
            if 'ne' in locals():
                t=1e-05*epsilon1
                bkc_eq=[mosek.boundkey.ra]*(int(ne))                
                bkc_eq=[mosek.boundkey.ra]*(int(ne))
                bkc=bkc_PD+bkc_ND+bkc_eq
                list3=[-1e-8]*(int(ne))
                blc = list1+list2+list3
                buc3=[1e-8]*(int(ne))
                buc=buc1+buc2+buc3
            numvar = nvar
            bkx1 = [mosek.boundkey.fr] * (numvar-n_r)
            blx1 = [-inf] * (numvar-n_r)
            bux1 = [inf] * (numvar-n_r)
            bkx2 = [mosek.boundkey.lo] * (n_r)
            blx2 = [1e-12] * n_r
            bux2 = [inf] * n_r
            bkx=bkx1+bkx2
            blx=blx1+blx2
            bux=bux1+bux2
            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[0][j])
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Set up and input quadratic objective
            #qsubi = [0, 1, 2, 2]
            #qsubj = [0, 1, 0, 2]
            #qval = [2.0, 0.2, -1.0, 2.0]

            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            #prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            # Output a solution
            xx = [0.] * numvar
            task.getxx(mosek.soltype.itr,
                       xx)

            if solsta == mosek.solsta.optimal:
                print("Primal and dual feasible")
            elif solsta==mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                raise Exception("Infeasible Solution\n")
    return xx