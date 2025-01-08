import numpy as np
import cdd
import cdd
from mpmath import mp
import numba
from numba import njit
# from plot_res import plot_polytope,plot_hype
from matplotlib import pyplot as plt
def finding_PWA_Invariat_set(h,V,A_dyn,H,n):
    A_new=[]
    H_new=[]
    V_new=[]
    h=np.reshape(h,(len(V),n+1))
    max=0
    for i in range(len(V)):
        vertex=V[i]
        val=h[i,0:n]@vertex.T+h[i,-1]
        if np.max(val)>max:
            max=np.max(val)
            point=vertex
            # print("check")
        if np.max(val)<=1e-8:
            # plot_polytope([vertex],"r--")
            pass
        elif np.min(val)>=-1e-8:
            A_new.append(A_dyn[i])
            H_new.append(H[i])
            V_new.append(V[i])
        else:
            pos_vertices=vertex[val>=-1e-9]
            A,b =compute_polytope_halfspaces(vertex)
            a_new=-h[i,0:n]
            b_new=h[i,-1]
            A_pos=np.vstack([A,a_new])
            b_pos=np.hstack([b,b_new])
            vertices=compute_polytope_vertices_float(A_pos,b_pos)
            vertices_n=np.vstack([pos_vertices,vertices])
            if len(vertices_n)<3:
                print("check")
            V_new.append(vertices_n)
            A_new.append(A_dyn[i])
            H_new.append(H[i])
            # plot_polytope([vertices_n],"b--")
            # plot_hype(h[i,0:n],h[i,-1],3.14)

        
    # plot_polytope(V_new,"r--")
    # plot_polytope(V,"b--")

    return V_new,A_new,H_new




def compute_polytope_halfspaces(vertices):
    r"""Compute the halfspace representation (H-rep) of a polytope.

    The polytope is defined as convex hull of a set of vertices:

    .. math::

        A x \leq b
        \quad \Leftrightarrow \quad
        x \in \mathrm{conv}(\mathrm{vertices})

    Parameters
    ----------
    vertices :
        List of polytope vertices.

    Returns
    -------
    :
        Tuple ``(A, b)`` of the halfspace representation, or empty array if it
        is empty.
    """
    V=np.vectorize(mp.mpf)(vertices)
    V = np.vstack(vertices)
    t = np.ones((V.shape[0], 1))  # first column is 1 for vertices
    tV = np.hstack([t, V])
    mat = cdd.Matrix(tV, number_type="fraction")
    mat.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(mat)
    bA = np.array(P.get_inequalities())
    if bA.shape == (0,):  # bA == []
        return bA
    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])
    A=np.vectorize(float)(A)
    b=np.vectorize(float)(b)
    return A, b




def compute_polytope_vertices_float(A,b):
#     A: np.ndarray, b: np.ndarray
# ) -> list[np.ndarray]:
    r"""Compute the vertices of a polytope.

    The polytope is given in halfspace representation by :math:`A x \leq b`.

    Parameters
    ----------
    A :
        Matrix of halfspace representation.
    b :
        Vector of halfspace representation.

    Returns
    -------
    :
        List of polytope vertices.

    Notes
    -----
    This method won't work well if your halfspace representation includes
    equality constraints :math:`A x = b` written as :math:`(A x \leq b \wedge
    -A x \leq -b)`. If this is your use case, consider using directly the
    linear set ``lin_set`` of `equality-constraint generatorsin pycddlib
    <https://pycddlib.readthedocs.io/en/latest/matrix.html>`_.
    """
    mp.dps = 10
    A = np.vectorize(mp.mpf)(A)
    b = np.vectorize(mp.mpf)(b)
    b = b.reshape((b.shape[0], 1))
    matrix=np.hstack([b, -A])
    mat = cdd.Matrix(matrix[0:-1,:], number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.extend(matrix[-1:,:], linear=True)
    # Define which rows are equalities (assuming only the last row is equality)
    # linear = [False] * (A.shape[0] - 1) + [True]

# Assign the linear property to the matrix
    # mat.lin_set = linear
    P = cdd.Polyhedron(mat)
    g = P.get_generators()
    V = np.array(g)
    V=np.vectorize(float)(V)
    vertices = []
    vertices=finding_vertices(V)

    # vertices = np.vectorize(float)(vertices)
    return vertices




@njit
def finding_vertices(V):
    vertices=[]
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise ValueError("Polyhedron is not a polytope")
        else:
            vertices.append(V[i, 1:])
    return vertices

