import numpy as np
from scipy import linalg
import math
import copy
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI

#This code calculates the example of thin film evolution (using the coarser solver presented in the manuscript) for three iterations using the classic Parareal algorithm.
#The code for LU decomposition is taken from the book "Numerical Recipes in C" by Press et al.

#The evolution of thin liquid film is solved using the approximate Newton-ADI method. For more details on the approximate Newton-ADI method, please see the article:
#Witelski, T.P., & Bowen, M. (2003). ADI schemes for higher-order nonlinear diffusion equations. Applied Numerical Mathematics, 45, 331-351.


class u_stencils:
    def __init__(self, uxp, uxpp, uxm, uxmm, uym, uyp, uymm, uypp, uxpyp, uxpym, uxmyp, uxmym):
        self.uxp = uxp
        self.uxpp = uxpp
        self.uxm = uxm
        self.uxmm = uxmm
        self.uym = uym
        self.uyp = uyp
        self.uymm = uymm
        self.uypp = uypp
        self.uxpyp = uxpyp
        self.uxpym = uxpym
        self.uxmyp = uxmyp
        self.uxmym = uxmym

class A_stencils:
    def __init__(self, Axp, Axm, Ayp, Aym):
        self.Axp = Axp
        self.Axm = Axm
        self.Ayp = Ayp
        self.Aym = Aym


def Ax(x, y):
    if (max(abs(x-.5*Lx), abs(y-.5*Ly))<=.25*Lx):
        z = A1
    else:
        z = A2

    return z


def Pi(h, x, y): #/* disjoining pressure */
    eh = eps/h
    return (eh*eh*eh/eps*(1.0-eh))*Ax(x,y)


def Q(h, x, y):
    return (-eps**2/2*1.0/h**2+eps**3/3*1.0/h**3)*Ax(x,y)


def swap(a, b):
    a1 = b
    b1 = a
    return a1, b1


def band2lu(ar, n, m1, m2):
    # This function calculates the LU decomposition of a band matrix
    # This is adapted from the code in book "Numerical Recipes in C" by Press et al.

    mm=m1+m2+1
    l=m1

    num_pts = ar.shape[0]
    indx = np.zeros(num_pts)

    a = copy.deepcopy(ar)
    al = np.zeros_like(ar)

    for i in range(1, m1+1):
        for j in range(m1+2-i, mm + 1):
            a[i-1, j-l-1] = a[i-1, j-1]

        l = l - 1

        for j in range(mm-l, mm+1):
            a[i-1, j-1] = 0.0

    l = m1

    for k in range(1, n+1):
        temp=a[k-1, 0]
        i = k
        if l < n:
            l = l + 1

        for j in range(k+1, l+1):
            if abs(a[j-1, 0])>abs(temp):
                temp = a[j-1, 0]
                i = j

        indx[k-1]=i

        if temp == 0.0:
            a[k-1, 0] = TINY

        if i != k:
            for j in range(1, mm+1):
                z1, z2 = swap(a[k-1, j-1], a[i-1,j-1])
                a[k-1, j-1] = z1
                a[i-1, j-1] = z2

        for i in range(k+1, l+1):
            temp=a[i-1,0]/a[k-1,0]
            al[k-1, i-k-1]=temp
            for j in range(2, mm+1):
                a[i-1, j-2]=a[i-1, j-1]-temp*a[k-1, j-1]

            a[i-1, mm-1]=0.0

    return a, al, indx

def bandsolve(a, n, m1, m2, al, indx, br):
    # This function solves a band matrix equation using the band LU decomposition above.
    # This is adapted from the code in book "Numerical Recipes in C" by Press et al.

    mm=m1+m2+1
    l=m1

    b = copy.deepcopy(br)

    for k in range(1, n+1):
        i = int(indx[k-1])

        if i != k:
            z1, z2 = swap(b[k-1], b[i-1])
            b[k-1] = z1
            b[i-1] = z2

        if l < n:
            l = l + 1

        for i in range(k+1, l+1):
            b[i-1] -= al[k-1, i-k-1]*b[k-1]

    l = 1

    for i in range(n, 0, -1):
        temp = b[i - 1]
        for k in range(2, l+1):
            temp = temp - a[i-1, k-1]*b[k+i-2]


        b[i-1]=temp/a[i-1,0]

        if l < mm:
            l = l + 1

    return b


def bdy_proc(i, j, u, N, M, dx, dy):
    # This function calculates the value of u and A and the values of their neighboring points
    # as given by the stencil in the case of no-flux boundary conditions

    u0 = u[j*N+i]

    if (i==0): #*left outer bdy*
        uxp=u[j*N+i+1]
        uxpp=u[j*N+i+2]
        uxm=uxp
        uxmm=uxpp

        Axp=Ax((i+1)*dx, j*dy)
        Axm=Axp


        if (j == M-1): #/*upper left corner*/
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]
            uyp=uym
            uypp=uymm

            uxpym=u[(j-1)*N+i+1]
            uxpyp=uxpym

            Aym=Ax(i*dx,(j-1)*dy)
            Ayp=Aym


        elif (j==0): # /*lower left corner*/
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=uyp
            uymm=uypp
            uxpyp=u[(j+1)*N+i+1]
            uxpym=uxpyp

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ayp


        elif (j == M-2): # /*inner upper left*/
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uymm=u[(j-2)*N+i]

            uxpyp=u[(j+1)*N+i+1]
            uxpym=u[(j-1)*N+i+1]

            uypp=u0

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)


        elif (j==1): # /*inner lower left*/
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]

            uxpyp=u[(j+1)*N+i+1]
            uxpym=u[(j-1)*N+i+1]

            uymm=u0

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)

        else:
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uymm=u[(j-2)*N+i]
            uypp=u[(j+2)*N+i]

            uxpyp=u[(j+1)*N+i+1]
            uxpym=u[(j-1)*N+i+1]

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)



        uxmyp=uxpyp
        uxmym=uxpym

    elif (i == N-1): #/*right outer bdy*/
        uxm=u[j*N+i-1]
        uxmm=u[j*N+i-2]
        uxp=uxm

        uxpp=uxmm

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Axm

        if (j==M-1): # /*upper right corner*/
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]
            uyp=uym
            uypp=uymm
            uxmym=u[(j-1)*N+i-1]
            uxmyp=uxmym

            Aym=Ax(i*dx,(j-1)*dy)
            Ayp=Aym

        elif (j==0): # /*lower right corner*/
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=uyp
            uymm=uypp

            uxmyp=u[(j+1)*N+i-1]
            uxmym=uxmyp

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ayp

        elif (j==M-2): #/*inner upper right*/
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uymm=u[(j-2)*N+i]
            uypp=u0

            uxmyp=u[(j+1)*N+i-1]
            uxmym=u[(j-1)*N+i-1]

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)

        elif (j==1): #/*inner lower right*/
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uymm=u0

            uxmyp=u[(j+1)*N+i-1]
            uxmym=u[(j-1)*N+i-1]

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)

        else:
            uym=u[(j-1)*N+i]
            uyp=u[(j+1)*N+i]
            uymm=u[(j-2)*N+i]
            uypp=u[(j+2)*N+i]

            uxmyp=u[(j+1)*N+i-1]
            uxmym=u[(j-1)*N+i-1]

            Ayp=Ax(i*dx,(j+1)*dy)
            Aym=Ax(i*dx,(j-1)*dy)

        uxpyp=uxmyp
        uxpym=uxmym

    elif (j==M-1 and i>0 and i<N-1): #/*top outer bdy, excluding the corner pts*/
        uym=u[(j-1)*N+i]
        uymm=u[(j-2)*N+i]
        uyp=uym

        uypp=uymm

        uxm=u[j*N+i-1]
        uxp=u[j*N+i+1]

        uxmym=u[(j-1)*N+i-1]
        uxpym=u[(j-1)*N+i+1]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Aym=Ax(i*dx,(j-1)*dy)
        Ayp=Aym

        if (i==1): #/*left*/
            uxpp=u[j*N+i+2]
            uxmm=u0
        elif (i==N-2): #/*right*/
            uxmm=u[j*N+i-2]
            uxpp=u0
        else:
            uxmm=u[j*N+i-2]
            uxpp=u[j*N+i+2]

        uxmyp=uxmym
        uxpyp=uxpym

    elif (j==0 and i>0 and i<N-1): #/*bottom outer bdy, excluding the corner pts*/
        uyp=u[(j+1)*N+i]
        uypp=u[(j+2)*N+i]
        uym=uyp

        uymm=uypp

        uxmyp=u[(j+1)*N+i-1]
        uxpyp=u[(j+1)*N+i+1]

        uxm=u[j*N+i-1]
        uxp=u[j*N+i+1]
        uxmym=uxmyp
        uxpym=uxpyp

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ayp


        if (i==1): #/*left*/
            uxpp=u[j*N+i+2]
            uxmm=u0
        elif (i==N-2): # /*right*/
            uxmm=u[j*N+i-2]
            uxpp=u0
        else:
            uxmm=u[j*N+i-2]
            uxpp=u[j*N+i+2]


    elif (i==1 and j>0 and j<M-1):  #/*inner left bdy*/
        uxp=u[j*N+i+1]
        uxpp=u[j*N+i+2]
        uxm=u[j*N+i-1]
        uxmm=u0

        uxpyp=u[(j+1)*N+i+1]
        uxpym=u[(j-1)*N+i+1]
        uxmym=u[(j-1)*N+i-1]
        uxmyp=u[(j+1)*N+i-1]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ax(i*dx,(j-1)*dy)


        if (j==M-2): # /*inner corner-upper left*/
            uyp=u[(j+1)*N+i]
            uypp=u0
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]

        elif (j==1): #/*inner corner - lower left*/
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=u[(j-1)*N+i]
            uymm=u0
        else:
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]

    elif (i==N-2 and j>0 and j<M-1): #/*inner right bdy*/
        uxp=u[j*N+i+1]
        uxpp=u0
        uxm=u[j*N+i-1]
        uxmm=u[j*N+i-2]

        uxpyp=u[(j+1)*N+i+1]
        uxpym=u[(j-1)*N+i+1]
        uxmym=u[(j-1)*N+i-1]
        uxmyp=u[(j+1)*N+i-1]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ax(i*dx,(j-1)*dy)


        if (j==M-2): #/*inner corner-upper right*/
            uyp=u[(j+1)*N+i]
            uypp=u0
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]

        elif (j==1): # /*inner corner - lower right*/
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=u[(j-1)*N+i]
            uymm=u0

        else:
            uyp=u[(j+1)*N+i]
            uypp=u[(j+2)*N+i]
            uym=u[(j-1)*N+i]
            uymm=u[(j-2)*N+i]

    elif (j==M-2 and i>1 and i<N-2): #/*inner top excluding the inner corner*/
        uyp=u[(j+1)*N+i]
        uypp=u0
        uym=u[(j-1)*N+i]
        uymm=u[(j-2)*N+i]

        uxpyp=u[(j+1)*N+i+1]
        uxpym=u[(j-1)*N+i+1]
        uxmym=u[(j-1)*N+i-1]
        uxmyp=u[(j+1)*N+i-1]

        uxp=u[j*N+i+1]
        uxpp=u[j*N+i+2]
        uxm=u[j*N+i-1]
        uxmm=u[j*N+i-2]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ax(i*dx,(j-1)*dy)

    elif (j==1 and i>1 and i<N-2): #/*inner bottom excluding the bottom corner */

        uyp=u[(j+1)*N+i]
        uypp=u[(j+2)*N+i]
        uym=u[(j-1)*N+i]
        uymm=u0

        uxpyp=u[(j+1)*N+i+1]
        uxpym=u[(j-1)*N+i+1]
        uxmym=u[(j-1)*N+i-1]
        uxmyp=u[(j+1)*N+i-1]

        uxp=u[j*N+i+1]
        uxpp=u[j*N+i+2]
        uxm=u[j*N+i-1]
        uxmm=u[j*N+i-2]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ax(i*dx,(j-1)*dy)

    else:
        uxp=u[j*N+i+1]
        uxpp=u[j*N+i+2]
        uxm=u[j*N+i-1]
        uxmm=u[j*N+i-2]
        uym=u[(j-1)*N+i]
        uyp=u[(j+1)*N+i]
        uymm=u[(j-2)*N+i]
        uypp=u[(j+2)*N+i]

        uxpyp=u[(j+1)*N+i+1]
        uxpym=u[(j-1)*N+i+1]
        uxmyp=u[(j+1)*N+i-1]
        uxmym=u[(j-1)*N+i-1]

        Axm=Ax((i-1)*dx,j*dy)
        Axp=Ax((i+1)*dx,j*dy)

        Ayp=Ax(i*dx,(j+1)*dy)
        Aym=Ax(i*dx,(j-1)*dy)

    u_st = u_stencils(uxp, uxpp, uxm, uxmm, uym, uyp, uymm, uypp, uxpyp, uxpym, uxmyp, uxmym)
    A_st = A_stencils(Axp, Axm, Ayp, Aym)

    return u_st, A_st


def F1(dt, u, U0, N, M, dx, dy):
    # This function calculates the left-hand-side value of the equation in the approximate Newton-ADI method.

    b = np.zeros(N*M)

    for j in range(M):
        for i in range(N):
            u0=u[j*N+i]
            A0=Ax(i*dx, j*dy)
            u_st, A_st = bdy_proc(i, j, u, N, M, dx, dy)

            uxm=u_st.uxm
            uxp=u_st.uxp
            uxpp=u_st.uxpp
            uxmm=u_st.uxmm

            uym=u_st.uym
            uyp=u_st.uyp
            uymm=u_st.uymm
            uypp=u_st.uypp
            uxpyp=u_st.uxpyp
            uxpym=u_st.uxpym
            uxmyp=u_st.uxmyp
            uxmym=u_st.uxmym

            Axm=A_st.Axm
            Axp=A_st.Axp
            Ayp=A_st.Ayp
            Aym=A_st.Aym

            t1 = u0+uxp
            t5 = eps*eps
            t10 = uxp*uxp
            t15 = 2.0*uxp
            t17 = dx*dx
            t18 = 1/t17
            t21 = dy*dy
            t22 = 1/t21
            t28 = u0*u0
            t32 = A0*t5*(1.0-eps/u0)/t28/u0
            t33 = 2.0*u0
            t35 = (uxp-t33+uxm)*t18
            t37 = (uyp-t33+uym)*t22
            t40 = 1/dx
            t42 = u0+uxm
            t50 = uxm*uxm
            t55 = 2.0*uxm
            t65 = u0+uyp
            t73 = uyp*uyp
            t78 = 2.0*uyp
            t85 = 1/dy
            t87 = u0+uym
            t95 = uym*uym
            t100 = 2.0*uym


            t110 = (t1*t1*t1*(-Axp*t5*(1.0-eps/uxp)/t10/uxp+(uxpp-t15+u0)*t18+(uxpyp-\
                t15+uxpym)*t22+t32-t35-t37)*t40/8.0-t42*t42*t42*(-t32+t35+t37+Axm*t5*(1.0-eps/\
                uxm)/t50/uxm-(u0-t55+uxmm)*t18-(uxmyp-t55+uxmym)*t22)*t40/8.0)*t40+(t65*t65*t65\
            *(-Ayp*t5*(1.0-eps/uyp)/t73/uyp+(uxpyp-t78+uxmyp)*t18+(uypp-t78+u0)*t22+t32-t35\
            -t37)*t85/8.0-t87*t87*t87*(-t32+t35+t37+Aym*t5*(1.0-eps/uym)/t95/uym-(uxpym-\
            t100+uxmym)*t18-(u0-t100+uymm)*t22)*t85/8.0)*t85

            u0old=U0[j*N+i]

            t107 = u0-u0old+dt*t110

            b[j*N+i]=t107*(-1)

    return b


def timestep(dt, u, U0, N, M, dx, dy): #/* return 1=good, 0=bad */
    # This function solves the Jacobian matrix that arises from the equation in Newton's iteration.

    Jx = np.zeros((N, 5))
    Jy = np.zeros((M, 5))


    bx = np.zeros(N)
    by = np.zeros(M)


    b = F1(dt, u, U0, N, M, dx, dy)
    ur = copy.deepcopy(u)

    for j in range(M):
        for i in range(N):
            u0=ur[j*N+i]
            A0=Ax(i*dx,j*dy)

            u_st, A_st = bdy_proc(i, j, ur, N, M, dx, dy)

            uxm=u_st.uxm
            uxp=u_st.uxp
            uxpp=u_st.uxpp
            uxmm=u_st.uxmm

            uym=u_st.uym
            uyp=u_st.uyp
            uymm=u_st.uymm
            uypp=u_st.uypp
            uxpyp=u_st.uxpyp
            uxpym=u_st.uxpym
            uxmyp=u_st.uxmyp
            uxmym=u_st.uxmym

            Axm=A_st.Axm
            Axp=A_st.Axp
            Ayp=A_st.Ayp
            Aym=A_st.Aym

            t1 = u0+uxm
            t6 = dx*dx
            t7 = t6*t6
            t9 = dt*t1*t1*t1/t7/8.0

            Jx[i, 0]=t9

            t1 = u0+uxm
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*u0
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t17 = u0*u0
            t22 = 2.0*uxm
            t30 = 1.0-eps/uxm
            t31 = uxm*uxm
            t40 = u0+uxp
            t51 = t31*t31
            t63 = 1/dx
            t68 = dt*(-3.0/2.0*t2*(t4*(uxp-t5+uxm)+t9*(uyp-t5+uym)-A0*t12*(1.0-eps/u0 \
          )/t17/u0-t4*(u0-t22+uxmm)-t9*(uxmyp-t22+uxmym)+Axm*t12*t30/t31/uxm)*t4+(-t40* \
          t40*t40/t3/dx/8.0-t2*t1*(3.0*t4+Axm*(t12*eps/t51/uxm-3.0*t12*t30/t51))*t63/2.0) \
          *t63)


            Jx[i, 1]=t68

            t1 = u0+uxp
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*uxp
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t17 = uxp*uxp
            t22 = 2.0*u0
            t24 = t4*(uxp-t22+uxm)
            t26 = t9*(uyp-t22+uym)
            t30 = 1.0-eps/u0
            t31 = u0*u0
            t35 = A0*t12*t30/t31/u0
            t38 = 1/dx
            t40 = u0+uxm
            t41 = t40*t40/4.0
            t42 = 2.0*uxm
            t51 = uxm*uxm
            t65 = t31*t31
            t75 = 3.0*t4+A0*(t12*eps/t65/u0-3.0*t12*t30/t65)
            t86 = 1.0+dt*(3.0/2.0*(t2*(t4*(uxpp-t5+u0)+t9*(uxpyp-t5+uxpym)-Axp*t12*( \
          1.0-eps/uxp)/t17/uxp-t24-t26+t35)*t38-t41*(t24+t26-t35-t4*(u0-t42+uxmm)-t9*( \
          uxmyp-t42+uxmym)+Axm*t12*(1.0-eps/uxm)/t51/uxm)*t38)*t38+(t2*t1*t75*t38/2.0+t41 \
          *t40*t75*t38/2.0)*t38)


            Jx[i, 2]=t86

            t1 = u0+uxp
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*uxp
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t16 = 1.0-eps/uxp
            t17 = uxp*uxp
            t22 = 2.0*u0
            t31 = u0*u0
            t44 = t17*t17
            t56 = 1/dx
            t58 = u0+uxm
            t68 = dt*(3.0/2.0*t2*(t4*(uxpp-t5+u0)+t9*(uxpyp-t5+uxpym)-Axp*t12*t16/t17 \
          /uxp-t4*(uxp-t22+uxm)-t9*(uyp-t22+uym)+A0*t12*(1.0-eps/u0)/t31/u0)*t4+(t2*t1*( \
          -3.0*t4-Axp*(t12*eps/t44/uxp-3.0*t12*t16/t44))*t56/2.0-t58*t58*t58/t3/dx/8.0)* \
          t56)


            Jx[i, 3]=t68

            t1 = u0+uxp
            t6 = dx*dx
            t7 = t6*t6
            t9 = dt*t1*t1*t1/t7/8.0

            Jx[i, 4]=t9


        Jx[0, 3]+=Jx[0, 1]
        Jx[0, 4]+=Jx[0, 0]

        Jx[1, 2]+=Jx[1, 0]

        Jx[N-1, 1]+=Jx[N-1, 3]
        Jx[N-1, 0]+=Jx[N-1, 4]

        Jx[N-2, 2]+=Jx[N-2, 4]

        for i in range(N):
            bx[i]=b[j*N+i]

        Jx, J1x, indx = band2lu(Jx,N,2,2)
        bx = bandsolve(Jx,N,2,2,J1x,indx,bx)

        for i in range(N):
            b[j*N+i]=bx[i]

    for i in range(N):
        for j in range(M):
            u0=ur[j*N+i]
            A0=Ax(i*dx,j*dy)
            u_st, A_st = bdy_proc(i, j, ur, N, M, dx, dy)

            uxm=u_st.uxm
            uxp=u_st.uxp
            uxpp=u_st.uxpp
            uxmm=u_st.uxmm

            uym=u_st.uym
            uyp=u_st.uyp
            uymm=u_st.uymm
            uypp=u_st.uypp
            uxpyp=u_st.uxpyp
            uxpym=u_st.uxpym
            uxmyp=u_st.uxmyp
            uxmym=u_st.uxmym

            Axm=A_st.Axm
            Axp=A_st.Axp
            Ayp=A_st.Ayp
            Aym=A_st.Aym


            t1 = u0+uym
            t6 = dy*dy
            t7 = t6*t6
            t9 = dt*t1*t1*t1/t7/8.0

            Jy[j, 0]=t9


            t1 = u0+uym
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*u0
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t17 = u0*u0
            t22 = 2.0*uym
            t30 = 1.0-eps/uym
            t31 = uym*uym
            t40 = u0+uyp
            t51 = t31*t31
            t63 = 1/dy
            t68 = dt*(-3.0/2.0*t2*(t4*(uxp-t5+uxm)+t9*(uyp-t5+uym)-A0*t12*(1.0-eps/u0\
                    )/t17/u0-t4*(uxpym-t22+uxmym)-t9*(u0-t22+uymm)+Aym*t12*t30/t31/uym)*t9+(-t40*\
                    t40*t40/t8/dy/8.0-t2*t1*(3.0*t9+Aym*(t12*eps/t51/uym-3.0*t12*t30/t51))*t63/2.0)\
                    *t63)


            Jy[j, 1]=t68

            t1 = u0+uyp
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*uyp
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t17 = uyp*uyp
            t22 = 2.0*u0
            t24 = t4*(uxp-t22+uxm)
            t26 = t9*(uyp-t22+uym)
            t30 = 1.0-eps/u0
            t31 = u0*u0
            t35 = A0*t12*t30/t31/u0
            t38 = 1/dy
            t40 = u0+uym
            t41 = t40*t40/4.0
            t42 = 2.0*uym
            t51 = uym*uym
            t65 = t31*t31
            t75 = 3.0*t9+A0*(t12*eps/t65/u0-3.0*t12*t30/t65)
            t86 = 1.0+dt*(3.0/2.0*(t2*(t4*(uxpyp-t5+uxmyp)+t9*(uypp-t5+u0)-Ayp*t12*(\
                        1.0-eps/uyp)/t17/uyp-t24-t26+t35)*t38-t41*(t24+t26-t35-t4*(uxpym-t42+uxmym)-t9*\
                        (u0-t42+uymm)+Aym*t12*(1.0-eps/uym)/t51/uym)*t38)*t38+(t2*t1*t75*t38/2.0+t41*\
                        t40*t75*t38/2.0)*t38)


            Jy[j, 2]=t86

            t1 = u0+uyp
            t2 = t1*t1/4.0
            t3 = dx*dx
            t4 = 1/t3
            t5 = 2.0*uyp
            t8 = dy*dy
            t9 = 1/t8
            t12 = eps*eps
            t16 = 1.0-eps/uyp
            t17 = uyp*uyp
            t22 = 2.0*u0
            t31 = u0*u0
            t44 = t17*t17
            t56 = 1/dy
            t58 = u0+uym
            t68 = dt*(3.0/2.0*t2*(t4*(uxpyp-t5+uxmyp)+t9*(uypp-t5+u0)-Ayp*t12*t16/t17 \
                        /uyp-t4*(uxp-t22+uxm)-t9*(uyp-t22+uym)+A0*t12*(1.0-eps/u0)/t31/u0)*t9+(t2*t1*( \
                        -3.0*t9-Ayp*(t12*eps/t44/uyp-3.0*t12*t16/t44))*t56/2.0-t58*t58*t58/t8/dy/8.0)*\
                        t56)

            Jy[j, 3]=t68

            t1 = u0+uyp
            t6 = dy*dy
            t7 = t6*t6
            t9 = dt*t1*t1*t1/t7/8.0

            Jy[j, 4]=t9

        Jy[0, 3]+=Jy[0, 1]
        Jy[0, 4]+=Jy[0, 0]

        Jy[1, 2]+=Jy[1, 0]

        Jy[M-1, 1]+=Jy[M-1, 3]
        Jy[M-1, 0]+=Jy[M-1, 4]

        Jy[M-2, 2]+=Jy[M-2, 4]

        for j in range(M):
            by[j]=b[j*N+i]

        Jy, J1y, indy = band2lu(Jy,M,2,2)
        by = bandsolve(Jy,M,2,2,J1y,indy,by)

        for j in range(M):
            b[j*N+i]=by[j]


    for i in range(N*M):
        ur[i]+=b[i]

    return ur, b



def adi_euler(U0, dt, num_steps, Lx, Ly, N, M, A1, A2, tol):
    # This function updates the thin film profile over num_steps time steps
    # using the approximate Newton-ADI method and the backward Euler's method.

    u = np.zeros(N*M)
    b = np.zeros(N*M)

    dx1 = Lx / (N - 1)
    dy1 = Ly / (M - 1)

    U0_r = copy.deepcopy(U0)

    for kk in range(num_steps):
        count = 0

        for i in range(N*M):
            b[i] = 1.0
            u[i] = U0_r[i]

        while linalg.norm(b) > tol:
            u, b = timestep(dt, u, U0_r, N, M, dx1, dy1)
            count = count + 1

            if (count > 200):
                print("Too many iterations\n")
                exit()

        for i in range(N * M):
            U0_r[i] = u[i]

    return u



comm = MPI.COMM_WORLD
rank = comm.Get_rank()


TINY = 1.0e-20

eps = .1

A1 = 1.0
A2 = 10.0

Lx = 2.0
Ly = 2.0

# Number of grid points in the fine grid
N = 100
M = 100

# Number of grid points in the coarse grid
Nc = 50
Mc = 50

dx0 = Lx / (N - 1)
dy0 = Ly / (M - 1)

dx2 = Lx / (Nc - 1)
dy2 = Ly / (Mc - 1)

dt = 1e-3
T_end_time = 2

NumTimeInterval = comm.size
LengthTimeInterval = T_end_time / NumTimeInterval

FineNumTimeStepsPerInterval = round(LengthTimeInterval / dt)
dt = LengthTimeInterval / FineNumTimeStepsPerInterval


r0 = 10

dT = r0*dt
CoarseNumTimeStepsPerInterval = round(LengthTimeInterval / dT)

dT = LengthTimeInterval/CoarseNumTimeStepsPerInterval

CorrFlag = 1
NumParSweeps = 3

tol_c = 1e-5
tol = 1e-5

x_sol_fn = np.zeros((N*M, NumTimeInterval))
x_ic_vec = np.zeros((N*M, NumTimeInterval))
x_sol = np.zeros((N*M, NumTimeInterval + 1))

x_sol_c = np.zeros((Nc*Mc, NumTimeInterval + 1))

g1 = np.zeros((N*M, NumTimeInterval))

U0 = np.zeros(N*M)
U0_c = np.zeros(Nc*Mc)

for j in range(M):
    for i in range(N):
        yy = j*dy0
        xx = i*dx0

        U0[j * N + i] = .2

for j in range(Mc):
    for i in range(Nc):
        yy = j*dy2
        xx = i*dx2

        U0_c[j * Nc + i] = .2


xvec = np.linspace(0, Lx, Nc)
yvec = np.linspace(0, Ly, Mc)
Xc, Yc = np.meshgrid(xvec, yvec)

xvec_f = np.linspace(0, Lx, N)
yvec_f = np.linspace(0, Ly, M)
Xf, Yf = np.meshgrid(xvec_f, yvec_f)


start_time = MPI.Wtime()

if rank == 0:
    
    # load serial solution
    x_true_vec = np.loadtxt("film_serial_final2.txt")
    x_true = x_true_vec[:, 2]


    x_sol[:, 0] = U0

    x_sol_c[:, 0] = U0_c
    
    for ll in range(NumTimeInterval):


        if Nc < N or Mc < M:

            X_ic = adi_euler(x_sol_c[:, ll], dT, CoarseNumTimeStepsPerInterval, Lx, Ly, Nc, Mc, A1, A2, tol_c)

            x_sol_c[:, ll+1] = X_ic

        else:
            X_ic = adi_euler(x_sol[:, ll], dT, CoarseNumTimeStepsPerInterval, Lx, Ly, Nc, Mc, A1, A2, tol_c)
            x_sol[:, ll+1] = X_ic

    if Nc < N or Mc < M:
        for ll in range(NumTimeInterval):
            Zc = np.reshape(x_sol_c[:, ll], (Mc, Nc))
            film_interp = RegularGridInterpolator((xvec, yvec), Zc, method='cubic')
            znew = film_interp((Xf, Yf))
            x_sol[:, ll+1] = np.reshape(znew, N*M)


    x_ic_vec[:, 0:] = x_sol[:, 0:NumTimeInterval]
    x_sol_fn[:, :] = x_sol[:, 1:]
    g1[:, :] = x_sol[:, 1:]

    AbsErrX = np.linalg.norm(x_sol[:, -1] - x_true)
    AbsErrX1 = AbsErrX/np.linalg.norm(x_true)

    #Print out the true error and the relative true error after the initial serial sweep

    print('AbsErrX=%.16g, AbsErrX1=%.16g\n' % (AbsErrX, AbsErrX1))




if rank == 0:

    x_send_buf = np.split(x_ic_vec, NumTimeInterval, axis=1)

    comm.send(x_ic_vec, dest=1, tag=11111)

else:
    x_send_buf = None


if rank == 1:
    x_ic_vec = comm.recv(source=0, tag=11111)

x_recv_obj = comm.scatter(x_send_buf, root=0)
x_sol_ic = x_recv_obj


x_sol_ic1 = np.squeeze(x_sol_ic, axis = 1)

comm.Barrier()


for ii in range(NumParSweeps):

    xf = adi_euler(x_sol_ic1, dt, FineNumTimeStepsPerInterval, Lx, Ly, N, M, A1, A2, tol)

    vec = comm.allgather(xf)

    comm.Barrier()
    
    if rank < 2:
        for qq in range(ii, NumTimeInterval):
            x_sol_fn[:, qq] = vec[qq]

    comm.Barrier()
        

    
    if rank == 0:
        for jj in range(ii + 1, NumTimeInterval):

            if Nc < N or Mc < M:
                Zf = np.reshape(x_sol_fn[:, jj-1], (M, N))
                film_interp = RegularGridInterpolator((xvec_f, yvec_f), Zf, method='cubic')
                znew2 = film_interp((Xc, Yc))
                X_g2_ic = np.reshape(znew2, Nc * Mc)
            else:
                X_g2_ic = x_sol_fn[:, jj - 1]

            X_g2 = adi_euler(X_g2_ic, dT, CoarseNumTimeStepsPerInterval, Lx, Ly, Nc, Mc, A1, A2, tol_c)

            if Nc < N or Mc < M:
                Zc2 = np.reshape(X_g2, (Mc, Nc))

                film_interp2 = RegularGridInterpolator((xvec, yvec), Zc2, method='cubic')

                znew2 = film_interp2((Xf, Yf))
                X_g2_interp = np.reshape(znew2, N*M)

                x_sol_fn[:, jj] = x_sol_fn[:, jj] + X_g2_interp - g1[:, jj]
                g1[:, jj] = X_g2_interp

            else:
                x_sol_fn[:, jj] = x_sol_fn[:, jj] + X_g2 - g1[:, jj]

                g1[:, jj] = X_g2
        

        for kk in range(ii+1, NumTimeInterval):
            x_ic_vec[:, kk] = x_sol_fn[:, kk-1]


        send_buf_x = np.split(x_ic_vec, NumTimeInterval, axis = 1)

        #Print out the true error and the relative true error after each iteration
        print(ii)
        RelErrX = np.linalg.norm(x_sol[:, -1] - x_sol_fn[:, -1])/np.linalg.norm(x_sol_fn[:, -1])
        AbsErrX = np.linalg.norm(x_sol_fn[:, -1] - x_true)
        AbsErrX1 = AbsErrX/np.linalg.norm(x_true)
 
        print('RelErr= %.16g, AbsErrX=%.16g, AbsErrX1=%.16g\n' % (RelErrX, AbsErrX, AbsErrX1))
        
        x_sol[:, 1:] = x_sol_fn
 
        
    else:
        send_buf_x = None


    recv_obj_x = comm.scatter(send_buf_x, root=0)
    x_sol_ic = recv_obj_x
    x_sol_ic1 = np.squeeze(x_sol_ic, axis = 1)
    comm.Barrier()


end_time = MPI.Wtime()

if rank == 0:
    run_time = end_time - start_time
    print("Total number of iterations performed = %d\t: Total time = %.16g\n" % (ii + 1, run_time))
    
