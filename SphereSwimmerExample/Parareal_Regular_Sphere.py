import math
import copy
import numpy as np
from scipy.integrate import solve_ivp
from scipy import linalg
from mpi4py import MPI

# This code calculates the example of an elastic sphere whose forces are modeled by Hookean springs, using the classic Parareal method.
# The code for Method of Regularized Stokeslets is adapted from the code provided by Minghao W. Rostami (SUNY Binghamton).


def Forcespr_new2(X,K,ed,rels):
    # This function calculates the forces exerted at points X on the spherical surface
    # modeled by Hookean springs with spring constant K.
    # ed and rels are inputs that specify the edges and resting length of springs between neighboring points.

    Nb = X.shape[0]

    F = np.zeros((Nb, 3))
    numed = ed.shape[0]

    for ei in range(numed):
        A = X[int(ed[ei, 0]-1), :]
        B = X[int(ed[ei, 1]-1), :]
        el = np.linalg.norm(B-A)
        F[int(ed[ei, 0] - 1), :] = F[int(ed[ei, 0]-1), :] - K/el*(B-A)*(el-rels[ei])
        F[int(ed[ei, 1] - 1), :] = F[int(ed[ei, 1]-1), :] - K/el*(A-B)*(el-rels[ei])

    return -F


def VelocityRegSto7half(f, ep, Xb, Xeval, Qweights, SourcedA):
    # This function evaluates linear velocities using the Method of Regularized
    # Stokeslets in 3D. The 7/2 blob function is used.

    # Source points exert forces only to the fluid. The domain is unbounded.

    # Inputs:
    # Xb: 3 by nsrc matrix. nsrc is the number of source points. The ith column
    #     contains the x, y, z coordinates of the ith source point.
    # f: 3 by nsrc matrix. The ith column contains the x, y, z forces exerted
    #    by the ith source point.
    # Xeval: 3 by neval matrix. neval is the number of evaluation points. The
    #        ith column contains the x, y, z coordinates of the ith evaluation
    #        point.
    # ep: 1 by 1 scalar. The regularization parameter, epsilon, in the Method
    #     of Regularized Stokeslets.
    # SourcedA: Currently set to 1 for exact calculation of the matix-vector product.
    #           This setting can be adapted to the Kernel-Independent Fast Multipole Method (KIFMM).
    #           In the KIFMM setting, this is the area of a small square (surface version) or the volumn of a small cube
    #           (corona version).
    # Qweights: 1 by nsrc row vector. Currently set to ones(1,nsrc) for exact calculation of the matrix-vector product.
    #           This setting can be adapted to the Kernel-Independent Fast Multipole Method (KIFMM)
    #           In the KIFMM setting, this vector holds the weights in the quadrature rule used for approximating
    #           integrals.
    # Output:
    # U: 3 by neval matrix, The ith column contains the x, y, z linear
    #    velocities of the ith evaluation point.

    # For further details on KIFMM, please refer to the article below:
    # Rostami, M.W., & Olson, S.D. (2016). Kernel-independent fast multipole method within the framework of regularized Stokeslets. Journal of Fluids and Structures, 67, 60-84.


    
    neval = Xeval.shape[1]
    nsrc = Xb.shape[1]
    
    U = np.zeros((3, neval))
    eps0 = ep**2
    cQweights = (SourcedA/8/np.pi)*Qweights

    ff = np.zeros_like(f)
    ff[0, :] = f[0, :]*cQweights
    ff[1, :] = f[1, :]*cQweights
    ff[2, :] = f[2, :]*cQweights

    dxmat = np.zeros((3, nsrc))
    u = np.zeros(3)

    for i in range(neval):
        xeval = Xeval[:, i]

        dxmat[0, :] = xeval[0] - Xb[0, :]
        dxmat[1, :] = xeval[1] - Xb[1, :]
        dxmat[2, :] = xeval[2] - Xb[2, :]

        rsvec = np.sum(dxmat**2, axis = 0)
        epsprsvec = eps0 + rsvec
        epsprsvec3 = epsprsvec**(3. / 2)
        h1vec = (2 * eps0 + rsvec) / epsprsvec3
        h2vec = 1. / epsprsvec3
        fdotdxvec = (ff[0, :]) * dxmat[0, :] + (ff[1, :]) * dxmat[1, :]+(ff[2, :]) * dxmat[2, :]
        fdotdxvec_h2vec = fdotdxvec * h2vec

        u[0] = np.matmul(ff[0, :], np.transpose(h1vec)) + np.matmul(dxmat[0, :], np.transpose(fdotdxvec_h2vec))
        u[1] = np.matmul(ff[1, :], np.transpose(h1vec)) + np.matmul(dxmat[1, :], np.transpose(fdotdxvec_h2vec))
        u[2] = np.matmul(ff[2, :], np.transpose(h1vec)) + np.matmul(dxmat[2, :], np.transpose(fdotdxvec_h2vec))

        U[:, i] = u

    return U


def RK2(dt,maxt,X,rels,ed,K,ep,mu,r_shear,v,A0):
    # This function calculates the position of the points on the spherical surface over maxt time steps using Runge-Kutta second order method (RK2)

    NumPts=X.shape[1]
    ub=np.zeros((3,NumPts))
    numtri=v.shape[0]

    Xr = copy.deepcopy(X)

    for i in range(maxt):
        F = Forcespr_new2(np.transpose(Xr),K,ed,rels)
        U = VelocityRegSto7half(np.transpose(F), ep, Xr, Xr, np.ones((1,NumPts)), 1)
        ub[0, :] = r_shear*Xr[2, :]

        XX = Xr + .5*dt*(1.0/mu + ub)

        F = Forcespr_new2(np.transpose(XX), K, ed, rels)
        U = VelocityRegSto7half(np.transpose(F), ep, XX, XX, np.ones((1,NumPts)), 1)

        ub[0, :] = r_shear*Xr[2, :]
        Xr = Xr + dt*(1/mu*U+ub)

        A = 0
        for j in range(numtri):
            A = A + tarea(Xr[:, int(v[j,0]-1)], Xr[:, int(v[j,1]-1)], Xr[:, int(v[j,2]-1)])

        surf_area_RelErr = abs(A-A0)/A0

        if surf_area_RelErr>0.01:
            print("Relative error in surface area too big!\n")
    
    return Xr


def Euler(dt,maxt,X,rels,ed,K,ep,mu,r_shear,v,A0):
    # This function calculates the position of the points on the spherical surface over maxt time steps using
    # the forward Euler's method

    NumPts=X.shape[1]
    ub=np.zeros((3, NumPts))
    numtri=v.shape[0]

    Xr = copy.deepcopy(X)

    for i in range(maxt):
        F = Forcespr_new2(np.transpose(Xr),K,ed,rels)
        U = VelocityRegSto7half(np.transpose(F), ep, Xr, Xr, np.ones((1,NumPts)), 1)
        U = U/mu

        ub[0, :] = r_shear*Xr[2, :]

        Xr = Xr + (U + ub)*dt

        ub[:, :] = 0

        A = 0

        for j in range(numtri):
            A = A + tarea(Xr[:, int(v[j,0]-1)], Xr[:, int(v[j,1]-1)], Xr[:, int(v[j,2]-1)])

        surf_area_RelErr = abs(A-A0)/A0

        if surf_area_RelErr>0.01:
            print("Relative error in surface area too big!\n")
    
    return Xr


def tarea(X,Y,Z):
    # This function calculates the area of a triangle formed by three points X, Y and Z

    x1 = Y - X
    x2 = Z - X
    s = .5*np.linalg.norm(np.cross(x1, x2))

    return s



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dt = 1e-3

NumTimeInterval = comm.size

T = 10

LengthTimeInterval = T / NumTimeInterval

FineNumTimeStepsPerInterval = round(LengthTimeInterval / dt)
dt = LengthTimeInterval / FineNumTimeStepsPerInterval

r0 = 4

dT = r0*dt
CoarseNumTimeStepsPerInterval = round(LengthTimeInterval / dT)

dT = LengthTimeInterval/CoarseNumTimeStepsPerInterval


CorrFlag = 1
NumParSweeps = 1

#Load the initial profile of points on the spherical surface
ed_rels = np.loadtxt("sphere_ed_rels.txt")
ed = ed_rels[:, 0:2]
rels = ed_rels[:, 2]

x_init = np.loadtxt("sphere_points.txt")
M = x_init.shape[0]
# M is the total number of vertices

v = np.loadtxt("sphere_v.txt")
num0 = M

#Use maximum edge length to determine epsilon
max_elength = 0
numed = ed.shape[0]
for ei in range(numed):
    A = x_init[int(ed[ei, 0]-1), :]
    B = x_init[int(ed[ei, 1]-1), :]
    el = np.linalg.norm(A - B)
    max_elength = max(max_elength,el)

ep = .3*max_elength

EP = ep

K = 0.1 #spring constant between boundary points
mu = 1e-3 #viscosity

r_shear = .1

g1 = np.zeros((3, M, NumTimeInterval))


x_sol_fn = np.zeros((3, M, NumTimeInterval))
x_ic_vec = np.zeros((3, M, NumTimeInterval))
x_sol = np.zeros((3, M, NumTimeInterval + 1))


numtri = v.shape[0]

triarea=np.zeros(numtri)
for i in range(numtri):
    triarea[i] = tarea(x_init[int(v[i,0]-1), :], x_init[int(v[i,1]-1), :], x_init[int(v[i,2]-1), :])
    
A0 = np.sum(triarea)

start_time = MPI.Wtime()

if rank == 0:
    
    #load serial solution 
    x_true = np.transpose(np.loadtxt("Sphere_Serial_T1_Final.txt"))

    x_sol[:, :, 0] = np.transpose(x_init)

  
    for ll in range(NumTimeInterval):
        X_ic = Euler(dT, CoarseNumTimeStepsPerInterval, x_sol[:,:, ll], rels, ed, K, EP, mu, r_shear, v, A0)
        x_sol[:,:,ll+1] = X_ic


    x_ic_vec[:, :, 0:] = x_sol[:, :, 0:NumTimeInterval]
    x_sol_fn[:, :, :] = x_sol[:, :, 1:]
    g1[:, :, :] = x_sol[:, :, 1:]

    #Print out the true error and the true relative error after the initial serial sweep
    AbsErrX = np.linalg.norm(x_sol[:, :, -1] - x_true)
    print(AbsErrX)
    AbsErrX1 = AbsErrX/np.linalg.norm(x_true)
    print(AbsErrX1)
                                                                                                        

if rank == 0:

    x_send_buf = np.split(x_ic_vec, NumTimeInterval, axis=2)

    comm.send(x_ic_vec, dest=1, tag=11111)

else:
    x_send_buf = None


if rank == 1:
    x_ic_vec = comm.recv(source=0, tag=11111)

x_recv_obj = comm.scatter(x_send_buf, root=0)
x_sol_ic = x_recv_obj

comm.Barrier()


for ii in range(NumParSweeps):

    xf = RK2(dt, FineNumTimeStepsPerInterval, x_sol_ic[:, :, 0], rels, ed, K, ep, mu, r_shear, v, A0)

    vec = comm.allgather(xf)

    comm.Barrier()
    
    if rank < 2:
        for qq in range(ii, NumTimeInterval):
            x_sol_fn[:,:,qq] = vec[qq]

    
    for jj in range(ii + 1, NumTimeInterval):
        if rank == 0:

            X_g2 = Euler(dT, CoarseNumTimeStepsPerInterval, x_sol_fn[:, :, jj-1], rels, ed, K, EP, mu, r_shear, v, A0)
            
            x_sol_fn[:,:,jj] = x_sol_fn[:,:,jj] + X_g2 - g1[:,:,jj]

            g1[:,:,jj] = X_g2
        

    if rank == 0: 
        for kk in range(ii+1, NumTimeInterval):
            x_ic_vec[:, :, kk] = x_sol_fn[:, :, kk-1]


        send_buf_x = np.split(x_ic_vec, NumTimeInterval, axis = 2)

        #Print out the true error and the relative true error after each iteration
        print(ii)
        RelErrX = np.linalg.norm(x_sol[:, :, -1] - x_sol_fn[:, :, -1])/np.linalg.norm(x_sol_fn[:, :, -1])
        AbsErrX = np.linalg.norm(x_sol_fn[:, :, -1] - x_true)
        AbsErrX1 = AbsErrX/np.linalg.norm(x_true)
 
        print('RelErr= %.16g, AbsErrX=%.16g, AbsErrX1=%.16g\n' % (RelErrX, AbsErrX, AbsErrX1))
        
        x_sol[:, :, 1:] = x_sol_fn
    
        
    else:
        send_buf_x = None


    recv_obj_x = comm.scatter(send_buf_x, root=0)
    x_sol_ic = recv_obj_x
    comm.Barrier()


end_time = MPI.Wtime()

if rank == 0:
    run_time = end_time - start_time
    print("Number of iterations performed = %d\t: Total time = %.16g\n" % (ii + 1, run_time))
    


    





    

    



        
