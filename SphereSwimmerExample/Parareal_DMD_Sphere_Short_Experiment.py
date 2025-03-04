import math
import copy
import numpy as np
from scipy.integrate import solve_ivp
from scipy import linalg
from mpi4py import MPI

#This code calculates the example of an elastic sphere whose forces are modeled by Hookean springs.
#The motion of the sphere is solved for time interval [0,10]. Parareal-HODMD is applied for one iteration.

#The code for Method of Regularized Stokeslets is adapted from the code provided by Minghao W. Rostami (SUNY Binghamton).
#The code for HODMD is adapted from Soledad Le Clainche and José M. Vega
# from the link: https://www.researchgate.net/publication/346396464_Matlab_Codes_from_the_book_Higher_order_Dynamic_Mode_Decomposition_and_its_Applications.


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
        #fcn to be edited
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


def tarea(X,Y,Z):
    # This function calculates the area of a triangle formed by three points X, Y and Z

    x1 = Y - X
    x2 = Z - X
    s = .5*np.linalg.norm(np.cross(x1, x2))

    return s

def Sphere_FE2_Snapshots(dt, maxt, X, rels, ed, K, ep, mu, r_shear, v, A0, rr):
    # This calculates the position of the surface points over maxt time steps using the forward Euler's method
    # and outputs the snapshots of the positions as a snapshot matrix

    Xr = copy.deepcopy(X)
    NumPts=X.shape[1]
    ub=np.zeros((3, NumPts))
 
    num_snapshots = int(maxt/rr) + 1
    
    X_snapshots1 = np.zeros((NumPts*3, num_snapshots))
         
    X_snapshots1[:, 0] = np.reshape(Xr, 3*M)


    # loop over maxt time steps
    for i in range(1, maxt + 1):
        
        F = Forcespr_new2(np.transpose(Xr),K,ed,rels)
        U = VelocityRegSto7half(np.transpose(F), ep, Xr, Xr, np.ones((1,NumPts)), 1)
        U = U/mu

        ub[0, :] = r_shear*Xr[2, :]

        Xr = Xr + (U + ub)*dt

        ub[:, :] = 0
               
        p1 = np.mod(i,rr)
    
        if p1 == 0: 
            q1 = int(i/rr)
            X_snapshots1[:, q1] = np.reshape(Xr, 3*M)
    

    return X_snapshots1

def HODMD(d,V,Time,varepsilon1,varepsilon):
    # DMD-d
    # INPUT:
    # d: parameter of DMD-d (higher order Koopman assumption)
    # V: snapshot matrix
    # Time: vector time
    # varepsilon1: first tolerance (SVD)
    # varepsilon: second tolerance (DMD-d modes)
    # OUTPUT:
    # Vreconst: reconstruction of the snapshot matrix V
    # GrowthRate: growth rate of DMD modes
    # Frequency: frequency of DMD modes(angular frequency)
    # Amplitude: amplitude of DMD modes
    # DMDmode: DMD modes

    J = np.size(V, 0)
    K = np.size(V, 1)

    #STEP 1: SVD of the original data
    U, sigmas, T = np.linalg.svd(V, full_matrices=False)
    Sigma = np.diag(sigmas)
    n = np.size(sigmas)
    T = np.transpose(np.conjugate(T))

    NormS = np.linalg.norm(sigmas)
    kk = 0

    for k in range(n):
        if np.linalg.norm(sigmas[k:n])/NormS > varepsilon1:
            kk = kk + 1

    U = U[:, 0:kk]


    # Spatial complexity: kk
    # Create reduced snapshots matrix

    hatT = np.dot(Sigma[0:kk, 0:kk], np.transpose(np.conjugate(T[:, 0:kk])))

    N = np.size(hatT, 0)

    # Create the modified snapshot matrix
    tildeT = np.zeros((d*N, K-d+1))

    for ppp in range(d):
        tildeT[ppp*N:(ppp+1)*N, :] = hatT[:, ppp:ppp+K-d+1]


    # Dimension reduction
    U1, sigmas1, T1 = np.linalg.svd(tildeT, full_matrices=False)
    Sigma1 = np.diag(sigmas1)
    T1 = np.transpose(np.conjugate(T1))

    Deltat = Time[1]-Time[0]
    n = np.size(sigmas1)

    NormS = np.linalg.norm(sigmas1)
    kk1 = 0

    RRMSEE = np.zeros(n)
    for k in range(n):
        RRMSEE[k] = np.linalg.norm(sigmas1[k:n])/NormS
        if RRMSEE[k] > varepsilon1:
            kk1 = kk1 + 1

    U1 = U1[:, 0:kk1]

    hatT1 = np.dot(Sigma1[0:kk1, 0:kk1], np.transpose(np.conjugate(T1[:, 0:kk1])))

    # Reduced modified snapshot matrix
    K1 = np.size(hatT1, 1)
    tildeU1, tildeSigma, tildeU2 = np.linalg.svd(hatT1[:, 0:K1-1], full_matrices = False)
    tildeU2 = np.transpose(np.conjugate(tildeU2))

    # Reduced modified Koopman matrix
    tildeR = np.dot(hatT1[:, 1:K1], np.dot(tildeU2, np.dot(np.linalg.inv(np.diag(tildeSigma)), np.transpose(np.conjugate(tildeU1)))))

    tildeMM, tildeQ = np.linalg.eig(tildeR)
    eigenvalues = tildeMM

    M = np.size(eigenvalues)
    qq = np.log(np.real(eigenvalues)+np.imag(eigenvalues)*1j)
    GrowthRate = np.real(qq)/Deltat
    Frequency = np.imag(qq)/Deltat

    Q = np.dot(U1, tildeQ)
    Q = Q[(d-1)*N:d*N, :]



    NN = np.size(Q, 0)
    MMM = np.size(Q, 1)

    for m in range(MMM):
        NormQ = Q[:, m]
        Q[:, m] = Q[:, m]/np.linalg.norm(NormQ)


    # Calculate amplitudes
    Mm = np.zeros((NN*K, M), dtype=complex)
    Bb = np.zeros(NN*K)
    aa = np.eye(MMM, dtype = complex)

    for k in range(K):
        Mm[k*NN:(k+1)*NN, :] = np.dot(Q, aa)
        aa[:, :] = np.dot(aa, np.diag(tildeMM))
        Bb[k*NN:(k+1)*NN] = hatT[:, k]


    Ur, Sigmar, Vr = np.linalg.svd(Mm, full_matrices = False)
    Vr = np.transpose(np.conjugate(Vr))

    a = np.dot(Vr, np.linalg.solve(np.diag(Sigmar), np.dot(np.transpose(np.conjugate(Ur)), Bb)))

    u = np.zeros((NN, M), dtype=complex)

    for m in range(M):
        u[:, m] = a[m]*Q[:, m]

    Amplitude = np.zeros(M)

    for m in range(M):
        aca = np.dot(U, u[:, m])
        Amplitude[m] = np.linalg.norm(aca[:])/np.sqrt(J)


    ml = np.size(u, 0)
    nl = np.size(u, 1)
    UU = np.zeros((ml+3, nl), dtype=complex)

    UU[0:ml, :] = u
    UU[ml, :] = GrowthRate
    UU[ml+1, :] = Frequency
    UU[ml+2, :] = Amplitude

    UU = np.transpose(UU)


    lp = np.size(UU, 1)
    lk = np.size(UU, 0)
    ww = np.zeros(lk, dtype=complex)
    zz = np.zeros(lk, dtype=complex)

    ww[:] = UU[:, NN+2]
    I_0 = ww.argsort()[::-1]


    UU1 = np.zeros((lk, lp), dtype = complex)
    for jj in range(lp):
        zz[:] = UU[:, jj]
        UU1[:, jj] = zz[I_0]

    UU = np.transpose(UU1)

    u[:] = UU[0:NN, :]

    GrowthRate = UU[NN, :]
    Frequency = UU[NN+1, :]
    Amplitude = UU[NN+2, :]
    kk3 = 0

    for m in range(M):
        if Amplitude[m]/Amplitude[0] > varepsilon:
            kk3 = kk3 + 1

    u = u[:, 0:kk3]

    GrowthRate = GrowthRate[0:kk3]
    Frequency = Frequency[0:kk3]
    Amplitude = Amplitude[0:kk3]

    # Reconstruction of the original snapshot matrix

    K1 = np.size(Time)
    hatTreconst = np.zeros((N, K1), dtype=complex)

    for k in range(K1):
        hatTreconst[:, k] = ContReconst_SIADS(Time[k], Time[0], u, GrowthRate, Frequency)


    Vreconst = np.matmul(U, hatTreconst)


    return Vreconst



def ContReconst_SIADS(t, t0, u, deltas, omegas):
    N = np.size(u, 0)
    M = np.size(u, 1)
    vv = np.zeros(M, dtype = complex)

    for m in range(M):
        vv[m] = np.exp((deltas[m] + omegas[m]*1j) * (t - t0))

    ContReconst = np.dot(u, vv)

    return ContReconst




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
NumTimeInterval = comm.size

dt = 1e-3  
T = 10

LengthTimeInterval = T / NumTimeInterval

FineNumTimeStepsPerInterval = round(LengthTimeInterval / dt)
dt = LengthTimeInterval / FineNumTimeStepsPerInterval

r0 = 4
dT = r0*dt
CoarseNumTimeStepsPerInterval = round(LengthTimeInterval / dT)
dT = LengthTimeInterval/CoarseNumTimeStepsPerInterval


r2 = 2
dT2 = r2*dT
CoarseNumTimeStepsPerInterval2 = round(LengthTimeInterval / dT2)
dT2 = LengthTimeInterval/CoarseNumTimeStepsPerInterval2

CorrFlag = 1
NumParSweeps = 1

rtest = .5
ntest = round(NumTimeInterval * rtest)


rr = 10

rr = 5

rr0 = 1
qq0 = round(CoarseNumTimeStepsPerInterval / rr0)
ni = NumTimeInterval - ntest


qq2 = round(CoarseNumTimeStepsPerInterval2 / rr0 / rr)

qq1 = round(CoarseNumTimeStepsPerInterval2 / rr0)

# Parameters used in HODMD
varepsilon1 = 1e-5
varepsilon = varepsilon1

d0 = 5

d = 5
rsvec = np.zeros(NumParSweeps)

rsvec[0] = .5
#rsvec[1]=.6


nt_vec = np.zeros(NumParSweeps)

for i in range(NumParSweeps):
    nt_vec[i] = round(qq0/r2*rsvec[i])
    
nt_c1 = int(nt_vec[0])
    
nt_c_max = int(max(nt_vec))


# Load the data for the points on the spherical surface

ed_rels = np.loadtxt("sphere_ed_rels.txt")
ed = ed_rels[:, 0:2]
rels = ed_rels[:, 2]

x_init = np.loadtxt("sphere_points.txt")
M = x_init.shape[0]
# M is the total number of vertices


v = np.loadtxt("sphere_v.txt")
num0 = M

#use maximum edge length to determine epsilon
max_elength = 0
numed = ed.shape[0]
for ei in range(numed):
    A = x_init[int(ed[ei, 0]-1), :]
    B = x_init[int(ed[ei, 1]-1), :]
    el = np.linalg.norm(A - B)
    max_elength = max(max_elength,el)

# Regularization parameter
ep = .3*max_elength

# Use a larger regularization parameter in the coarse solver
EP = .5*max_elength

K = 0.1 #spring constant between boundary points

K_c = 0.1

mu = 1e-3 #viscosity

r_shear = .1


x_sol_fn = np.zeros((3, M, NumTimeInterval))
x_ic_vec = np.zeros((3, M, NumTimeInterval))
x_sol = np.zeros((3, M, NumTimeInterval + 1))

x_sol_c = np.zeros((3, M, NumTimeInterval + 1))


numtri = v.shape[0]

triarea=np.zeros(numtri)
for i in range(numtri):
    triarea[i] = tarea(x_init[int(v[i,0]-1), :], x_init[int(v[i,1]-1), :], x_init[int(v[i,2]-1), :])
    
A0=np.sum(triarea)
start_time = MPI.Wtime()


if rank == 0:
    
    #load serial solution 
    x_true = np.transpose(np.loadtxt("Sphere_Serial_T1_Final.txt"))
        
    X_snapshots_all = np.zeros((3*M, ntest * qq2 + 1))
    X_snapshots_all[:, 0] = np.reshape(np.transpose(x_init), 3*M)
    X_snapshots_old_mat = np.zeros((M*3, nt_c_max + 1, NumTimeInterval))
    X_correct_ic = np.zeros((3, M, NumTimeInterval))
    
    X_snapshots_old_c = np.zeros((M * 3, nt_c_max + 1, NumTimeInterval))
    
    
    #Store the solution value at the endpoint of each subinterval in the serial correction
    X_end0 = np.zeros((3*M, NumTimeInterval))
    X_end2 = np.zeros((3*M, NumTimeInterval))

    x_sol[:, :, 0] = np.transpose(x_init)
    
    ratio_rr = int(rr/rr0*r2)

    for ll in range(ntest):
        X_snapshots = Sphere_FE2_Snapshots(dT, CoarseNumTimeStepsPerInterval, x_sol[:,:, ll], rels, ed, K, ep, mu, r_shear, v, A0, rr0)

        x_sol[:, :, ll + 1] = np.reshape(X_snapshots[:, -1], (3, M))
        X_snapshots_all[:, (ll * qq2 + 1):((ll + 1) * qq2 + 1)] = X_snapshots[:, 1:(qq0+1):ratio_rr]
        
        X_snapshots_old_mat[:, 0:(nt_c1+1), ll] = X_snapshots[:, 0:(nt_c1*r2+1):r2]


 
if rank == 1:
    X_snapshots_all_c = np.zeros((3*M, ntest*qq2 + 1))
    X_snapshots_ni = np.zeros((3*M, ni))
    
    x_sol_c[:, :, 0] = np.transpose(x_init)
    X_snapshots_all_c[:, 0] = np.reshape(np.transpose(x_init), 3*M) 
    
    ratio_rr2 = int(rr/rr0)
    
    for ll in range(ntest):
        X_snapshots_coarse = Sphere_FE2_Snapshots(dT2, CoarseNumTimeStepsPerInterval2, x_sol_c[:,:, ll], rels, ed, K_c, EP, mu, r_shear, v, A0, rr0)    
        x_sol_c[:, :, ll + 1] = np.reshape(X_snapshots_coarse[:, -1], (3, M))
        X_snapshots_all_c[:, (ll * qq2 + 1):((ll + 1) * qq2 + 1)] = X_snapshots_coarse[:, 1:(qq1+1):ratio_rr2]
        
    for ll in range(ntest, NumTimeInterval):
        X_snapshots_coarse_ni = Sphere_FE2_Snapshots(dT2, CoarseNumTimeStepsPerInterval2, x_sol_c[:,:, ll], rels, ed, K_c, EP, mu, r_shear, v, A0, rr0)    
        x_sol_c[:, :, ll + 1] = np.reshape(X_snapshots_coarse_ni[:, -1], (3, M))
        X_snapshots_ni[:, ll - ntest] = X_snapshots_coarse_ni[:, -1]
        #X_snapshots_old_c[:, 0:(nt_c1+1), ll] = X_snapshots_coarse[:, 0:(nt_c1+1)]  
        
        
    comm.send(X_snapshots_ni, dest=0, tag=11)
    comm.send(X_snapshots_all_c, dest=0, tag=1)
  


if rank == 0:

    X_snapshots_ni = comm.recv(source=1, tag=11)
    X_snapshots_all_c = comm.recv(source=1, tag=1)
  
    X_diff_mat = np.zeros((3 * M, ntest * qq2 + 1))
    
    #The difference snapshot matrix of the solution on the first ntest intervals
    X_diff_mat[:, :] = X_snapshots_all - X_snapshots_all_c
   
    X_diff_p = HODMD(d0, X_diff_mat, np.arange(qq2 * NumTimeInterval + 1), varepsilon1, varepsilon)

    for ll in np.arange(ntest, NumTimeInterval):
        X_t1 = X_snapshots_ni[:, ll - ntest] + X_diff_p[:, (ll+1)*qq2]
        x_sol[:, :, ll + 1] = np.reshape(X_t1, (3, M))
        
    x_ic_vec[:, :, :] = x_sol[:, :, 0:NumTimeInterval]
    x_sol_fn[:, :, :] = x_sol[:, :, 1:]
    
    x_send_buf = np.split(x_ic_vec, NumTimeInterval, axis=2)

    comm.send(x_ic_vec, dest=2, tag=11111) 
    comm.send(x_ic_vec, dest=3, tag=12212)    

    AbsErrX = np.linalg.norm(x_sol[:, :, -1] - x_true)
    print(AbsErrX)
    AbsErrX1 = AbsErrX/np.linalg.norm(x_true)
    print(AbsErrX1)    

else:
    x_send_buf = None


if rank == 2:
    x_ic_vec = comm.recv(source=0, tag=11111)

if rank == 3:
    x_ic_vec = comm.recv(source=0, tag=12212)

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
        nt = int(nt_vec[ii])
        if ii > 0:
            nt_c0 = int(nt_vec[ii-1])
                        
        if rank == 2:
            if ii == 0:
                if jj >= ntest:
                    X_snapshots_old = Sphere_FE2_Snapshots(dT, nt*r2, x_ic_vec[:, :, jj], rels, ed, K, ep, mu, r_shear, v, A0, rr0*r2)
                    
                    comm.send(X_snapshots_old, dest=0, tag=101)
            
            else:
                if nt > nt_c0:
                    X_snapshots_old = Sphere_FE2_Snapshots(dT, (nt-nt_c0)*r2, X_correct_ic[:, :, jj], rels, ed, K, ep, mu, r_shear, v, A0, rr0*r2)
                    
                    comm.send(X_snapshots_old, dest=0, tag=101)
                    

        if rank == 3 and ii == 0:
            X_snapshots_old_ct0 = Sphere_FE2_Snapshots(dT2, CoarseNumTimeStepsPerInterval2, x_ic_vec[:, :, jj], rels, ed, K_c, EP, mu, r_shear, v, A0, rr0)
                         
            comm.send(X_snapshots_old_ct0[:, 0:(nt_c_max+1)], dest=0, tag=1001)
            comm.send(X_snapshots_old_ct0[:, -1], dest=0, tag=123)
    
            
        if rank == 1:
            if jj == ii + 1:
                ic_x = x_sol_fn[:, :, jj - 1]
            
            X_snapshots_g2_c = Sphere_FE2_Snapshots(dT2, CoarseNumTimeStepsPerInterval2, ic_x, rels, ed, K_c, EP, mu, r_shear, v, A0, rr0)
            
            comm.send(X_snapshots_g2_c[:, 0:(nt_c_max+1)], dest=0, tag=111)
            comm.send(X_snapshots_g2_c[:, -1], dest=0, tag=321)
            
            ic_x = comm.recv(source=0, tag=101)
            

        if rank == 0:

            if ii == 0:
                X_snapshots_old_ct0 = comm.recv(source=3, tag=1001)
                X_snapshots_old_c[:, :, jj] = X_snapshots_old_ct0
                X_end0[:, jj] = comm.recv(source=3, tag=123)

                if jj >= ntest:
                    X_snapshots_old = comm.recv(source=2, tag=101)
                    X_snapshots_old_mat[:, 0:(nt+1), jj] = X_snapshots_old
                
            else:
                if nt > nt_c0:
                    X_snapshots_old = comm.recv(source=2, tag=101)
                    X_snapshots_old_mat[:, (nt_c0):(nt+1), jj] = X_snapshots_old
                    
                            
            X_snapshots_g2_c = comm.recv(source=1, tag=111)
            X_end2[:, jj] = comm.recv(source=1, tag=321)
                
            X_snapshots_g2 = Sphere_FE2_Snapshots(dT, nt*r2, x_sol_fn[:, :, jj-1], rels, ed, K, ep, mu, r_shear, v, A0, rr0*r2)
            
            g2_snapshots_diff = X_snapshots_g2 - X_snapshots_old_mat[:, 0:(nt+1), jj]             
            g2_snapshots_diff_c = X_snapshots_g2_c[:, 0:(nt+1)] - X_snapshots_old_c[:, 0:(nt+1), jj]
            
            g2_diff_mat = g2_snapshots_diff - g2_snapshots_diff_c
            
            X_diff_g2_p = HODMD(d, g2_diff_mat, np.arange(qq1 + 1), varepsilon1, varepsilon)
            
            g2_X_diff_tend_c = X_end2[:, jj] - X_end0[:, jj]
            
            
            Xq = np.reshape(g2_X_diff_tend_c + X_diff_g2_p[:, -1], (3, M))
            
            x_sol_fn[:,:,jj] = x_sol_fn[:,:,jj] + Xq

            X_end0[:, jj] = X_end2[:, jj]
            
            X_snapshots_old_mat[:, 0:(nt+1), jj] = X_snapshots_g2
            X_snapshots_old_c[:, 0:(nt_c_max+1), jj] = X_snapshots_g2_c
        
            X_correct_ic[:, :, jj] = np.reshape(X_snapshots_g2[:, -1], (3, M))
            
            comm.send(x_sol_fn[:, :, jj], dest=1, tag=101)


    if rank == 0: 
     
        for kk in range(ii+1, NumTimeInterval):
            x_ic_vec[:, :, kk] = x_sol_fn[:, :, kk-1]


        send_buf_x = np.split(x_ic_vec, NumTimeInterval, axis = 2)
    
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


    if rank == 0 and ii < NumParSweeps - 1 and nt < nt_vec[ii + 1]:
        comm.send(X_correct_ic, dest=2, tag=1101)


    if rank == 2 and ii < NumParSweeps - 1 and nt < nt_vec[ii + 1]:
        X_correct_ic = comm.recv(source=0, tag=1101)

    
    comm.Barrier()

    
end_time = MPI.Wtime()
if rank == 0:
    run_time = end_time - start_time
    print("Number of iterations performed = %d\t: Total time = %.16g\n" %(ii, run_time))




    

    



        
