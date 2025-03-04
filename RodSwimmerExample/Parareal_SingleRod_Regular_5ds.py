import math
import copy
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy import linalg
from mpi4py import MPI
from scipy.linalg import sqrtm
import sys

#This code calculates the example of a single rod-like swimmer with epsilon = 5*sigma using the classic Parareal algorithm.
#The code currently uses the same spatial grid in both the coarse and fine solver. The coarse solver uses a larger time step.
#The code for Method of Regularized Stokeslets is adapted from the code provided by Minghao W. Rostami (SUNY Binghamton).

def VelocitiesRegSto7halfRodWall(f, n, ep, Xb, Xeval, Qweights, SourcedA):
    # This function computes the velocities by the Method of Regularized
    # Stokeslets (force and torque with wall). The 7/2 blob function is used.
    # The wall is located at z=0.

    # Inputs:
    # Xb: a 3 by nsrc matrix. It holds the positions of the points forces.
    #     nsrc is the number of point forces.
    # f: a 3 by nsrc matrix. It holds the points forces. The ith column is the
    #    forces exerted by the ith point.
    # n: a 3 by nsrc matrix. It holds the points torques. The ith column is the
    #    forces exerted by the ith point.
    # Xeval: a 3 by neval matrix. It holds the positions of the target points
    #        (that is, the points whose velocties we want to compute). neval
    #        is the number of target points.
    # SourcedA: This is currently set to 1 for exact calculation of the matrix-vector product.
    # This setting can be adapted to Fast Multipole Method (FMM).
    # Qweights: This is currently set to 1 for exact calculation of the matrix-vector product.
    # This setting can be adapted to Fast Multipole Method (FMM).
    # ep: the regularization parameter epsilon in the method of regularized
    #     Stokeslets
    # Outputs:
    # U: a 3*neval matrix. It holds the velocties at the points in Xeval.
    # W: a 3*neval matrix. It holds the angular velocties at the points in Xeval.

    wa = 0 # wall location is z=wa

    if np.any(np.where(Xb[2,:]<wa)) | np.any(np.where(Xeval[2,:]<wa)):
        print('There is a source or target point not in the fluid domain!', file=sys.stderr)


    sz = Xeval.shape
    neval = sz[1]

    sz = Xb.shape
    nsrc = sz[1]

    W = np.zeros((3, neval))
    U = np.zeros((3, neval))

    eps0 = ep**2
    c1 = SourcedA/(8*np.pi) 
    c2 = SourcedA/(4*np.pi)

    hvec = Xb[2,:]-wa 
    # the z coordinates of the sources points.

    hsvec = hvec*hvec
    dhvec = 2*hvec
    Xbim3 = wa-hvec
    # z coordinates of the image points of the sources points
    ff = np.zeros(f.shape)
    ff[0, :] = f[0, :]*Qweights
    ff[1, :] = f[1, :]*Qweights
    ff[2, :] = f[2, :]*Qweights

    nn = np.zeros(n.shape)
    nn[0, :] = n[0, :]*Qweights
    nn[1, :] = n[1, :]*Qweights
    nn[2, :] = n[2, :]*Qweights

    # loop over the target points

    for i in range(neval):
        # calculate the radially symmetric functions and their scaled
        # versions
        xeval = Xeval[:, i]
        dxmat = np.zeros((3, nsrc))
        w = np.zeros((3, 1))
        u = np.zeros((3, 1))
        dxmat[0, :] = xeval[0]-Xb[0, :]
        dxmat[1, :] = xeval[1]-Xb[1, :]
        dxmat[2, :] = xeval[2]-Xb[2, :]

        rs12vec = np.sum((dxmat[0:2, :])**2,axis=0)

        rsvec = rs12vec + (dxmat[2, :])**2
        epsprsvec = eps0 + rsvec
        epsprsvec3 = epsprsvec**(3./2) 
        epsprsvec5 = epsprsvec3*epsprsvec
    
        hlfq1vec = (c1/2)*((5*eps0+2*rsvec)/epsprsvec5)
        hlfq2vec = (c2/2)/epsprsvec3
        qtrd2vec = -(1./4*3*c2)/epsprsvec5
        imdxmat3 = xeval[2]-Xbim3
        imrsvec = rs12vec + imdxmat3**2 
        imepsprsvec = eps0 + imrsvec  
        imepsprsvec3 = imepsprsvec**(3./2)
        imepsprsvec5 = imepsprsvec3*imepsprsvec
        imepsprsvec7 = imepsprsvec5*imepsprsvec
        imhlfq1vec = (c1/2)*((5*eps0+2*imrsvec)/imepsprsvec5)
        imhlfCURL = (15./2*eps0*c2)/imepsprsvec7
        imhlfhCURL = hvec*imhlfCURL
        imhlfhsCURL = hsvec*imhlfCURL
        imD1vec = c1*((-10*eps0**2+7*eps0*imrsvec+2*imrsvec**2)/imepsprsvec7)
        imD2vec = c1*((-21*eps0-6*imrsvec)/imepsprsvec7)
        imhlfD2vec = 1./2*imD2vec
        imhD2vec = hvec*imD2vec
        imm3dvimepsprsvec5 = -3/imepsprsvec5
        imh2vec = c1/imepsprsvec3
        imdh2dvrvec = c1*imm3dvimepsprsvec5
        imdh1dvrvec = (imdh2dvrvec/3)*(4*eps0+imrsvec)
        imd2vec = c2*imm3dvimepsprsvec5
        imhd2vec = hvec*imd2vec
        imhsd2vec = hsvec*imd2vec
        imd1vec = (imd2vec/3)*(2*eps0-imrsvec)
        imhd1vec = hvec*imd1vec
        imhlfq2vec = (c2/2)/imepsprsvec3
        im2hh2vec = dhvec*imh2vec
        
        #calculate the combinations of these functions
        imdh1dvr_h2vec = imdh1dvrvec + imh2vec
        im2hdh1dvr_h2vec = dhvec*imdh1dvr_h2vec
        imdh1dvr_h2vec_imhlfq2vec = imdh1dvr_h2vec + imhlfq2vec
        h1vec_imh1vec = (c1*((2*eps0+rsvec)/epsprsvec3)) - (2*eps0+imrsvec)*imh2vec
        imhlfq1vec_imhlfhsCURL = imhlfq1vec + imhlfhsCURL
        
        # calculate the combinations of forces, torques, and position vectors
        f12_dxvec = (ff[0,:])*dxmat[0,:] + (ff[1,:])*dxmat[1,:]
        n12_dxvec = (nn[0,:])*dxmat[0,:] + (nn[1,:])*dxmat[1,:] 
        f3_imdxmat3 = (ff[2,:])*imdxmat3
        g_imdxvec = -f12_dxvec + f3_imdxmat3
        q_imdxvec = (nn[1,:])*dxmat[0,:] - (nn[0,:])*dxmat[1,:]
        
        # calculate the combinations of radially symmetric functions,
        # forces, torques, and position vectors
        imdh2dvrvec_imdxmat3 = imdh2dvrvec*imdxmat3
        imhlfD2vec_imdxmat3 = imhlfD2vec*imdxmat3
        imdxmat3_im2hh2vec_imhsd1vec = imdxmat3*im2hh2vec - (imhsd2vec/3)*(2*eps0-imrsvec)
        f_dxvec_h2vec = (f12_dxvec + (ff[2,:])*dxmat[2,:])*(c1/epsprsvec3)
        f1_hlfq1vec = (ff[0,:])*hlfq1vec
        f2_hlfq1vec = (ff[1,:])*hlfq1vec
    
        n1_hlfq2vec = (nn[0,:])*hlfq2vec
        n2_hlfq2vec = (nn[1,:])*hlfq2vec

        n12_dxvec_imhlfD2vec = n12_dxvec*imhlfD2vec
        n_dxvec_qtrd2vec = (n12_dxvec+(nn[2,:])*dxmat[2,:])*qtrd2vec
        n1_imdh1dvr_h2vec_imhlfq2vec = (nn[0,:])*imdh1dvr_h2vec_imhlfq2vec 
        
        # calculate coefficients for the terms in the linear and angular
        # velocities
        u12f12coef = h1vec_imh1vec - imdxmat3_im2hh2vec_imhsd1vec
        u12y21coef = (nn[2,:])*(hlfq2vec-imhlfq2vec)
        u12n21coef = imhd1vec - imdh1dvr_h2vec*imdxmat3
        u1yim3coef = (ff[0,:])*im2hdh1dvr_h2vec + (nn[1,:])*imdh1dvr_h2vec_imhlfq2vec
        u2yim3coef = (ff[1,:])*im2hdh1dvr_h2vec - n1_imdh1dvr_h2vec_imhlfq2vec
        u3yim3coef = -(f12_dxvec + f3_imdxmat3)*imh2vec + ff[2,:]*im2hh2vec\
                     + g_imdxvec*(imdh2dvrvec_imdxmat3*dhvec - imhsd2vec)\
                     + q_imdxvec*(imhd2vec-2*imdh2dvrvec_imdxmat3)
        u12y12coef = u3yim3coef + f_dxvec_h2vec
        w3n3coef = 1/4*imd1vec-(qtrd2vec/3)*(2*eps0-rsvec)
        w12f21coef = dhvec*imhlfq1vec + hvec*imD1vec - imhd1vec
        w3yim3coef1 = ((ff[1,:])*dxmat[0,:]-(ff[0,:])*dxmat[1,:])*(imhD2vec-imhd2vec)\
                      + (n12_dxvec + (nn[2,:])*imdxmat3)*(1/4*(imd2vec))\
                      + 1/2*n12_dxvec*imd2vec - n12_dxvec_imhlfD2vec
        w12y12coef = w3yim3coef1 - n_dxvec_qtrd2vec
        w12y21coef = (ff[2,:])*(hlfq1vec-imhlfq1vec+imhlfhsCURL)\
                     + g_imdxvec*imhD2vec - q_imdxvec*imhlfD2vec
        w1yim3coef1 = (ff[1,:])*imhlfq1vec_imhlfhsCURL - (nn[0,:])*imhlfhCURL
        w2yim3coef1 = (ff[0,:])*imhlfq1vec_imhlfhsCURL + (nn[1,:])*imhlfhCURL
        w12n12coef = w3n3coef + 1/2*(imd1vec-imD1vec)
        
        #calculate the linear and angular velocities
        u[0] = np.matmul(ff[0,:],np.transpose(u12f12coef)) + np.matmul(nn[1,:],np.transpose(u12n21coef))\
               + np.matmul(dxmat[0,:],np.transpose(u12y12coef)) - np.matmul(dxmat[1,:],np.transpose(u12y21coef))\
               + np.matmul(dxmat[2,:],np.transpose(n2_hlfq2vec)) - np.matmul(imdxmat3,np.transpose(u1yim3coef))
    
        u[1] = np.matmul(ff[1,:],np.transpose(u12f12coef)) - np.matmul(nn[0,:],np.transpose(u12n21coef))\
               + np.matmul(dxmat[0,:],np.transpose(u12y21coef)) + np.matmul(dxmat[1,:],np.transpose(u12y12coef))\
               - np.matmul(dxmat[2,:],np.transpose(n1_hlfq2vec)) - np.matmul(imdxmat3,np.transpose(u2yim3coef))
      
        u[2] = np.matmul(ff[2,:],np.transpose(h1vec_imh1vec+imdxmat3_im2hh2vec_imhsd1vec))+ np.matmul(g_imdxvec,np.transpose(dhvec*imdh1dvrvec)) - np.matmul(q_imdxvec,np.transpose(imdh1dvr_h2vec))\
               + np.matmul(dxmat[0,:],np.transpose(u1yim3coef-n2_hlfq2vec)) + np.matmul(dxmat[1,:],np.transpose(u2yim3coef+n1_hlfq2vec))\
               + np.matmul(dxmat[2,:],np.transpose(f_dxvec_h2vec)) + np.matmul(imdxmat3,np.transpose(u3yim3coef))    
    
        w[0] = np.matmul(ff[1,:],np.transpose(w12f21coef)) + np.matmul(nn[0,:],np.transpose(w12n12coef))\
               + np.matmul(dxmat[0,:],np.transpose(w12y12coef)) - np.matmul(dxmat[1,:],np.transpose(w12y21coef))\
               + np.matmul(dxmat[2,:],np.transpose(f2_hlfq1vec)) - np.matmul(imdxmat3,np.transpose(w1yim3coef1-(nn[0,:])*imhlfD2vec_imdxmat3))
    
        w[1] = np.matmul(-ff[0,:],np.transpose(w12f21coef)) + np.matmul(nn[1,:],np.transpose(w12n12coef))\
               + np.matmul(dxmat[0,:],np.transpose(w12y21coef)) + np.matmul(dxmat[1,:],np.transpose(w12y12coef))\
               - np.matmul(dxmat[2,:],np.transpose(f1_hlfq1vec)) + np.matmul(imdxmat3,np.transpose(w2yim3coef1+(nn[1,:])*imhlfD2vec_imdxmat3))
      
        w[2] = np.matmul(nn[2,:],np.transpose(w3n3coef))\
               + np.matmul(dxmat[0,:],np.transpose(-f2_hlfq1vec+w1yim3coef1)) + np.matmul(dxmat[1,:],np.transpose(f1_hlfq1vec-w2yim3coef1))\
               - np.matmul(dxmat[2,:],np.transpose(n_dxvec_qtrd2vec)) + np.matmul(imdxmat3,np.transpose(w3yim3coef1-n12_dxvec_imhlfD2vec))
    
        U[:,[i]] = u
        W[:,[i]] = w
    

    return U, W



def Krod_Swave_Wall_FE2(dt,maxt,GridParam,RodsParam,SwaveParam,CurrentTime,ep,ViscosityNu):
    #This calculates the position of the rods over maxt time steps using the forward Euler's method

    # extract the parameters from the parameter structures

    M = GridParam.M
    ds = GridParam.ds
    X = GridParam.X
    D1 = GridParam.D1
    D2 = GridParam.D2
    D3 = GridParam.D3
    RodDisPtsVec = GridParam.RodDisPtsVec

    Xr = copy.deepcopy(X)
    D1_r = copy.deepcopy(D1)
    D2_r = copy.deepcopy(D2)
    D3_r = copy.deepcopy(D3)

    # rod parameters
    NumRods = RodsParam.NumRods
    avec = RodsParam.avec
    bvec = RodsParam.bvec

    a1 = avec[0]
    a2 = avec[1] 
    a3 = avec[2]

    b1 = bvec[0] 
    b2 = bvec[1] 
    b3 = bvec[2]

    # Sinusoidal wave parameters
    OmegaFactor = SwaveParam.OmegaFactor
    FrequencySigma = SwaveParam.FrequencySigma
    NumPeriods = SwaveParam.NumPeriods

    NumPts = M*NumRods
    RodLengthVec = np.zeros((maxt, NumRods))
    locmf = np.zeros((3, M))
    locmn = np.zeros((3, M))
    
    Fh = np.zeros((3, M - 1))
    Nh = np.zeros((3, M - 1))

    D1h = np.zeros((3, M - 1))
    D2h = np.zeros((3, M - 1))
    D3h = np.zeros((3, M - 1))
    
    mf = np.zeros((3, NumPts))
    mn = np.zeros((3, NumPts))
    
    locX = np.zeros((3, M))
    locD1 = np.zeros((3, M))
    locD2 = np.zeros((3, M))
    locD3 = np.zeros((3, M))

    ## loop over maxt time steps
    for i in range(1, maxt + 1):
        # Compute forces and torques
        mf[:, :] = 0
        mn[:, :] = 0
        # loop over the rods
        for j in range(NumRods):
            locmf[:,:] = 0
            locmn[:,:] = 0
            StartPt = j*M
            EndPt = (j+1)*M
        
            locX[:, :] = Xr[:, StartPt:EndPt]
            locD1[:, :] = D1_r[:, StartPt:EndPt]
            locD2[:, :] = D2_r[:, StartPt:EndPt]
            locD3[:, :] = D3_r[:, StartPt:EndPt]

            Fh[:,:] = 0
            Nh[:,:] = 0

            D1h[:,:] = 0
            D2h[:,:] = 0
            D3h[:,:] = 0
            
            diffX = np.diff(locX, 1, 1)
            diffD1 = np.diff(locD1, 1, 1)
            diffD2 = np.diff(locD2, 1, 1)
            diffD3 = np.diff(locD3, 1, 1)
        
            for k in range(M-1): 
            #iteration over spacial discretization of the rod

                A = np.outer(locD1[:, k+1], np.transpose(locD1[:, k])) + np.outer(locD2[:, k+1], np.transpose(locD2[:, k])) \
                    + np.outer(locD3[:, k+1], np.transpose(locD3[:, k]))
            
            #rotation matrix from point k to k+0.5
                roma = sqrtm(A)

                D1h[:, k] = np.matmul(roma, locD1[:, k])
                D2h[:, k] = np.matmul(roma, locD2[:, k])
                D3h[:, k] = np.matmul(roma, locD3[:, k])

            Omega = OmegaFactor * np.sin(1. / NumPeriods * (RodDisPtsVec + 0.5 * ds) - FrequencySigma * ((i - 1) * dt + CurrentTime))
            f1h = ((D1h[0,:]) * diffX[0,:] + (D1h[1,:]) * diffX[1,:] + (D1h[2,:]) * diffX[2,:])*(b1 / ds)
            f2h = ((D2h[0,:]) * diffX[0,:] + (D2h[1,:]) * diffX[1,:] + (D2h[2,:]) * diffX[2,:])*(b2 / ds)
            f3h = (((D3h[0,:]) * diffX[0,:] + (D3h[1,:]) * diffX[1,:] + (D3h[2,:]) * diffX[2,:]) / ds - 1)*b3
            n1h = (((D3h[0,:]) * diffD2[0,:] + (D3h[1,:]) * diffD2[1,:] + (D3h[2,:]) * diffD2[2,:]) / ds)*a1
            n2h = (((D1h[0,:]) * diffD3[0,:] + (D1h[1,:]) * diffD3[1,:] + (D1h[2,:]) * diffD3[2,:]) / ds - Omega[0:(M-1)])*a2
            n3h = (((D2h[0,:]) * diffD1[0,:] + (D2h[1,:]) * diffD1[1,:] + (D2h[2,:]) * diffD1[2,:]) / ds)*a3

            Fh[0,:] = f1h * D1h[0,:] + f2h * D2h[0,:] + f3h * D3h[0,:]
            Fh[1,:] = f1h * D1h[1,:] + f2h * D2h[1,:] + f3h * D3h[1,:]
            Fh[2,:] = f1h * D1h[2,:] + f2h * D2h[2,:] + f3h * D3h[2,:]

            Nh[0,:] = n1h * D1h[0,:] + n2h * D2h[0,:] + n3h * D3h[0,:]
            Nh[1,:] = n1h * D1h[1,:] + n2h * D2h[1,:] + n3h * D3h[1,:]
            Nh[2,:] = n1h * D1h[2,:] + n2h * D2h[2,:] + n3h * D3h[2,:]

            locmf[:, 0] = Fh[:, 0]
            locmf[:, 1:(M-1)] = np.diff(Fh, 1, 1)
            locmf[:, M-1] = -Fh[:, M - 2]
        
            diffNh = np.diff(Nh, 1, 1)

            locmn[:, 0] = Nh[:, 0] + 0.5 * np.cross(diffX[:, 0], Fh[:, 0])
            locmn[0, 1:(M - 1)] =  diffNh[0,:] + (- (diffX[2, 1:(M-1)]) * Fh[1, 1:(M - 1)] + (diffX[1, 1:(M-1)])*Fh[2, 1:(M - 1)] \
                            - (diffX[2, 0:(M-2)])*Fh[1, 0:(M - 2)] + (diffX[1, 0:(M-2)]) * Fh[2, 0:(M - 2)]) * 0.5
            locmn[1, 1:(M - 1)] =  diffNh[1,:] + (- (diffX[0, 1:(M-1)]) * Fh[2, 1:(M - 1)] + (diffX[2, 1:(M-1)])*Fh[0, 1:(M - 1)]\
                                         - (diffX[0, 0:(M-2)]) * Fh[2, 0:(M - 2)] + (diffX[2, 0:(M-2)])*Fh[0, 0:(M - 2)])*0.5
            locmn[2, 1:(M - 1)] =  diffNh[2,:] + (- (diffX[1, 1:(M-1)]) * Fh[0, 1:(M - 1)] + (diffX[0, 1:(M-1)])*Fh[1, 1:(M - 1)]\
                                         - (diffX[1, 0:(M-2)]) * Fh[0, 0:(M - 2)] + (diffX[0, 0:(M-2)])*Fh[1, 0:(M - 2)])*0.5
            locmn[:, M-1] = -Nh[:, M - 2] + 0.5 * np.cross(diffX[:, M - 2], Fh[:, M - 2])

            mf[:, StartPt: EndPt] = locmf
            mn[:, StartPt: EndPt] = locmn
        
            # Compute linear and angular velocities

        U, W = VelocitiesRegSto7halfRodWall(mf, mn, ep, Xr, Xr, np.ones(NumPts), 1)
        U = U / ViscosityNu
        W = W / ViscosityNu

        Xr = Xr + U * dt

        #Update triads
        for k in range(NumPts):
            w = np.linalg.norm(W[:, k])
            roax = W[:, k] / w
            theta = w * dt
            roax_t = roax.transpose()

            D1_r[:, k] = np.cos(theta) * D1_r[:, k] + (1 - np.cos(theta)) * roax * (np.dot(roax_t, D1_r[:, k])) + np.sin(theta) * np.cross(roax, D1_r[:, k])
            D2_r[:, k] = np.cos(theta) * D2_r[:, k] + (1 - np.cos(theta)) * roax * (np.dot(roax_t, D2_r[:, k])) + np.sin(theta) * np.cross(roax, D2_r[:, k])
            D3_r[:, k] = np.cos(theta) * D3_r[:, k] + (1 - np.cos(theta)) * roax * (np.dot(roax_t, D3_r[:, k])) + np.sin(theta) * np.cross(roax, D3_r[:, k])

        #Calculate the rod length
        for j in range(1,NumRods+1):
            StartPt = (j - 1) * M
            EndPt = j * M
            RodLengthVec[i-1, j-1] = sum(np.sqrt(sum((np.diff(Xr[:, StartPt:EndPt], 1, 1))**2)))


    return Xr, D1_r, D2_r, D3_r, RodLengthVec



def Krod_Swave_Wall_RK2_2(dt,maxt,GridParam,RodsParam,SwaveParam,CurrentTime,ep,ViscosityNu):
    # Thic code calculates the position of rods over maxt time steps using Runge-Kutta second-order method (RK2)

    M = GridParam.M
    ds = GridParam.ds
    X = GridParam.X
    D1 = GridParam.D1
    D2 = GridParam.D2
    D3 = GridParam.D3
    RodDisPtsVec = GridParam.RodDisPtsVec
    
    Xr = copy.deepcopy(X)
    D1_r = copy.deepcopy(D1)
    D2_r = copy.deepcopy(D2)
    D3_r = copy.deepcopy(D3) 

    # rod parameters
    NumRods = RodsParam.NumRods
    avec = RodsParam.avec
    bvec = RodsParam.bvec

    a1 = avec[0]
    a2 = avec[1] 
    a3 = avec[2]

    b1 = bvec[0] 
    b2 = bvec[1] 
    b3 = bvec[2]

    # Sinusoidal wave parameters
    OmegaFactor = SwaveParam.OmegaFactor
    FrequencySigma = SwaveParam.FrequencySigma
    NumPeriods = SwaveParam.NumPeriods

    NumPts = M*NumRods
    RodLengthVec = np.zeros((maxt, NumRods))

    D1_FE = np.zeros_like(D1)
    D2_FE = np.zeros_like(D2)
    D3_FE = np.zeros_like(D3)
    
    locmf = np.zeros((3, M))
    locmn = np.zeros((3, M))
    
    Fh = np.zeros((3, M - 1))
    Nh = np.zeros((3, M - 1))

    D1h = np.zeros((3, M - 1))
    D2h = np.zeros((3, M - 1))
    D3h = np.zeros((3, M - 1))
    
    mf = np.zeros((3, NumPts))
    mn = np.zeros((3, NumPts))
    
    locX = np.zeros((3, M))
    locD1 = np.zeros((3, M))
    locD2 = np.zeros((3, M))
    locD3 = np.zeros((3, M))


    ## loop over maxt time steps
    for i in range(1, maxt + 1):
        # Compute forces and torques
        # initialization
        mf[:, :] = 0
        mn[:, :] = 0
        # loop over the rods
        for j in range(NumRods):
            locmf[:, :] = 0
            locmn[:, :] = 0
        
            StartPt = j*M
            EndPt = (j+1)*M

            locX[:, :] = Xr[:, StartPt:EndPt]
            locD1[:, :] = D1_r[:, StartPt:EndPt]
            locD2[:, :] = D2_r[:, StartPt:EndPt]
            locD3[:, :] = D3_r[:, StartPt:EndPt]

            Fh[:, :] = 0
            Nh[:, :] = 0

            D1h[:, :] = 0
            D2h[:, :] = 0
            D3h[:, :] = 0
        
            diffX = np.diff(locX, 1, 1)
            diffD1 = np.diff(locD1, 1, 1)
            diffD2 = np.diff(locD2, 1, 1)
            diffD3 = np.diff(locD3, 1, 1)
        
            for k in range(M-1): 
            #iteration over spacial discretization of the rod
            

                A = np.outer(locD1[:, k + 1], np.transpose(locD1[:, k])) + np.outer(locD2[:, k + 1], np.transpose(locD2[:, k])) \
                    + np.outer(locD3[:, k + 1], np.transpose(locD3[:, k]))
            
            #rotation matrix from point k to k+0.5
                roma = sqrtm(A)

                D1h[:, k] = np.matmul(roma, locD1[:, k])
                D2h[:, k] = np.matmul(roma, locD2[:, k])
                D3h[:, k] = np.matmul(roma, locD3[:, k])

            Omega = OmegaFactor * np.sin(1. / NumPeriods * (RodDisPtsVec + 0.5 * ds) - FrequencySigma * ((i - 1) * dt + CurrentTime))
            f1h = ((D1h[0,:]) * diffX[0,:] + (D1h[1,:]) * diffX[1,:] + (D1h[2,:]) * diffX[2,:])*(b1 / ds)
            f2h = ((D2h[0,:]) * diffX[0,:] + (D2h[1,:]) * diffX[1,:] + (D2h[2,:]) * diffX[2,:])*(b2 / ds)
            f3h = (((D3h[0,:]) * diffX[0,:] + (D3h[1,:]) * diffX[1,:] + (D3h[2,:]) * diffX[2,:]) / ds - 1)*b3
            n1h = (((D3h[0,:]) * diffD2[0,:] + (D3h[1,:]) * diffD2[1,:] + (D3h[2,:]) * diffD2[2,:]) / ds)*a1
            n2h = (((D1h[0,:]) * diffD3[0,:] + (D1h[1,:]) * diffD3[1,:] + (D1h[2,:]) * diffD3[2,:]) / ds - Omega[0:(M-1)])*a2
            n3h = (((D2h[0,:]) * diffD1[0,:] + (D2h[1,:]) * diffD1[1,:] + (D2h[2,:]) * diffD1[2,:]) / ds)*a3



            Fh[0,:] = f1h * D1h[0,:] + f2h * D2h[0,:] + f3h * D3h[0,:]
            Fh[1,:] = f1h * D1h[1,:] + f2h * D2h[1,:] + f3h * D3h[1,:]
            Fh[2,:] = f1h * D1h[2,:] + f2h * D2h[2,:] + f3h * D3h[2,:]

            Nh[0,:] = n1h * D1h[0,:] + n2h * D2h[0,:] + n3h * D3h[0,:]
            Nh[1,:] = n1h * D1h[1,:] + n2h * D2h[1,:] + n3h * D3h[1,:]
            Nh[2,:] = n1h * D1h[2,:] + n2h * D2h[2,:] + n3h * D3h[2,:]

            locmf[:, 0] = Fh[:, 0]
            locmf[:, 1:(M-1)] = np.diff(Fh, 1, 1)
            locmf[:, M-1] = -Fh[:, M - 2]
        
            diffNh = np.diff(Nh, 1, 1)

            locmn[:, 0] = Nh[:, 0] + 0.5 * np.cross(diffX[:, 0], Fh[:, 0])
            locmn[0, 1:(M - 1)] = diffNh[0,:] + (- (diffX[2, 1:(M-1)]) * Fh[1, 1:(M - 1)] + (diffX[1, 1:(M-1)])*Fh[2, 1:(M - 1)]\
                          - (diffX[2, 0:(M-2)])*Fh[1, 0:(M - 2)] + (diffX[1, 0:(M-2)]) * Fh[2, 0:(M - 2)]) * 0.5
            locmn[1, 1:(M - 1)] = diffNh[1,:] + (- (diffX[0, 1:(M-1)]) * Fh[2, 1:(M - 1)] + (diffX[2, 1:(M-1)])*Fh[0, 1:(M - 1)]\
                                         - (diffX[0, 0:(M-2)]) * Fh[2, 0:(M - 2)] + (diffX[2, 0:(M-2)])*Fh[0, 0:(M - 2)])*0.5
            locmn[2, 1:(M - 1)] = diffNh[2,:] + (- (diffX[1, 1:(M-1)]) * Fh[0, 1:(M - 1)] + (diffX[0, 1:(M-1)])*Fh[1, 1:(M - 1)]\
                                         - (diffX[1, 0:(M-2)]) * Fh[0, 0:(M - 2)] + (diffX[0, 0:(M-2)])*Fh[1, 0:(M - 2)])*0.5
            locmn[:, M-1] = -Nh[:, M - 2] + 0.5 * np.cross(diffX[:, M - 2], Fh[:, M - 2])

            mf[:, StartPt: EndPt] = locmf
            mn[:, StartPt: EndPt] = locmn

        
            # Compute linear and angular velocities

        U_FE, W_FE = VelocitiesRegSto7halfRodWall(mf, mn, ep, Xr, Xr, np.ones(NumPts), 1)
        U_FE = U_FE / ViscosityNu
        W_FE = W_FE / ViscosityNu

        dX_FE = U_FE * dt
        X_FE = Xr + dX_FE

        #Update triads
        for k in range(NumPts):
            w = np.linalg.norm(W_FE[:, k])
            roax = W_FE[:, k] / w
            theta = w * dt
            roax_t = roax.transpose()

            D1_FE[:, k] = np.cos(theta) * D1_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D1_r[:, k])) + np.sin(theta) * np.cross(roax, D1_r[:, k])
            D2_FE[:, k] = np.cos(theta) * D2_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D2_r[:, k])) + np.sin(theta) * np.cross(roax, D2_r[:, k])
            D3_FE[:, k] = np.cos(theta) * D3_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D3_r[:, k])) + np.sin(theta) * np.cross(roax, D3_r[:, k])



        #initialization
        mf[:, :] = 0
        mn[:, :] = 0


        for j in range(NumRods):
            locmf[:, :] = 0
            locmn[:, :] = 0
        
            StartPt = j*M
            EndPt = (j+1)*M
        
            locX[:, :] = X_FE[:,StartPt:EndPt]
            locD1[:, :] = D1_FE[:,StartPt:EndPt]
            locD2[:, :] = D2_FE[:,StartPt:EndPt]
            locD3[:, :] = D3_FE[:,StartPt:EndPt]

            Fh[:, :] = 0
            Nh[:, :] = 0

            D1h[:, :] = 0
            D2h[:, :] = 0
            D3h[:, :] = 0
        
            diffX = np.diff(locX, 1, 1)
            diffD1 = np.diff(locD1, 1, 1)
            diffD2 = np.diff(locD2, 1, 1)
            diffD3 = np.diff(locD3, 1, 1)
        
            for k in range(M-1): 
            #iteration over spacial discretization of the rod
            
                A = np.outer(locD1[:, k + 1], np.transpose(locD1[:, k])) + np.outer(locD2[:, k + 1], np.transpose(locD2[:, k])) \
                    + np.outer(locD3[:, k + 1], np.transpose(locD3[:, k]))
            
            #rotation matrix from point k to k+0.5
                roma = sqrtm(A)

                D1h[:, k] = np.matmul(roma, locD1[:, k])
                D2h[:, k] = np.matmul(roma, locD2[:, k])
                D3h[:, k] = np.matmul(roma, locD3[:, k])

            Omega = OmegaFactor * np.sin(1. / NumPeriods * (RodDisPtsVec + 0.5 * ds) - FrequencySigma * (i * dt + CurrentTime))
            f1h = ((D1h[0,:]) * diffX[0,:] + (D1h[1,:]) * diffX[1,:] + (D1h[2,:]) * diffX[2,:])*(b1 / ds)
            f2h = ((D2h[0,:]) * diffX[0,:] + (D2h[1,:]) * diffX[1,:] + (D2h[2,:]) * diffX[2,:])*(b2 / ds)
            f3h = (((D3h[0,:]) * diffX[0,:] + (D3h[1,:]) * diffX[1,:] + (D3h[2,:]) * diffX[2,:]) / ds - 1)*b3
            n1h = (((D3h[0,:]) * diffD2[0,:] + (D3h[1,:]) * diffD2[1,:] + (D3h[2,:]) * diffD2[2,:]) / ds)*a1
            n2h = (((D1h[0,:]) * diffD3[0,:] + (D1h[1,:]) * diffD3[1,:] + (D1h[2,:]) * diffD3[2,:]) / ds - Omega[0:(M-1)])*a2
            n3h = (((D2h[0,:]) * diffD1[0,:] + (D2h[1,:]) * diffD1[1,:] + (D2h[2,:]) * diffD1[2,:]) / ds)*a3


            Fh[0,:] = f1h * D1h[0,:] + f2h * D2h[0,:] + f3h * D3h[0,:]
            Fh[1,:] = f1h * D1h[1,:] + f2h * D2h[1,:] + f3h * D3h[1,:]
            Fh[2,:] = f1h * D1h[2,:] + f2h * D2h[2,:] + f3h * D3h[2,:]

            Nh[0,:] = n1h * D1h[0,:] + n2h * D2h[0,:] + n3h * D3h[0,:]
            Nh[1,:] = n1h * D1h[1,:] + n2h * D2h[1,:] + n3h * D3h[1,:]
            Nh[2,:] = n1h * D1h[2,:] + n2h * D2h[2,:] + n3h * D3h[2,:]

            locmf[:, 0] = Fh[:, 0]
            locmf[:, 1:(M-1)] = np.diff(Fh, 1, 1)
            locmf[:, M-1] = -Fh[:, M - 2]
        
            diffNh = np.diff(Nh, 1, 1)

            locmn[:, 0] = Nh[:, 0] + 0.5 * np.cross(diffX[:, 0], Fh[:, 0])
            locmn[0, 1:(M - 1)] = diffNh[0, :] + (- (diffX[2, 1:(M-1)]) * Fh[1, 1:(M - 1)] + (diffX[1, 1:(M-1)])*Fh[2, 1:(M - 1)]\
                          - (diffX[2, 0:(M-2)])*Fh[1, 0:(M - 2)] + (diffX[1, 0:(M-2)]) * Fh[2, 0:(M - 2)]) * 0.5
            locmn[1, 1:(M - 1)] = diffNh[1, :] + (- (diffX[0, 1:(M-1)]) * Fh[2, 1:(M - 1)] + (diffX[2, 1:(M-1)])*Fh[0, 1:(M - 1)]\
                                         - (diffX[0, 0:(M-2)]) * Fh[2, 0:(M - 2)] + (diffX[2, 0:(M-2)])*Fh[0, 0:(M - 2)])*0.5
            locmn[2, 1:(M - 1)] = diffNh[2, :] + (- (diffX[1, 1:(M-1)]) * Fh[0, 1:(M - 1)] + (diffX[0, 1:(M-1)])*Fh[1, 1:(M - 1)]\
                                         - (diffX[1, 0:(M-2)]) * Fh[0, 0:(M - 2)] + (diffX[0, 0:(M-2)])*Fh[1, 0:(M - 2)])*0.5
            locmn[:, M-1] = -Nh[:, M - 2] + 0.5 * np.cross(diffX[:, M - 2], Fh[:, M - 2])

            mf[:, StartPt: EndPt] = locmf
            mn[:, StartPt: EndPt] = locmn


        # Compute linear and angular velocities

        U, W = VelocitiesRegSto7halfRodWall(mf, mn, ep, X_FE, X_FE, np.ones(NumPts), 1)
        U = U / ViscosityNu
        W = W / ViscosityNu

        Xr = Xr + 0.5*(dX_FE+U*dt)

        #Update triads
        for k in range(NumPts):
            avgW = 0.5*(W_FE[:, k]+W[:, k])
            w = np.linalg.norm(avgW)
            roax = avgW/w
            theta = w * dt
            roax_t = roax.transpose()

            D1_r[:, k] = np.cos(theta) * D1_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D1_r[:, k])) + np.sin(theta)*np.cross(roax, D1_r[:, k])
            D2_r[:, k] = np.cos(theta) * D2_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D2_r[:, k])) + np.sin(theta)*np.cross(roax, D2_r[:, k])
            D3_r[:, k] = np.cos(theta) * D3_r[:, k] + (1 - np.cos(theta)) * roax * (np.matmul(roax_t, D3_r[:, k])) + np.sin(theta)*np.cross(roax, D3_r[:, k])

        #Calculate the rod length
        for j in range(1, NumRods+1):
            StartPt = (j - 1) * M
            EndPt = j * M
            RodLengthVec[i-1, j-1] = np.sum(np.sqrt(np.sum((np.diff(Xr[:, StartPt:EndPt], 1, 1))**2, axis=0)), axis=0)

    
    return Xr, D1_r, D2_r, D3_r, RodLengthVec




def ExtractCoarseGrid2(Xf,Df1,Df2,Df3,RodDisCoarsePtsVec,RodDisFinePtsVec):
    #Extract coarse grid from the fine grid using cubic spline

    m = np.size(RodDisCoarsePtsVec)
    Xc = np.zeros((3, m))
    Dc1 = np.zeros((3, m))
    Dc2 = np.zeros((3, m))
    Dc3 = np.zeros((3, m))

    for i in range(3):
        cs_x = CubicSpline(RodDisFinePtsVec, Xf[i, :])
        Xc[i, :] = cs_x(RodDisCoarsePtsVec)

        cs_d1 = CubicSpline(RodDisFinePtsVec, Df1[i, :])
        Dc1[i, :] = cs_d1(RodDisCoarsePtsVec)
        
        cs_d2 = CubicSpline(RodDisFinePtsVec, Df2[i, :])
        Dc2[i, :] = cs_d2(RodDisCoarsePtsVec)
        
        cs_d3 = CubicSpline(RodDisFinePtsVec, Df3[i, :])
        Dc3[i, :] = cs_d3(RodDisCoarsePtsVec)


    # re-orthogonalize
    for kk in range(m):
        d1 = Dc1[:, kk]
        d2 = Dc2[:, kk]
        d3 = Dc3[:, kk]
        d3 = d3/np.linalg.norm(d3)
        d1 = d1 - np.dot(d1, d3)*d3
        d1 = d1/np.linalg.norm(d1)
        d2 = d2 - np.dot(d2, d3)*d3 - np.dot(d2, d1)*d1
        d2 = d2/np.linalg.norm(d2)
        Dc3[:, kk] = d3 
        Dc1[:, kk] = d1 
        Dc2[:, kk] = d2 

    return Xc, Dc1, Dc2, Dc3


def CuSpInsertFineGrid(RodDisCoarsePtsVec,RodDisFinePtsVec,Xc,Dc1,Dc2,Dc3):
    #Interpolate the coarse grid using the cubic spline


    #initialize the fine grids
    M = np.size(RodDisFinePtsVec)
    Xf = np.zeros((3, M))
    Df1 = np.zeros((3, M))
    Df2 = np.zeros((3, M))
    Df3 = np.zeros((3, M))

    for i in range(3):
        cs_x = CubicSpline(RodDisCoarsePtsVec, Xc[i, :])
        Xf[i, :] = cs_x(RodDisFinePtsVec)

        cs_d1 = CubicSpline(RodDisCoarsePtsVec, Dc1[i, :])
        Df1[i, :] = cs_d1(RodDisFinePtsVec)
        
        cs_d2 = CubicSpline(RodDisCoarsePtsVec, Dc2[i, :])
        Df2[i, :] = cs_d2(RodDisFinePtsVec)
        
        cs_d3 = CubicSpline(RodDisCoarsePtsVec, Dc3[i, :])
        Df3[i, :] = cs_d3(RodDisFinePtsVec)


    # re-orthogonalize
    for kk in range(M):
        d1 = Df1[:, kk]
        d2 = Df2[:, kk]
        d3 = Df3[:, kk]
        d3 = d3/np.linalg.norm(d3)
        d1 = d1 - np.dot(d1, d3)*d3
        d1 = d1/np.linalg.norm(d1)
        d2 = d2 - np.dot(d2, d3)*d3 - np.dot(d2, d1)*d1
        d2 = d2/np.linalg.norm(d2)
        Df3[:, kk] = d3 
        Df1[:, kk] = d1 
        Df2[:, kk] = d2 

    return Xf, Df1, Df2, Df3



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dt = 1e-6  

NumTimeInterval = comm.size


T = .1

LengthTimeInterval = T / NumTimeInterval


FineNumTimeStepsPerInterval = round(LengthTimeInterval / dt)
dt = LengthTimeInterval / FineNumTimeStepsPerInterval

r0 = 8
dT = r0*dt
CoarseNumTimeStepsPerInterval = round(LengthTimeInterval / dT)

dT = LengthTimeInterval/CoarseNumTimeStepsPerInterval


CorrFlag = 1
NumParSweeps = 1
tol = 1e-10


#Number of grid points 
M = 301

#Rod length
L = 60.

ds = L/(M-1)

a1 = 1
a2 = 1
a3 = 1

b1 = .6
b2 = .6
b3 = .6

avec = np.array([a1, a2, a3])
bvec = np.array([b1, b2, b3])

RodDisFinePtsVec = np.arange(0,M)*ds
TotalNumRods = 1
NumRods = TotalNumRods

FrequencySigma = 40*np.pi
WaveLengthLambda = L*.5
NumPeriods = WaveLengthLambda/(2*np.pi)

AmplitudeA = 1.
AmplitudeB = 1.
OmegaFactor = -1./(NumPeriods**2)*AmplitudeB

ep = 5*ds
ViscosityNu = 1e-6


class GridParam:
   def __init__(self, M, ds, X, D1, D2, D3, RodDisPtsVec):
       self.M = M
       self.ds = ds
       self.X = X
       self.D1 = D1
       self.D2 = D2
       self.D3 = D3
       self.RodDisPtsVec = RodDisPtsVec


class RodsParam:
   def __init__(self, NumRods, avec, bvec):
       self.NumRods = NumRods
       self.avec = avec
       self.bvec = bvec


class SwaveParam:
   def __init__(self, OmegaFactor, FrequencySigma, NumPeriods):
       self.OmegaFactor = OmegaFactor
       self.FrequencySigma = FrequencySigma
       self.NumPeriods = NumPeriods


TotalNumFinePts = TotalNumRods*M

RodsParam = RodsParam(NumRods, avec, bvec)
SwaveParam = SwaveParam(OmegaFactor, FrequencySigma, NumPeriods)


# initial coordinates of the fine mesh points
X = np.zeros((3, TotalNumFinePts))
D1 = np.zeros((3, TotalNumFinePts))
D2 = np.zeros((3, TotalNumFinePts))
D3 = np.zeros((3, TotalNumFinePts))

ShiftFromOrigin = 0
DistanceFromWall = 10

RodLength = L

D3[0, :] = np.ones(TotalNumFinePts)
D1[1, :] = np.ones(TotalNumFinePts)
D2[2, :] = np.ones(TotalNumFinePts)

# The code currently considers a single rod initialized as a straight rod
X[0, 0:M] = RodDisFinePtsVec
X[0, 0:M] += - RodLength
X[2, 0:M] = DistanceFromWall


if rank == 0:
    # Load serial solution
    in_file = np.loadtxt("Serial_5ds_T1.txt", dtype=float)
    
    X_true1 = np.transpose(in_file[:,0:3])
    D1_true1 = np.transpose(in_file[:,3:6])
    D2_true1 = np.transpose(in_file[:,6:9])
    D3_true1 = np.transpose(in_file[:,9:12])

start_time = MPI.Wtime()

if rank == 0:

    Xfn = np.zeros((3,TotalNumFinePts,NumTimeInterval+1))
    D1fn = np.zeros((3,TotalNumFinePts,NumTimeInterval+1))
    D2fn = np.zeros((3,TotalNumFinePts,NumTimeInterval+1))
    D3fn = np.zeros((3,TotalNumFinePts,NumTimeInterval+1))
    
    #Assign initial condition to Xfn
    Xfn[:,:,0] = X
    D1fn[:,:,0] = D1
    D2fn[:,:,0] = D2
    D3fn[:,:,0] = D3
    

    for ll in range(NumTimeInterval):
        GridParam0 = GridParam(M, ds, Xfn[:,:,ll], D1fn[:,:,ll], D2fn[:,:,ll], D3fn[:,:,ll], RodDisFinePtsVec)
        Xr, D1_r, D2_r, D3_r, RodLengthVec = Krod_Swave_Wall_FE2(dT, CoarseNumTimeStepsPerInterval, GridParam0, RodsParam, SwaveParam, ll * LengthTimeInterval, ep, ViscosityNu)
        Xfn[:, :, ll + 1] = Xr
        D1fn[:, :, ll + 1] = D1_r
        D2fn[:, :, ll + 1] = D2_r
        D3fn[:, :, ll + 1] = D3_r

    
    g1_X = np.zeros((3,TotalNumFinePts,NumTimeInterval))
    g1_D1 = np.zeros((3,TotalNumFinePts,NumTimeInterval))
    g1_D2 = np.zeros((3,TotalNumFinePts,NumTimeInterval))
    g1_D3 = np.zeros((3,TotalNumFinePts,NumTimeInterval))
    
    g1_X[:,:,:] = Xfn[:,:,1:]
    g1_D1[:,:,:] = D1fn[:,:,1:]
    g1_D2[:,:,:] = D2fn[:,:,1:]
    g1_D3[:,:,:] = D3fn[:,:,1:]

    g2_vec_X = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    g2_vec_D1 = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    g2_vec_D2 = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    g2_vec_D3 = np.zeros((3, TotalNumFinePts, NumTimeInterval))

    X_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D1_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D2_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D3_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    
    X_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D1_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D2_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D3_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))

    X_ic_vec[:,:,:] = Xfn[:,:,0:NumTimeInterval]
    D1_ic_vec[:,:,:] = D1fn[:,:,0:NumTimeInterval]
    D2_ic_vec[:,:,:] = D2fn[:,:,0:NumTimeInterval]
    D3_ic_vec[:,:,:] = D3fn[:,:,0:NumTimeInterval]

    X_sol_fn[:,:,:] = Xfn[:,:,1:]
    D1_sol_fn[:,:,:] = D1fn[:,:,1:]
    D2_sol_fn[:,:,:] = D2fn[:,:,1:]
    D3_sol_fn[:,:,:] = D3fn[:,:,1:]


if rank == 0:
    send_buf_x = np.split(X_ic_vec, NumTimeInterval, axis = 2)
    send_buf_d1 = np.split(D1_ic_vec, NumTimeInterval, axis = 2)
    send_buf_d2 = np.split(D2_ic_vec, NumTimeInterval, axis = 2)
    send_buf_d3 = np.split(D3_ic_vec, NumTimeInterval, axis = 2)
    


else:
    send_buf_x = None
    send_buf_d1 = None
    send_buf_d2 = None
    send_buf_d3 = None


recv_obj_x = comm.scatter(send_buf_x, root=0)
x_sol_ic = recv_obj_x

recv_obj_d1 = comm.scatter(send_buf_d1, root=0)
d1_sol_ic = recv_obj_d1

recv_obj_d2 = comm.scatter(send_buf_d2, root=0)
d2_sol_ic = recv_obj_d2

recv_obj_d3 = comm.scatter(send_buf_d3, root=0)
d3_sol_ic = recv_obj_d3
   
comm.Barrier()


if rank == 0:

    # Print out the true error and relative true error after the initial serial sweep
    AbsErrX = np.linalg.norm(Xfn[:,:,-1]-X_true1)
    AbsErrX1 = AbsErrX / np.linalg.norm(X_true1)
    print(AbsErrX)
    print(AbsErrX1)

for ii in range(NumParSweeps):

    # Parallel fine sweep using the IC received from rank 0
    
    GridParam_f = GridParam(M, ds, x_sol_ic[:,:,0], d1_sol_ic[:,:,0], d2_sol_ic[:,:,0], d3_sol_ic[:,:,0], RodDisFinePtsVec)
    Xf, D1_f, D2_f, D3_f, RodLengthVec = Krod_Swave_Wall_RK2_2(dt, FineNumTimeStepsPerInterval, GridParam_f, RodsParam, SwaveParam, rank * LengthTimeInterval, ep, ViscosityNu)



    Xf_vec = comm.gather(Xf, root = 0)
    D1f_vec = comm.gather(D1_f, root = 0)
    D2f_vec = comm.gather(D2_f, root = 0)
    D3f_vec = comm.gather(D3_f, root = 0) 


    comm.Barrier()


    if rank == 0:

        if (ii>0):
            g1_X[:,:,ii:] = g2_vec_X[:,:,ii:]
            g1_D1[:,:,ii:] = g2_vec_D1[:,:,ii:]
            g1_D2[:,:,ii:] = g2_vec_D2[:,:,ii:]
            g1_D3[:,:,ii:] = g2_vec_D3[:,:,ii:]
            

        for qq in range(ii, NumTimeInterval):
            X_sol_fn[:,:,qq] = Xf_vec[qq]
            D1_sol_fn[:,:,qq] = D1f_vec[qq]
            D2_sol_fn[:,:,qq] = D2f_vec[qq]
            D3_sol_fn[:,:,qq] = D3f_vec[qq]
     
        
        
        
        for jj in range(ii + 1, NumTimeInterval):
            GridParam1 = GridParam(M, ds, X_sol_fn[:,:,jj-1], D1_sol_fn[:,:,jj-1], D2_sol_fn[:,:,jj-1], D3_sol_fn[:,:,jj-1], RodDisFinePtsVec)
            X_g2, D1_g2, D2_g2, D3_g2, RodLengthVec = Krod_Swave_Wall_FE2(dT, CoarseNumTimeStepsPerInterval, GridParam1, RodsParam, SwaveParam, jj * LengthTimeInterval, ep, ViscosityNu)


            g2_vec_X[:,:,jj] = X_g2
            g2_vec_D1[:,:,jj] = D1_g2
            g2_vec_D2[:,:,jj] = D2_g2
            g2_vec_D3[:,:,jj] = D3_g2
            
            
            X_sol_fn[:,:,jj] = X_sol_fn[:,:,jj] + (X_g2 - g1_X[:,:,jj])
            D1_sol_fn[:,:,jj] = D1_sol_fn[:,:,jj] + (D1_g2 - g1_D1[:,:,jj])
            D2_sol_fn[:,:,jj] = D2_sol_fn[:,:,jj] + (D2_g2 - g1_D2[:,:,jj])
            D3_sol_fn[:,:,jj] = D3_sol_fn[:,:,jj] + (D3_g2 - g1_D3[:,:,jj])
            
            #Re-orthogonalization of triads
            for kk in range(TotalNumFinePts):
                d1 = D1_sol_fn[:,kk,jj]
                d2 = D2_sol_fn[:,kk,jj]
                d3 = D3_sol_fn[:,kk,jj]
                d3 = d3/np.linalg.norm(d3)
                d2 = d2 - np.dot(d2,d3)*d3
                d2 = d2/np.linalg.norm(d2)
                d1 = d1 - np.dot(d1,d3)*d3 - np.dot(d1,d2)*d2
                d1 = d1/np.linalg.norm(d1)
                D3_sol_fn[:,kk,jj] = d3
                D1_sol_fn[:,kk,jj] = d1
                D2_sol_fn[:,kk,jj] = d2
            
        
        for kk in range(ii+1, NumTimeInterval):
            X_ic_vec[:,:,kk] = X_sol_fn[:,:,kk-1]
            D1_ic_vec[:,:,kk] = D1_sol_fn[:,:,kk-1]
            D2_ic_vec[:,:,kk] = D2_sol_fn[:,:,kk-1]
            D3_ic_vec[:,:,kk] = D3_sol_fn[:,:,kk-1]

        send_buf_x = np.split(X_ic_vec, NumTimeInterval, axis = 2)
        send_buf_d1 = np.split(D1_ic_vec, NumTimeInterval, axis = 2)
        send_buf_d2 = np.split(D2_ic_vec, NumTimeInterval, axis = 2)
        send_buf_d3 = np.split(D3_ic_vec, NumTimeInterval, axis = 2)
    

        print(ii)
        RelErrX = np.linalg.norm(X_sol_fn[:,:,-1] - Xfn[:,:,-1])/np.linalg.norm(X_sol_fn[:,:,-1])
        AbsErrX = np.linalg.norm(X_sol_fn[:,:,-1] - X_true1)
        AbsErrX1 = AbsErrX/np.linalg.norm(X_true1)

        # Print out the relative increment, true error and relative true error
        print('RelErr= %.16g, AbsErrX=%.16g, AbsErrX1=%.16g\n' % (RelErrX, AbsErrX, AbsErrX1))
        
        Xfn[:,:,1:] = X_sol_fn
        D1fn[:,:,1:] = D1_sol_fn
        D2fn[:,:,1:] = D2_sol_fn
        D3fn[:,:,1:] = D3_sol_fn
        
    
        
    else:
        send_buf_x = None
        send_buf_d1 = None
        send_buf_d2 = None
        send_buf_d3 = None


    recv_obj_x = comm.scatter(send_buf_x, root=0)
    x_sol_ic = recv_obj_x

    recv_obj_d1 = comm.scatter(send_buf_d1, root=0)
    d1_sol_ic = recv_obj_d1

    recv_obj_d2 = comm.scatter(send_buf_d2, root=0)
    d2_sol_ic = recv_obj_d2

    recv_obj_d3 = comm.scatter(send_buf_d3, root=0)
    d3_sol_ic = recv_obj_d3
    
    comm.Barrier()

    end_time = MPI.Wtime()

run_time = end_time - start_time
if rank == 0:
    print("Total number of iterations performed = %d\t: Total time = %.16g\n" %(ii + 1, run_time))
    

    



        
