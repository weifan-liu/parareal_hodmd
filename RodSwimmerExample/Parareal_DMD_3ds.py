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

#This code calculates the example of a single rod-like swimmer with epsilon = 3*sigma for three iterations using the Parareal-HODMD.

#The code for Method of Regularized Stokeslets is adapted from the code provided by Minghao W. Rostami (SUNY Binghamton).
#The code for HODMD is adapted from Soledad Le Clainche and José M. Vega 
# from the link: https://www.researchgate.net/publication/346396464_Matlab_Codes_from_the_book_Higher_order_Dynamic_Mode_Decomposition_and_its_Applications.


def VelocitiesRegSto7halfRodWall(f, n, ep, Xb, Xeval, Qweights, SourcedA):
#This function computes the velocities by the Method of Regularized
#Stokeslets (force and torque with wall). The 7/2 blob function is used. 
#The wall is located at z=0. 

#Inputs:
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
        
        # calculate coefficients for the terms in the linear and angular velocities
        
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


def Krod_Swave_Wall_FE2_Snapshots(dt,maxt,GridParam,RodsParam,SwaveParam,CurrentTime,ep,ViscosityNu,rr):
    # This calculates the position of the rods over maxt time steps using the forward Euler's method 
    # and outputs the snapshots of the rods as a snapshot matrix

    # Extract the parameters from the parameter structures
    
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
    
    num_snapshots = int(maxt/rr) + 1
    
    X_snapshots = np.zeros((NumPts*3, num_snapshots))
    D1_snapshots = np.zeros((NumPts*3, num_snapshots))
    D2_snapshots = np.zeros((NumPts*3, num_snapshots))
    D3_snapshots = np.zeros((NumPts*3, num_snapshots))
    
         
    X_snapshots[0:NumPts, 0] = Xr[0, :]
    X_snapshots[NumPts:(2*NumPts), 0] = Xr[1, :]
    X_snapshots[2*NumPts:(3*NumPts), 0] = Xr[2, :]

    D1_snapshots[0:NumPts, 0] = D1_r[0, :]
    D1_snapshots[NumPts:(2*NumPts), 0] = D1_r[1, :]
    D1_snapshots[2*NumPts:(3*NumPts), 0] = D1_r[2, :]

    D2_snapshots[0:NumPts, 0] = D2_r[0, :]
    D2_snapshots[NumPts:(2*NumPts), 0] = D2_r[1, :]
    D2_snapshots[2*NumPts:(3*NumPts), 0] = D2_r[2, :]

    D3_snapshots[0:NumPts, 0] = D3_r[0, :]
    D3_snapshots[NumPts:(2*NumPts), 0] = D3_r[1, :]
    D3_snapshots[2*NumPts:(3*NumPts), 0] = D3_r[2, :]

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
            
        p1 = np.mod(i,rr)
    
        if p1 == 0: 
            q1 = int(i/rr)
            X_snapshots[0:NumPts, q1] = Xr[0, :]
            X_snapshots[NumPts:(2*NumPts), q1] = Xr[1, :]
            X_snapshots[(2*NumPts):(3*NumPts), q1] = Xr[2, :]

            D1_snapshots[0:NumPts, q1] = D1_r[0, :]
            D1_snapshots[NumPts:(2*NumPts), q1] = D1_r[1, :]
            D1_snapshots[(2*NumPts):(3*NumPts), q1] = D1_r[2, :]
        
            D2_snapshots[0:NumPts, q1] = D2_r[0, :]
            D2_snapshots[NumPts:(2*NumPts), q1] = D2_r[1, :]
            D2_snapshots[(2*NumPts):(3*NumPts), q1] = D2_r[2, :]
        
            D3_snapshots[0:NumPts, q1] = D3_r[0, :]
            D3_snapshots[NumPts:(2*NumPts), q1] = D3_r[1, :]
            D3_snapshots[(2*NumPts):(3*NumPts), q1] = D3_r[2, :]


    return X_snapshots, D1_snapshots, D2_snapshots, D3_snapshots, RodLengthVec


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

            #print(D3h[:, 289])


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

    #initialize the coarse grids
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
    #Interpolate the coarse grid using cubic spline
    
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


T = .1

dt = 1e-6

LengthTimeInterval = T / NumTimeInterval

FineNumTimeStepsPerInterval = round(LengthTimeInterval / dt)
dt = LengthTimeInterval / FineNumTimeStepsPerInterval

r0 = 1
dT = r0 * dt
CoarseNumTimeStepsPerInterval = round(LengthTimeInterval / dT)

dT = LengthTimeInterval / CoarseNumTimeStepsPerInterval

CorrFlag = 1
NumParSweeps = 3


# Rod length
L = 60.

# Number of fine grid points
M = 301
m = 51


ds = L / (M - 1)
dS = L / (m - 1)

a1 = 1
a2 = 1
a3 = 1

b1 = .6
b2 = .6
b3 = .6

avec = np.array([a1, a2, a3])
bvec = np.array([b1, b2, b3])

RodDisFinePtsVec = np.arange(0, M) * ds
RodDisCoarsePtsVec = np.arange(0, m) * dS
TotalNumRods = 1
NumRods = TotalNumRods

FrequencySigma = 40 * np.pi
WaveLengthLambda = L * .5
NumPeriods = WaveLengthLambda / (2 * np.pi)

AmplitudeA = 1.
AmplitudeB = 1.
OmegaFactor = -1. / (NumPeriods ** 2) * AmplitudeB

ep = 3 * ds
EP = 3 * ds

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


TotalNumFinePts = TotalNumRods * M
TotalNumCoarsePts = TotalNumRods * m

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

# The value of Kt
ntest = round(NumTimeInterval * .6)


#DMD sub-interval size for initial serial sweep
rr = 50
qq = round(CoarseNumTimeStepsPerInterval / rr)
ni = NumTimeInterval - ntest

#SVD parameters used in HODMD 
varepsilon1 = 1e-8
varepsilon = varepsilon1

d0 = 10
d = 12

rr0 = 2

rs_vec = np.zeros(NumParSweeps)

rs_vec[0] = .44
rs_vec[1] = .78
rs_vec[2] = .96


nt_vec = np.round(CoarseNumTimeStepsPerInterval * rs_vec, decimals=1)

#Make sure nt_vec is even
for pp in range(NumParSweeps):
    n0 = int(nt_vec[pp])
    if np.mod(n0, 2) != 0:
        nt_vec[pp] = n0 + 1
    nt_vec[pp] = int(nt_vec[pp])



#DMD sub-interval size for serial correction
nt_c_vec = np.round(nt_vec / rr0 / 2, decimals = 1)

for pp in range(NumParSweeps):
    nt_c_vec[pp] = int(nt_c_vec[pp])

nt_c1 = int(nt_c_vec[0])

#Total number of time steps each interval in the serial correction
qq_c = int(np.round(CoarseNumTimeStepsPerInterval / rr0 / 2, decimals = 1))
nt_c_max = int(np.max(nt_c_vec))

qq0 = int(round(CoarseNumTimeStepsPerInterval / rr0))

X_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
D1_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
D2_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))
D3_sol_fn = np.zeros((3, TotalNumFinePts, NumTimeInterval))

ratio_rr = int(rr/rr0)

if rank == 0:
    # Load the serial solution 
    in_file = np.loadtxt("Serial_3ds_T1_New.txt", dtype=float)
 
    X_true1 = np.transpose(in_file[:, 0:3])
    D1_true1 = np.transpose(in_file[:, 3:6])
    D2_true1 = np.transpose(in_file[:, 6:9])
    D3_true1 = np.transpose(in_file[:, 9:12])

start_time = MPI.Wtime()

if rank == 0:

    Xfn = np.zeros((3, TotalNumFinePts, NumTimeInterval + 1))
    D1fn = np.zeros((3, TotalNumFinePts, NumTimeInterval + 1))
    D2fn = np.zeros((3, TotalNumFinePts, NumTimeInterval + 1))
    D3fn = np.zeros((3, TotalNumFinePts, NumTimeInterval + 1))

    # Assign initial condition to Xfn
    Xfn[:, :, 0] = X
    D1fn[:, :, 0] = D1
    D2fn[:, :, 0] = D2
    D3fn[:, :, 0] = D3

    X_snapshots_all = np.zeros((TotalNumFinePts * 3, ntest * qq + 1))
    D1_snapshots_all = np.zeros((TotalNumFinePts * 3, ntest * qq + 1))
    D2_snapshots_all = np.zeros((TotalNumFinePts * 3, ntest * qq + 1))
    D3_snapshots_all = np.zeros((TotalNumFinePts * 3, ntest * qq + 1))
    
    X_snapshots_old_mat = np.zeros((3 * TotalNumFinePts, nt_c_max + 1, NumTimeInterval))
    D1_snapshots_old_mat = np.zeros((3 * TotalNumFinePts, nt_c_max + 1, NumTimeInterval))
    D2_snapshots_old_mat = np.zeros((3 * TotalNumFinePts, nt_c_max + 1, NumTimeInterval))
    D3_snapshots_old_mat = np.zeros((3 * TotalNumFinePts, nt_c_max + 1, NumTimeInterval))

    X_snapshots_all[0:TotalNumFinePts, 0] = X[0, :]
    X_snapshots_all[TotalNumFinePts:(2 * TotalNumFinePts), 0] = X[1, :]
    X_snapshots_all[2 * TotalNumFinePts:(3 * TotalNumFinePts), 0] = X[2, :]

    D1_snapshots_all[0:TotalNumFinePts, 0] = D1[0, :]
    D1_snapshots_all[TotalNumFinePts:(2 * TotalNumFinePts), 0] = D1[1, :]
    D1_snapshots_all[2 * TotalNumFinePts:(3 * TotalNumFinePts), 0] = D1[2, :]

    D2_snapshots_all[0:TotalNumFinePts, 0] = D2[0, :]
    D2_snapshots_all[TotalNumFinePts:(2 * TotalNumFinePts), 0] = D2[1, :]
    D2_snapshots_all[2 * TotalNumFinePts:(3 * TotalNumFinePts), 0] = D2[2, :]

    D3_snapshots_all[0:TotalNumFinePts, 0] = D3[0, :]
    D3_snapshots_all[TotalNumFinePts:(2 * TotalNumFinePts), 0] = D3[1, :]
    D3_snapshots_all[2 * TotalNumFinePts:(3 * TotalNumFinePts), 0] = D3[2, :]

    for ll in range(ntest):
        GridParam0 = GridParam(M, ds, Xfn[:, :, ll], D1fn[:, :, ll], D2fn[:, :, ll], D3fn[:, :, ll], RodDisFinePtsVec)
        X_snapshots, D1_snapshots, D2_snapshots, D3_snapshots, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(dT,
                                                                                                            CoarseNumTimeStepsPerInterval,
                                                                                                            GridParam0,
                                                                                                            RodsParam,
                                                                                                            SwaveParam,
                                                                                                            ll * LengthTimeInterval,
                                                                                                            ep,
                                                                                                            ViscosityNu,
                                                                                                            rr0)

        # Update Xfn here
        Xfn[:, :, ll + 1] = np.reshape(X_snapshots[:, -1], (3, TotalNumFinePts))
        D1fn[:, :, ll + 1] = np.reshape(D1_snapshots[:, -1], (3, TotalNumFinePts))
        D2fn[:, :, ll + 1] = np.reshape(D2_snapshots[:, -1], (3, TotalNumFinePts))
        D3fn[:, :, ll + 1] = np.reshape(D3_snapshots[:, -1], (3, TotalNumFinePts))


        X_snapshots_all[:, (ll * qq + 1):((ll + 1) * qq + 1)] = X_snapshots[:, 1:(qq0+1):ratio_rr]
        D1_snapshots_all[:, (ll * qq + 1):((ll + 1) * qq + 1)] = D1_snapshots[:, 1:(qq0+1):ratio_rr]
        D2_snapshots_all[:, (ll * qq + 1):((ll + 1) * qq + 1)] = D2_snapshots[:, 1:(qq0+1):ratio_rr]
        D3_snapshots_all[:, (ll * qq + 1):((ll + 1) * qq + 1)] = D3_snapshots[:, 1:(qq0+1):ratio_rr]


        #snapshot matrix stored for serial correction with interval ratio r0
        X_snapshots_old_mat[:, 0:(nt_c1+1), ll] = X_snapshots[:, 0:(2*nt_c1+1):2]
        D1_snapshots_old_mat[:, 0:(nt_c1+1), ll] = D1_snapshots[:, 0:(2*nt_c1+1):2]
        D2_snapshots_old_mat[:, 0:(nt_c1+1), ll] = D2_snapshots[:, 0:(2*nt_c1+1):2]
        D3_snapshots_old_mat[:, 0:(nt_c1+1), ll] = D3_snapshots[:, 0:(2*nt_c1+1):2]

        print("Initial serial sweep on fine grid - first %d intervals"%(ll))



if rank == 1:

    # rank 1 calculates the coarse-grid solution
    
    # initial coordinates of the coarse mesh points
    Xc = np.zeros((3, TotalNumCoarsePts))
    D1c = np.zeros((3, TotalNumCoarsePts))
    D2c = np.zeros((3, TotalNumCoarsePts))
    D3c = np.zeros((3, TotalNumCoarsePts))
    X3Vec = np.ones(TotalNumRods)*DistanceFromWall

    D3c[0,:] = np.ones(TotalNumCoarsePts)
    D1c[1,:] = np.ones(TotalNumCoarsePts)
    D2c[2,:] = np.ones(TotalNumCoarsePts)

    Xc[0, 0:m] = RodDisCoarsePtsVec
    Xc[0, 0:m] += -RodLength
    Xc[2, 0:m] = DistanceFromWall

    X_snapshots_c = np.zeros((TotalNumCoarsePts*3, ntest*qq + 1))
    D1_snapshots_c = np.zeros((TotalNumCoarsePts*3, ntest*qq + 1))
    D2_snapshots_c = np.zeros((TotalNumCoarsePts*3, ntest*qq + 1))
    D3_snapshots_c = np.zeros((TotalNumCoarsePts*3, ntest*qq + 1))
    
    ni = NumTimeInterval - ntest
    X_snapshots_ni = np.zeros((3, TotalNumCoarsePts, ni))
    D1_snapshots_ni = np.zeros((3, TotalNumCoarsePts, ni))
    D2_snapshots_ni = np.zeros((3, TotalNumCoarsePts, ni))
    D3_snapshots_ni = np.zeros((3, TotalNumCoarsePts, ni))

    
    
    X_snapshots_old_c = np.zeros((TotalNumCoarsePts * 3, nt_c_max + 1, NumTimeInterval))
    D1_snapshots_old_c = np.zeros((TotalNumCoarsePts * 3, nt_c_max + 1, NumTimeInterval))
    D2_snapshots_old_c = np.zeros((TotalNumCoarsePts * 3, nt_c_max + 1, NumTimeInterval))
    D3_snapshots_old_c = np.zeros((TotalNumCoarsePts * 3, nt_c_max + 1, NumTimeInterval))

    # The coarse-grid solution at the end of each time interval
    X_tend_old_c = np.zeros((TotalNumCoarsePts*3, NumTimeInterval))
    D1_tend_old_c = np.zeros((TotalNumCoarsePts*3, NumTimeInterval))
    D2_tend_old_c = np.zeros((TotalNumCoarsePts*3, NumTimeInterval))
    D3_tend_old_c = np.zeros((TotalNumCoarsePts*3, NumTimeInterval))

    # The initial condition (coarse mesh)
    X_snapshots_c[0:TotalNumCoarsePts, 0] = Xc[0, :]
    X_snapshots_c[TotalNumCoarsePts:(2*TotalNumCoarsePts), 0] = Xc[1, :]
    X_snapshots_c[(2*TotalNumCoarsePts):(3*TotalNumCoarsePts), 0] = Xc[2, :]

    D1_snapshots_c[0:TotalNumCoarsePts, 0] = D1c[0, :]
    D1_snapshots_c[TotalNumCoarsePts:(2*TotalNumCoarsePts), 0] = D1c[1, :]
    D1_snapshots_c[(2*TotalNumCoarsePts):(3*TotalNumCoarsePts), 0] = D1c[2, :]

    D2_snapshots_c[0:TotalNumCoarsePts, 0] = D2c[0, :]
    D2_snapshots_c[TotalNumCoarsePts:(2*TotalNumCoarsePts), 0] = D2c[1, :]
    D2_snapshots_c[(2*TotalNumCoarsePts):(3*TotalNumCoarsePts), 0] = D2c[2, :]

    D3_snapshots_c[0:TotalNumCoarsePts, 0] = D3c[0, :]
    D3_snapshots_c[TotalNumCoarsePts:(2*TotalNumCoarsePts), 0] = D3c[1, :]
    D3_snapshots_c[(2*TotalNumCoarsePts):(3*TotalNumCoarsePts), 0] = D3c[2, :]
    
    #Calculate the coarse-grid solutions for the first ntest intervals
    for ll in range(ntest):
        CoarseGridParam0 = GridParam(m, dS, Xc, D1c, D2c, D3c, RodDisCoarsePtsVec)
        X_snapshots_t, D1_snapshots_t, D2_snapshots_t, D3_snapshots_t, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(dT, CoarseNumTimeStepsPerInterval, CoarseGridParam0, RodsParam, SwaveParam, ll * LengthTimeInterval, EP, ViscosityNu, rr0)
        
        X_snapshots_c[:, (ll*qq+1):((ll+1)*qq+1)] = X_snapshots_t[:, 1:(qq0+1):ratio_rr]
        D1_snapshots_c[:, (ll*qq+1):((ll+1)*qq+1)] = D1_snapshots_t[:, 1:(qq0+1):ratio_rr]
        D2_snapshots_c[:, (ll*qq+1):((ll+1)*qq+1)] = D2_snapshots_t[:, 1:(qq0+1):ratio_rr]
        D3_snapshots_c[:, (ll*qq+1):((ll+1)*qq+1)] = D3_snapshots_t[:, 1:(qq0+1):ratio_rr]

        #the snapshot matrix for the coarse-grid solutions, stored for serial correction
        X_snapshots_old_c[:, 0:(nt_c1+1), ll] = X_snapshots_t[:, 0:(2*nt_c1+1):2]
        D1_snapshots_old_c[:, 0:(nt_c1+1), ll] = D1_snapshots_t[:, 0:(2*nt_c1+1):2]
        D2_snapshots_old_c[:, 0:(nt_c1+1), ll] = D2_snapshots_t[:, 0:(2*nt_c1+1):2]
        D3_snapshots_old_c[:, 0:(nt_c1+1), ll] = D3_snapshots_t[:, 0:(2*nt_c1+1):2]

        X_tend_old_c[:, ll] = X_snapshots_t[:, -1]
        D1_tend_old_c[:, ll] = D1_snapshots_t[:, -1]
        D2_tend_old_c[:, ll] = D2_snapshots_t[:, -1]
        D3_tend_old_c[:, ll] = D3_snapshots_t[:, -1]

        Xc[:, :] = np.reshape(X_snapshots_t[:, -1], (3, TotalNumCoarsePts))
        D1c[:, :] = np.reshape(D1_snapshots_t[:, -1], (3, TotalNumCoarsePts))
        D2c[:, :] = np.reshape(D2_snapshots_t[:, -1], (3, TotalNumCoarsePts))
        D3c[:, :] = np.reshape(D3_snapshots_t[:, -1], (3, TotalNumCoarsePts))

        print("Initial serial sweep on coarse grid - interval %d on rank %d\n"%(ll, rank))


    for ll in range(ntest, NumTimeInterval):
        CoarseGridParam0 = GridParam(m, dS, Xc, D1c, D2c, D3c, RodDisCoarsePtsVec)

        X_snapshots_ni1, D1_snapshots_ni1, D2_snapshots_ni1, D3_snapshots_ni1, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(dT, CoarseNumTimeStepsPerInterval, CoarseGridParam0, RodsParam, SwaveParam, ll * LengthTimeInterval, EP, ViscosityNu, rr0)

        X_snapshots_ni[:, :, ll - ntest] = np.reshape(X_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D1_snapshots_ni[:, :, ll - ntest] = np.reshape(D1_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D2_snapshots_ni[:, :, ll - ntest] = np.reshape(D2_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D3_snapshots_ni[:, :, ll - ntest] = np.reshape(D3_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        
        X_snapshots_old_c[:, 0:(nt_c1+1), ll] = X_snapshots_ni1[:, 0:(2*nt_c1+1):2]
        D1_snapshots_old_c[:, 0:(nt_c1+1), ll] = D1_snapshots_ni1[:, 0:(2*nt_c1+1):2]
        D2_snapshots_old_c[:, 0:(nt_c1+1), ll] = D2_snapshots_ni1[:, 0:(2*nt_c1+1):2]
        D3_snapshots_old_c[:, 0:(nt_c1+1), ll] = D3_snapshots_ni1[:, 0:(2*nt_c1+1):2]

        X_tend_old_c[:, ll] = X_snapshots_ni1[:, -1]
        D1_tend_old_c[:, ll] = D1_snapshots_ni1[:, -1]
        D2_tend_old_c[:, ll] = D2_snapshots_ni1[:, -1]
        D3_tend_old_c[:, ll] = D3_snapshots_ni1[:, -1]

        Xc[:, :] = np.reshape(X_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D1c[:, :] = np.reshape(D1_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D2c[:, :] = np.reshape(D2_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))
        D3c[:, :] = np.reshape(D3_snapshots_ni1[:, -1], (3, TotalNumCoarsePts))

        print("Initial serial sweep on coarse grid - interval %d on rank %d\n"%(ll, rank))

    comm.send(X_snapshots_ni, dest=0, tag=11)
    comm.send(D1_snapshots_ni, dest=0, tag=22)
    comm.send(D2_snapshots_ni, dest=0, tag=33)
    comm.send(D3_snapshots_ni, dest=0, tag=44)

    comm.send(X_snapshots_c, dest=0, tag=1)
    comm.send(D1_snapshots_c, dest=0, tag=2)
    comm.send(D2_snapshots_c, dest=0, tag=3)
    comm.send(D3_snapshots_c, dest=0, tag=4)


    comm.send(X_snapshots_old_c, dest=0, tag=10)
    comm.send(D1_snapshots_old_c, dest=0, tag=20)
    comm.send(D2_snapshots_old_c, dest=0, tag=30)
    comm.send(D3_snapshots_old_c, dest=0, tag=40)

    
if rank == 0:
    X_snapshots_ni = comm.recv(source=1, tag=11)
    D1_snapshots_ni = comm.recv(source=1, tag=22)
    D2_snapshots_ni = comm.recv(source=1, tag=33)
    D3_snapshots_ni = comm.recv(source=1, tag=44)
    
    X_snapshots_c = comm.recv(source=1, tag=1)
    D1_snapshots_c = comm.recv(source=1, tag=2)
    D2_snapshots_c = comm.recv(source=1, tag=3)
    D3_snapshots_c = comm.recv(source=1, tag=4)

    X_snapshots_old_c = comm.recv(source=1, tag=10)
    D1_snapshots_old_c = comm.recv(source=1, tag=20)
    D2_snapshots_old_c = comm.recv(source=1, tag=30)
    D3_snapshots_old_c = comm.recv(source=1, tag=40)

    #the snapshot matrix used for storing fine-grid solutions for serial correction
    X_snapshots_old_f = np.zeros((TotalNumFinePts * 3, nt_c_max + 1, NumTimeInterval))
    D1_snapshots_old_f = np.zeros((TotalNumFinePts * 3, nt_c_max + 1, NumTimeInterval))
    D2_snapshots_old_f = np.zeros((TotalNumFinePts * 3, nt_c_max + 1, NumTimeInterval))
    D3_snapshots_old_f = np.zeros((TotalNumFinePts * 3, nt_c_max + 1, NumTimeInterval))


    # Below are some arrays for storing the interpolated coarse-grid solution

    X_correct_ic = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D1_correct_ic = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D2_correct_ic = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D3_correct_ic = np.zeros((3, TotalNumFinePts, NumTimeInterval))

    X_cs2_r = np.zeros((3, TotalNumFinePts, qq_c + 1))
    D1_cs2_r = np.zeros((3, TotalNumFinePts, qq_c + 1))
    D2_cs2_r = np.zeros((3, TotalNumFinePts, qq_c + 1))
    D3_cs2_r = np.zeros((3, TotalNumFinePts, qq_c + 1))

    X_cs2 = np.zeros((3 * TotalNumFinePts, qq_c + 1))
    D1_cs2 = np.zeros((3 * TotalNumFinePts, qq_c + 1))
    D2_cs2 = np.zeros((3 * TotalNumFinePts, qq_c + 1))
    D3_cs2 = np.zeros((3 * TotalNumFinePts, qq_c + 1))
    
    X_cs1 = np.zeros((3*TotalNumFinePts, ntest*qq + 1))
    D1_cs1 = np.zeros((3*TotalNumFinePts, ntest*qq + 1))
    D2_cs1 = np.zeros((3*TotalNumFinePts, ntest*qq + 1))
    D3_cs1 = np.zeros((3*TotalNumFinePts, ntest*qq + 1))

    X_cs1_r = np.zeros((3, TotalNumFinePts, ntest*qq + 1))
    D1_cs1_r = np.zeros((3, TotalNumFinePts, ntest*qq + 1))
    D2_cs1_r = np.zeros((3, TotalNumFinePts, ntest*qq + 1))
    D3_cs1_r = np.zeros((3, TotalNumFinePts, ntest*qq + 1))
    
    X_cs0_r = np.zeros((3, TotalNumFinePts, nt_c_max + 1))
    D1_cs0_r = np.zeros((3, TotalNumFinePts, nt_c_max + 1))
    D2_cs0_r = np.zeros((3, TotalNumFinePts, nt_c_max + 1))
    D3_cs0_r = np.zeros((3, TotalNumFinePts, nt_c_max + 1))

    X_cs0 = np.zeros((3 * TotalNumFinePts, nt_c_max + 1))
    D1_cs0 = np.zeros((3 * TotalNumFinePts, nt_c_max + 1))
    D2_cs0 = np.zeros((3 * TotalNumFinePts, nt_c_max + 1))
    D3_cs0 = np.zeros((3 * TotalNumFinePts, nt_c_max + 1))

    X_interp_end0 = np.zeros((3 * TotalNumFinePts, NumTimeInterval))
    D1_interp_end0 = np.zeros((3 * TotalNumFinePts, NumTimeInterval))
    D2_interp_end0 = np.zeros((3 * TotalNumFinePts, NumTimeInterval))
    D3_interp_end0 = np.zeros((3 * TotalNumFinePts, NumTimeInterval))

    X_interp_end0_r = np.zeros((3, TotalNumFinePts))
    D1_interp_end0_r = np.zeros((3, TotalNumFinePts))
    D2_interp_end0_r = np.zeros((3, TotalNumFinePts))
    D3_interp_end0_r = np.zeros((3, TotalNumFinePts))

    X_ni_r = np.zeros((3, TotalNumFinePts))
    D1_ni_r = np.zeros((3, TotalNumFinePts))
    D2_ni_r = np.zeros((3, TotalNumFinePts))
    D3_ni_r = np.zeros((3, TotalNumFinePts))

    
    lw = ntest*qq + 1
    for ll in range(lw):
        for kk in range(TotalNumRods):
            cStartId = kk*m
            cEndId = (kk+1)*m

            #Assume one rod for now    
            Xc_r1 = np.reshape(X_snapshots_c[:, ll], (3, TotalNumCoarsePts))
            D1c_r1 = np.reshape(D1_snapshots_c[:, ll], (3, TotalNumCoarsePts))
            D2c_r1 = np.reshape(D2_snapshots_c[:, ll], (3, TotalNumCoarsePts))
            D3c_r1 = np.reshape(D3_snapshots_c[:, ll], (3, TotalNumCoarsePts))
                
            # cubic spline is used in interpolation
            Xf1,D1f1,D2f1,D3f1 = CuSpInsertFineGrid(RodDisCoarsePtsVec,RodDisFinePtsVec,Xc_r1[:,cStartId:cEndId],\
                    D1c_r1[:,cStartId:cEndId],D2c_r1[:,cStartId:cEndId],D3c_r1[:,cStartId:cEndId])
            
            fStartId = kk*M
            fEndId = (kk+1)*M
            
            X_cs1_r[:, fStartId:fEndId, ll] = Xf1
            D1_cs1_r[:, fStartId:fEndId, ll] = D1f1
            D2_cs1_r[:, fStartId:fEndId, ll] = D2f1
            D3_cs1_r[:, fStartId:fEndId, ll] = D3f1

    for lq in range(lw):
        X_cs1[:, lq] = np.reshape(X_cs1_r[:,:,lq], 3*TotalNumFinePts)
        D1_cs1[:, lq] = np.reshape(D1_cs1_r[:,:,lq], 3*TotalNumFinePts)
        D2_cs1[:, lq] = np.reshape(D2_cs1_r[:,:,lq], 3*TotalNumFinePts)
        D3_cs1[:, lq] = np.reshape(D3_cs1_r[:,:,lq], 3*TotalNumFinePts)
        

    for pp in range(ntest):
        X_interp_end0[:, pp] = X_cs1[:, (pp+1)*qq] 
        D1_interp_end0[:, pp] = D1_cs1[:, (pp+1)*qq] 
        D2_interp_end0[:, pp] = D2_cs1[:, (pp+1)*qq] 
        D3_interp_end0[:, pp] = D3_cs1[:, (pp+1)*qq] 
    
    print("Interpolation done in rank 0.\n")    

    X_diff_mat = np.zeros((3 * TotalNumFinePts, ntest * qq + 1))
    D1_diff_mat = np.zeros((3 * TotalNumFinePts, ntest * qq + 1))
    D2_diff_mat = np.zeros((3 * TotalNumFinePts, ntest * qq + 1))
    D3_diff_mat = np.zeros((3 * TotalNumFinePts, ntest * qq + 1))
    
    #The difference snapshot matrix of the solution on the first ntest intervals
    X_diff_mat[:, :] = X_snapshots_all - X_cs1
    D1_diff_mat[:, :] = D1_snapshots_all - D1_cs1
    D2_diff_mat[:, :] = D2_snapshots_all - D2_cs1
    D3_diff_mat[:, :] = D3_snapshots_all - D3_cs1

    X_diff_p = HODMD(d0, X_diff_mat, np.arange(qq * NumTimeInterval + 1), varepsilon1, varepsilon)
    D1_diff_p = HODMD(d0, D1_diff_mat, np.arange(qq * NumTimeInterval + 1), varepsilon1, varepsilon)
    D2_diff_p = HODMD(d0, D2_diff_mat, np.arange(qq * NumTimeInterval + 1), varepsilon1, varepsilon)
    D3_diff_p = HODMD(d0, D3_diff_mat, np.arange(qq * NumTimeInterval + 1), varepsilon1, varepsilon)

    print("HODMD done in rank 0.\n")

   #Add the predicted difference to the coarse-grid solutions to get the predicted coarse solution
    for ll in np.arange(ntest, NumTimeInterval):
        for kk in range(TotalNumRods):
            cStartId = kk*m
            cEndId = (kk+1)*m
            
            Xc_r3 = X_snapshots_ni[:, :, ll-ntest]
            D1c_r3 = D1_snapshots_ni[:, :, ll-ntest]
            D2c_r3 = D2_snapshots_ni[:, :, ll-ntest]
            D3c_r3 = D3_snapshots_ni[:, :, ll-ntest]
                
            # cubic spline interpolation first
            Xfi,D1fi,D2fi,D3fi = CuSpInsertFineGrid(RodDisCoarsePtsVec,RodDisFinePtsVec,Xc_r3[:,cStartId:cEndId],\
                D1c_r3[:,cStartId:cEndId],D2c_r3[:,cStartId:cEndId],D3c_r3[:,cStartId:cEndId])
            
            fStartId = kk*M
            fEndId = (kk+1)*M
            
            X_ni_r[:, fStartId:fEndId] = Xfi
            D1_ni_r[:, fStartId:fEndId] = D1fi
            D2_ni_r[:, fStartId:fEndId] = D2fi
            D3_ni_r[:, fStartId:fEndId] = D3fi

        X_interp_end0[:, ll] = np.reshape(X_ni_r, 3*TotalNumFinePts)
        D1_interp_end0[:, ll] = np.reshape(D1_ni_r, 3*TotalNumFinePts) 
        D2_interp_end0[:, ll] = np.reshape(D2_ni_r, 3*TotalNumFinePts) 
        D3_interp_end0[:, ll] = np.reshape(D3_ni_r, 3*TotalNumFinePts)  


    for ll in np.arange(ntest, NumTimeInterval):
        X_t1 = X_interp_end0[:, ll] + X_diff_p[:, (ll+1)*qq]
        D1_t1 = D1_interp_end0[:, ll] + D1_diff_p[:, (ll+1)*qq]
        D2_t1 = D2_interp_end0[:, ll] + D2_diff_p[:, (ll+1)*qq]
        D3_t1 = D3_interp_end0[:, ll] + D3_diff_p[:, (ll+1)*qq]
 
        X_rs = np.reshape(X_t1, (3, TotalNumFinePts))
        D1_rs = np.reshape(D1_t1, (3, TotalNumFinePts))
        D2_rs = np.reshape(D2_t1, (3, TotalNumFinePts))
        D3_rs = np.reshape(D3_t1, (3, TotalNumFinePts))

        #Re-orthogonalization of triads
        for kk in range(TotalNumFinePts):
            d1 = D1_rs[:, kk]
            d2 = D2_rs[:, kk]
            d3 = D3_rs[:, kk]
            d3 = d3/np.linalg.norm(d3)
            d2 = d2 - np.dot(d2,d3)*d3
            d2 = d2/np.linalg.norm(d2)
            d1 = d1 - np.dot(d1,d3)*d3 - np.dot(d1,d2)*d2
            d1 = d1/np.linalg.norm(d1)
            D3_rs[:, kk] = d3
            D1_rs[:, kk] = d1
            D2_rs[:, kk] = d2

        Xfn[:, :, ll + 1] = X_rs
        D1fn[:, :, ll + 1] = D1_rs
        D2fn[:, :, ll + 1] = D2_rs
        D3fn[:, :, ll + 1] = D3_rs

        print("Initial serial sweep done in rank 0.\n")  


    X_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D1_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D2_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))
    D3_ic_vec = np.zeros((3, TotalNumFinePts, NumTimeInterval))

    X_ic_vec[:, :, :] = Xfn[:, :, 0:NumTimeInterval]
    D1_ic_vec[:, :, :] = D1fn[:, :, 0:NumTimeInterval]
    D2_ic_vec[:, :, :] = D2fn[:, :, 0:NumTimeInterval]
    D3_ic_vec[:, :, :] = D3fn[:, :, 0:NumTimeInterval]

    X_sol_fn[:, :, :] = Xfn[:, :, 1:]
    D1_sol_fn[:, :, :] = D1fn[:, :, 1:]
    D2_sol_fn[:, :, :] = D2fn[:, :, 1:]
    D3_sol_fn[:, :, :] = D3fn[:, :, 1:]

if rank == 0:
    send_buf_x = np.split(X_ic_vec, NumTimeInterval, axis=2)
    send_buf_d1 = np.split(D1_ic_vec, NumTimeInterval, axis=2)
    send_buf_d2 = np.split(D2_ic_vec, NumTimeInterval, axis=2)
    send_buf_d3 = np.split(D3_ic_vec, NumTimeInterval, axis=2)

    # send ic data to rank 2 for computing g0 in the first serial correction
    comm.send(X_ic_vec, dest=2, tag=11111)
    comm.send(D1_ic_vec, dest=2, tag=22222)
    comm.send(D2_ic_vec, dest=2, tag=33333)
    comm.send(D3_ic_vec, dest=2, tag=44444)

    comm.send(X_ic_vec, dest=3, tag=13)
    comm.send(D1_ic_vec, dest=3, tag=23)
    comm.send(D2_ic_vec, dest=3, tag=33)
    comm.send(D3_ic_vec, dest=3, tag=43)

else:
    send_buf_x = None
    send_buf_d1 = None
    send_buf_d2 = None
    send_buf_d3 = None

if rank == 2:
    X_ic_vec = comm.recv(source=0, tag=11111)
    D1_ic_vec = comm.recv(source=0, tag=22222)
    D2_ic_vec = comm.recv(source=0, tag=33333)
    D3_ic_vec = comm.recv(source=0, tag=44444)


if rank == 3:
    X_ic_vec = comm.recv(source=0, tag=13)
    D1_ic_vec = comm.recv(source=0, tag=23)
    D2_ic_vec = comm.recv(source=0, tag=33)
    D3_ic_vec = comm.recv(source=0, tag=43)

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

    AbsErrX = np.linalg.norm(Xfn[:, :, -1] - X_true1)
    AbsErrX1 = AbsErrX/np.linalg.norm(X_true1)
    print("Initial serial sweep AbsErrX = %.16g\n" % AbsErrX)
    print("Initial serial sweep Relative AbsErrX1 = %.16g\n" % AbsErrX1)

for ii in range(NumParSweeps):

    # Parallel fine sweep using the IC received from rank 0

    GridParam_f = GridParam(M, ds, x_sol_ic[:, :, 0], d1_sol_ic[:, :, 0], d2_sol_ic[:, :, 0], d3_sol_ic[:, :, 0],
                            RodDisFinePtsVec)
    Xf, D1_f, D2_f, D3_f, RodLengthVec = Krod_Swave_Wall_RK2_2(dt, FineNumTimeStepsPerInterval, GridParam_f, RodsParam,
                                                               SwaveParam, rank * LengthTimeInterval, ep, ViscosityNu)

    # Send fine-grid solution to all processors
    Xf_vec = comm.allgather(Xf)
    D1f_vec = comm.allgather(D1_f)
    D2f_vec = comm.allgather(D2_f)
    D3f_vec = comm.allgather(D3_f)

    comm.Barrier()

    if rank < 4:
        for qq in range(ii, NumTimeInterval):
            X_sol_fn[:, :, qq] = Xf_vec[qq]
            D1_sol_fn[:, :, qq] = D1f_vec[qq]
            D2_sol_fn[:, :, qq] = D2f_vec[qq]
            D3_sol_fn[:, :, qq] = D3f_vec[qq]


    for jj in range(ii + 1, NumTimeInterval):
        nt_c = int(nt_c_vec[ii])
        
        if ii > 0:
            nt_c0 = int(nt_c_vec[ii-1])
    

        if rank == 2:
            if ii == 0:
                if jj >= ntest:
                    FineGridParam0 = GridParam(M, ds, X_ic_vec[:, :, jj], D1_ic_vec[:, :, jj], D2_ic_vec[:, :, jj],
                                       D3_ic_vec[:, :, jj], RodDisFinePtsVec)
  
            # Compute g2 on the fine spatial grid for nt time steps
                    X_snapshots_old, D1_snapshots_old, D2_snapshots_old, D3_snapshots_old, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(
                        dT, nt_c*2*rr0, FineGridParam0, RodsParam, SwaveParam, jj * LengthTimeInterval, ep, ViscosityNu, rr0*2)
                    
                    comm.send(X_snapshots_old, dest=0, tag=1111)
                    comm.send(D1_snapshots_old, dest=0, tag=2222)
                    comm.send(D2_snapshots_old, dest=0, tag=3333)
                    comm.send(D3_snapshots_old, dest=0, tag=4444)

            else:
                if nt_c > nt_c0:
                    FineGridParam0 = GridParam(M, ds, X_correct_ic[:, :, jj], D1_correct_ic[:, :, jj], D2_correct_ic[:, :, jj],
                                       D3_correct_ic[:, :, jj], RodDisFinePtsVec)


                    X_snapshots_old, D1_snapshots_old, D2_snapshots_old, D3_snapshots_old, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(
                    dT, (nt_c-nt_c0)*2*rr0, FineGridParam0, RodsParam, SwaveParam, jj * LengthTimeInterval + dT*nt_c0*2*rr0, ep, ViscosityNu, rr0*2)
                        
                    comm.send(X_snapshots_old, dest=0, tag=1111)
                    comm.send(D1_snapshots_old, dest=0, tag=2222)
                    comm.send(D2_snapshots_old, dest=0, tag=3333)
                    comm.send(D3_snapshots_old, dest=0, tag=4444)

        if rank == 3 and ii == 0:
            xc_1, d1_c, d2_c, d3_c = ExtractCoarseGrid2(X_ic_vec[:, :, jj], D1_ic_vec[:, :, jj], D2_ic_vec[:, :, jj], D3_ic_vec[:, :, jj], RodDisCoarsePtsVec, RodDisFinePtsVec)

            CoarseGridParam12 = GridParam(m, dS, xc_1, d1_c, d2_c, d3_c, RodDisCoarsePtsVec)
    

            X_snapshots_old_ct0, D1_snapshots_old_ct0, D2_snapshots_old_ct0, D3_snapshots_old_ct0, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(
                    dT*2, int(round(CoarseNumTimeStepsPerInterval/2)), CoarseGridParam12, RodsParam, SwaveParam, jj * LengthTimeInterval, EP, ViscosityNu, rr0)
            
            
            comm.send(X_snapshots_old_ct0[:, 0:(nt_c_max+1)], dest=0, tag=1001)
            comm.send(D1_snapshots_old_ct0[:, 0:(nt_c_max+1)], dest=0, tag=2002)
            comm.send(D2_snapshots_old_ct0[:, 0:(nt_c_max+1)], dest=0, tag=3003)
            comm.send(D3_snapshots_old_ct0[:, 0:(nt_c_max+1)], dest=0, tag=4004)

                
            comm.send(X_snapshots_old_ct0[:, -1], dest=0, tag=10010)
            comm.send(D1_snapshots_old_ct0[:, -1], dest=0, tag=20020)
            comm.send(D2_snapshots_old_ct0[:, -1], dest=0, tag=30030)
            comm.send(D3_snapshots_old_ct0[:, -1], dest=0, tag=40040)

        
        if rank == 1:
            if jj == ii + 1:
                ic_x = X_sol_fn[:, :, jj - 1]
                ic_d1 = D1_sol_fn[:, :, jj - 1]
                ic_d2 = D2_sol_fn[:, :, jj - 1]
                ic_d3 = D3_sol_fn[:, :, jj - 1]

            xc_1, d1_c, d2_c, d3_c = ExtractCoarseGrid2(ic_x, ic_d1, ic_d2, ic_d3, RodDisCoarsePtsVec, RodDisFinePtsVec)

            CoarseGridParam11 = GridParam(m, dS, xc_1, d1_c, d2_c, d3_c, RodDisCoarsePtsVec)
   
            X_snapshots_g2_c, D1_snapshots_g2_c, D2_snapshots_g2_c, D3_snapshots_g2_c, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(
                dT*2, int(round(CoarseNumTimeStepsPerInterval/2)), CoarseGridParam11, RodsParam, SwaveParam, jj * LengthTimeInterval,
                EP, ViscosityNu, rr0)

            comm.send(X_snapshots_g2_c, dest=0, tag=111)
            comm.send(D1_snapshots_g2_c, dest=0, tag=222)
            comm.send(D2_snapshots_g2_c, dest=0, tag=333)
            comm.send(D3_snapshots_g2_c, dest=0, tag=444)

            ic_x = comm.recv(source=0, tag=101)
            ic_d1 = comm.recv(source=0, tag=202)
            ic_d2 = comm.recv(source=0, tag=303)
            ic_d3 = comm.recv(source=0, tag=404)

        if rank == 0:
            
            if ii == 0:
                if jj >= ntest:
                    X_snapshots_old = comm.recv(source=2, tag=1111)
                    D1_snapshots_old = comm.recv(source=2, tag=2222)
                    D2_snapshots_old = comm.recv(source=2, tag=3333)
                    D3_snapshots_old = comm.recv(source=2, tag=4444)
                    
                    X_snapshots_old_mat[:, 0:(nt_c+1), jj] = X_snapshots_old
                    D1_snapshots_old_mat[:, 0:(nt_c+1), jj] = D1_snapshots_old
                    D2_snapshots_old_mat[:, 0:(nt_c+1), jj] = D2_snapshots_old
                    D3_snapshots_old_mat[:, 0:(nt_c+1), jj] = D3_snapshots_old

                
                X_snapshots_old_ct0 = comm.recv(source=3, tag=1001)
                D1_snapshots_old_ct0 = comm.recv(source=3, tag=2002)
                D2_snapshots_old_ct0 = comm.recv(source=3, tag=3003)
                D3_snapshots_old_ct0 = comm.recv(source=3, tag=4004)

                X_end_c0 = comm.recv(source=3, tag=10010)
                D1_end_c0 = comm.recv(source=3, tag=20020)
                D2_end_c0 = comm.recv(source=3, tag=30030)
                D3_end_c0 = comm.recv(source=3, tag=40040)
                    

            else:
                if nt_c > nt_c0:
                    X_snapshots_old = comm.recv(source=2, tag=1111)
                    D1_snapshots_old = comm.recv(source=2, tag=2222)
                    D2_snapshots_old = comm.recv(source=2, tag=3333)
                    D3_snapshots_old = comm.recv(source=2, tag=4444)


            g2_X_snapshots_c = comm.recv(source=1, tag=111)
            g2_D1_snapshots_c = comm.recv(source=1, tag=222)
            g2_D2_snapshots_c = comm.recv(source=1, tag=333)
            g2_D3_snapshots_c = comm.recv(source=1, tag=444)
            

            FineGridParam1 = GridParam(M, ds, X_sol_fn[:, :, jj - 1], D1_sol_fn[:, :, jj - 1], D2_sol_fn[:, :, jj - 1],
                                       D3_sol_fn[:, :, jj - 1], RodDisFinePtsVec)
        
            X_snapshots_g2, D1_snapshots_g2, D2_snapshots_g2, D3_snapshots_g2, RodLengthVec = Krod_Swave_Wall_FE2_Snapshots(
                dT, nt_c*2*rr0, FineGridParam1, RodsParam, SwaveParam, jj * LengthTimeInterval, ep, ViscosityNu, rr0*2)


            if ii == 0:
                
                g2_X_snapshots_diff = X_snapshots_g2 - X_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D1_snapshots_diff = D1_snapshots_g2 - D1_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D2_snapshots_diff = D2_snapshots_g2 - D2_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D3_snapshots_diff = D3_snapshots_g2 - D3_snapshots_old_mat[:, 0:(nt_c+1), jj]

            else:
                if nt_c > nt_c0:
                    X_snapshots_old_mat[:, nt_c0:(nt_c+1), jj] = X_snapshots_old
                    D1_snapshots_old_mat[:, nt_c0:(nt_c+1), jj] = D1_snapshots_old
                    D2_snapshots_old_mat[:, nt_c0:(nt_c+1), jj] = D2_snapshots_old
                    D3_snapshots_old_mat[:, nt_c0:(nt_c+1), jj] = D3_snapshots_old
                
                g2_X_snapshots_diff = X_snapshots_g2 - X_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D1_snapshots_diff = D1_snapshots_g2 - D1_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D2_snapshots_diff = D2_snapshots_g2 - D2_snapshots_old_mat[:, 0:(nt_c+1), jj]
                g2_D3_snapshots_diff = D3_snapshots_g2 - D3_snapshots_old_mat[:, 0:(nt_c+1), jj]
                
            X_snapshots_old_mat[:, 0:(nt_c+1), jj] = X_snapshots_g2
            D1_snapshots_old_mat[:, 0:(nt_c+1), jj] = D1_snapshots_g2
            D2_snapshots_old_mat[:, 0:(nt_c+1), jj] = D2_snapshots_g2
            D3_snapshots_old_mat[:, 0:(nt_c+1), jj] = D3_snapshots_g2

            X_correct_ic[:, :, jj] = np.reshape(X_snapshots_g2[:, -1], (3, TotalNumFinePts))
            D1_correct_ic[:, :, jj] = np.reshape(D1_snapshots_g2[:, -1], (3, TotalNumFinePts))
            D2_correct_ic[:, :, jj] = np.reshape(D2_snapshots_g2[:, -1], (3, TotalNumFinePts))
            D3_correct_ic[:, :, jj] = np.reshape(D3_snapshots_g2[:, -1], (3, TotalNumFinePts))

           # Spatial interpolation of the coarse-grid solution - snapshots_old_c and g2_snapshots_c
            # The spatial interpolation over time steps could be potentially further parallelized
            if ii == 0:
                for ll in range(nt_c + 1):
                    for kk in range(TotalNumRods):
                        cStartId = kk * m
                        cEndId = (kk + 1) * m

                        Xc_r0 = np.reshape(X_snapshots_old_ct0[:, ll], (3, TotalNumCoarsePts))
                        D1c_r0 = np.reshape(D1_snapshots_old_ct0[:, ll], (3, TotalNumCoarsePts))
                        D2c_r0 = np.reshape(D2_snapshots_old_ct0[:, ll], (3, TotalNumCoarsePts))
                        D3c_r0 = np.reshape(D3_snapshots_old_ct0[:, ll], (3, TotalNumCoarsePts))

                        # cubic spline is used in interpolation
                        Xf0, D1f0, D2f0, D3f0 = CuSpInsertFineGrid(RodDisCoarsePtsVec, RodDisFinePtsVec,
                                                               Xc_r0[:, cStartId:cEndId], \
                                                               D1c_r0[:, cStartId:cEndId], D2c_r0[:, cStartId:cEndId],
                                                               D3c_r0[:, cStartId:cEndId])

                        fStartId = kk * M
                        fEndId = (kk + 1) * M

                        X_cs0_r[:, fStartId:fEndId, ll] = Xf0
                        D1_cs0_r[:, fStartId:fEndId, ll] = D1f0
                        D2_cs0_r[:, fStartId:fEndId, ll] = D2f0
                        D3_cs0_r[:, fStartId:fEndId, ll] = D3f0

                for lq in range(nt_c + 1):
                    X_cs0[:, lq] = np.reshape(X_cs0_r[:, :, lq], 3 * TotalNumFinePts)
                    D1_cs0[:, lq] = np.reshape(D1_cs0_r[:, :, lq], 3 * TotalNumFinePts)
                    D2_cs0[:, lq] = np.reshape(D2_cs0_r[:, :, lq], 3 * TotalNumFinePts)
                    D3_cs0[:, lq] = np.reshape(D3_cs0_r[:, :, lq], 3 * TotalNumFinePts)
            else:
                X_cs0[:, 0:(nt_c+1)]  = X_snapshots_old_f[:, 0:(nt_c+1), jj]
                D1_cs0[:, 0:(nt_c+1)] = D1_snapshots_old_f[:, 0:(nt_c+1), jj]
                D2_cs0[:, 0:(nt_c+1)] = D2_snapshots_old_f[:, 0:(nt_c+1), jj]
                D3_cs0[:, 0:(nt_c+1)] = D3_snapshots_old_f[:, 0:(nt_c+1), jj]

            if ii == 0:
                for kk in range(TotalNumRods):
                    cStartId = kk * m
                    cEndId = (kk + 1) * m

                    Xc_r0 = np.reshape(X_end_c0, (3, m))
                    D1c_r0 = np.reshape(D1_end_c0, (3, m))
                    D2c_r0 = np.reshape(D2_end_c0, (3, m))
                    D3c_r0 = np.reshape(D3_end_c0, (3, m))

                    # cubic spline is used in interpolation
                    Xf0, D1f0, D2f0, D3f0 = CuSpInsertFineGrid(RodDisCoarsePtsVec, RodDisFinePtsVec,
                                                               Xc_r0[:, cStartId:cEndId], \
                                                               D1c_r0[:, cStartId:cEndId], D2c_r0[:, cStartId:cEndId],
                                                               D3c_r0[:, cStartId:cEndId])

                    fStartId = kk * M
                    fEndId = (kk + 1) * M

                    X_interp_end0_r[:, fStartId:fEndId] = Xf0
                    D1_interp_end0_r[:, fStartId:fEndId] = D1f0
                    D2_interp_end0_r[:, fStartId:fEndId] = D2f0
                    D3_interp_end0_r[:, fStartId:fEndId] = D3f0


                    X_interp_end0[:, jj] = np.reshape(X_interp_end0_r, 3*TotalNumFinePts)
                    D1_interp_end0[:, jj] = np.reshape(D1_interp_end0_r, 3*TotalNumFinePts)
                    D2_interp_end0[:, jj] = np.reshape(D2_interp_end0_r, 3*TotalNumFinePts)
                    D3_interp_end0[:, jj] = np.reshape(D3_interp_end0_r, 3*TotalNumFinePts)

            for ll in range(qq_c + 1):
                for kk in range(TotalNumRods):
                    cStartId = kk * m
                    cEndId = (kk + 1) * m

                    Xc_r2 = np.reshape(g2_X_snapshots_c[:, ll], (3, TotalNumCoarsePts))
                    D1c_r2 = np.reshape(g2_D1_snapshots_c[:, ll], (3, TotalNumCoarsePts))
                    D2c_r2 = np.reshape(g2_D2_snapshots_c[:, ll], (3, TotalNumCoarsePts))
                    D3c_r2 = np.reshape(g2_D3_snapshots_c[:, ll], (3, TotalNumCoarsePts))

                    # cubic spline is used in interpolation
                    Xf2, D1f2, D2f2, D3f2 = CuSpInsertFineGrid(RodDisCoarsePtsVec, RodDisFinePtsVec,
                                                               Xc_r2[:, cStartId:cEndId], \
                                                               D1c_r2[:, cStartId:cEndId], D2c_r2[:, cStartId:cEndId],
                                                               D3c_r2[:, cStartId:cEndId])

                    fStartId = kk * M
                    fEndId = (kk + 1) * M

                    X_cs2_r[:, fStartId:fEndId, ll] = Xf2
                    D1_cs2_r[:, fStartId:fEndId, ll] = D1f2
                    D2_cs2_r[:, fStartId:fEndId, ll] = D2f2
                    D3_cs2_r[:, fStartId:fEndId, ll] = D3f2

            for lq in range(qq_c + 1):
                X_cs2[:, lq] = np.reshape(X_cs2_r[:, :, lq], 3 * TotalNumFinePts)
                D1_cs2[:, lq] = np.reshape(D1_cs2_r[:, :, lq], 3 * TotalNumFinePts)
                D2_cs2[:, lq] = np.reshape(D2_cs2_r[:, :, lq], 3 * TotalNumFinePts)
                D3_cs2[:, lq] = np.reshape(D3_cs2_r[:, :, lq], 3 * TotalNumFinePts)

            
            g2_X_snapshots_diff_coarse = X_cs2[:, 0:(nt_c + 1)] - X_cs0[:, 0:(nt_c + 1)]
            g2_D1_snapshots_diff_coarse = D1_cs2[:, 0:(nt_c + 1)] - D1_cs0[:, 0:(nt_c + 1)]
            g2_D2_snapshots_diff_coarse = D2_cs2[:, 0:(nt_c + 1)] - D2_cs0[:, 0:(nt_c + 1)]
            g2_D3_snapshots_diff_coarse = D3_cs2[:, 0:(nt_c + 1)] - D3_cs0[:, 0:(nt_c + 1)]
            
            
            g2_diff_mat_X = g2_X_snapshots_diff - g2_X_snapshots_diff_coarse
            g2_diff_mat_D1 = g2_D1_snapshots_diff - g2_D1_snapshots_diff_coarse
            g2_diff_mat_D2 = g2_D2_snapshots_diff - g2_D2_snapshots_diff_coarse
            g2_diff_mat_D3 = g2_D3_snapshots_diff - g2_D3_snapshots_diff_coarse

            X_diff_g2_p = HODMD(d, g2_diff_mat_X, np.arange(qq_c + 1), varepsilon1, varepsilon)
            D1_diff_g2_p = HODMD(d, g2_diff_mat_D1, np.arange(qq_c + 1), varepsilon1,
                                 varepsilon)
            D2_diff_g2_p = HODMD(d, g2_diff_mat_D2, np.arange(qq_c + 1), varepsilon1,
                                 varepsilon)
            D3_diff_g2_p = HODMD(d, g2_diff_mat_D3, np.arange(qq_c + 1), varepsilon1,
                                 varepsilon)

            g2_X_diff_tend_c = X_cs2[:, -1] - X_interp_end0[:, jj]
            g2_D1_diff_tend_c = D1_cs2[:, -1] - D1_interp_end0[:, jj]
            g2_D2_diff_tend_c = D2_cs2[:, -1] - D2_interp_end0[:, jj]
            g2_D3_diff_tend_c = D3_cs2[:, -1] - D3_interp_end0[:, jj]

            X_interp_end0[:, jj] = X_cs2[:, -1]
            D1_interp_end0[:, jj] = D1_cs2[:, -1]
            D2_interp_end0[:, jj] = D2_cs2[:, -1]
            D3_interp_end0[:, jj] = D3_cs2[:, -1]

            X_snapshots_old_f[:, 0:(nt_c_max + 1), jj] = X_cs2[:, 0:(nt_c_max + 1)]
            D1_snapshots_old_f[:, 0:(nt_c_max + 1), jj] = D1_cs2[:, 0:(nt_c_max + 1)]
            D2_snapshots_old_f[:, 0:(nt_c_max + 1), jj] = D2_cs2[:, 0:(nt_c_max + 1)]
            D3_snapshots_old_f[:, 0:(nt_c_max + 1), jj] = D3_cs2[:, 0:(nt_c_max + 1)]

            Xq = np.reshape(g2_X_diff_tend_c + X_diff_g2_p[:, -1], (3, TotalNumFinePts))
            D1q = np.reshape(g2_D1_diff_tend_c + D1_diff_g2_p[:, -1], (3, TotalNumFinePts))
            D2q = np.reshape(g2_D2_diff_tend_c + D2_diff_g2_p[:, -1], (3, TotalNumFinePts))
            D3q = np.reshape(g2_D3_diff_tend_c + D3_diff_g2_p[:, -1], (3, TotalNumFinePts))

            X_sol_fn[:, :, jj] = X_sol_fn[:, :, jj] + Xq
            D1_sol_fn[:, :, jj] = D1_sol_fn[:, :, jj] + D1q
            D2_sol_fn[:, :, jj] = D2_sol_fn[:, :, jj] + D2q
            D3_sol_fn[:, :, jj] = D3_sol_fn[:, :, jj] + D3q

            # Re-orthogonalization of triads
            for kk in range(TotalNumFinePts):
                d1 = D1_sol_fn[:, kk, jj]
                d2 = D2_sol_fn[:, kk, jj]
                d3 = D3_sol_fn[:, kk, jj]
                d3 = d3 / np.linalg.norm(d3)
                d2 = d2 - np.dot(d2, d3) * d3
                d2 = d2 / np.linalg.norm(d2)
                d1 = d1 - np.dot(d1, d3) * d3 - np.dot(d1, d2) * d2
                d1 = d1 / np.linalg.norm(d1)
                D3_sol_fn[:, kk, jj] = d3
                D1_sol_fn[:, kk, jj] = d1
                D2_sol_fn[:, kk, jj] = d2

            comm.send(X_sol_fn[:, :, jj], dest=1, tag=101)
            comm.send(D1_sol_fn[:, :, jj], dest=1, tag=202)
            comm.send(D2_sol_fn[:, :, jj], dest=1, tag=303)
            comm.send(D3_sol_fn[:, :, jj], dest=1, tag=404)

        comm.Barrier()

    if rank == 0:
        for kk in range(ii + 1, NumTimeInterval):
            X_ic_vec[:, :, kk] = X_sol_fn[:, :, kk - 1]
            D1_ic_vec[:, :, kk] = D1_sol_fn[:, :, kk - 1]
            D2_ic_vec[:, :, kk] = D2_sol_fn[:, :, kk - 1]
            D3_ic_vec[:, :, kk] = D3_sol_fn[:, :, kk - 1]
        

        send_buf_x = np.split(X_ic_vec, NumTimeInterval, axis=2)
        send_buf_d1 = np.split(D1_ic_vec, NumTimeInterval, axis=2)
        send_buf_d2 = np.split(D2_ic_vec, NumTimeInterval, axis=2)
        send_buf_d3 = np.split(D3_ic_vec, NumTimeInterval, axis=2)

        #Print out the true error and the relative true error after each iteration
        print(ii)
        RelErrX = np.linalg.norm(X_sol_fn[:, :, -1] - Xfn[:, :, -1]) / np.linalg.norm(X_sol_fn[:, :, -1])
        AbsErrX = np.linalg.norm(X_sol_fn[:, :, -1] - X_true1)
        AbsErrX1 = AbsErrX/np.linalg.norm(X_true1)
  
        print('RelErr= %.16g, AbsErrX=%.16g, AbsErrX1=%.16g\n' % (RelErrX, AbsErrX, AbsErrX1))


        Xfn[:, :, 1:] = X_sol_fn
        D1fn[:, :, 1:] = D1_sol_fn
        D2fn[:, :, 1:] = D2_sol_fn
        D3fn[:, :, 1:] = D3_sol_fn


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
    
    
    if rank == 0 and ii < NumParSweeps - 1 and nt_c < nt_c_vec[ii + 1]:
        comm.send(X_correct_ic, dest=2, tag=1101)
        comm.send(D1_correct_ic, dest=2, tag=2202)
        comm.send(D2_correct_ic, dest=2, tag=3303)
        comm.send(D3_correct_ic, dest=2, tag=4404)


    if rank == 2 and ii < NumParSweeps - 1 and nt_c < nt_c_vec[ii + 1]:
        X_correct_ic = comm.recv(source=0, tag=1101)
        D1_correct_ic = comm.recv(source=0, tag=2202)
        D2_correct_ic = comm.recv(source=0, tag=3303)
        D3_correct_ic = comm.recv(source=0, tag=4404)

        
    
    comm.Barrier()




end_time = MPI.Wtime()

if rank == 0:
    run_time = end_time - start_time
    print("Number of iterations performed = %d\t: Total time = %.16g\n" %(ii + 1, run_time))


