# Parareal-HODMD
This is the Python implementation of the Parareal-HODMD algorithm, which a parallel-in-time algorithm based on Parareal and High-Order Dynamic Mode Decomposition for solving differential equations.

The repository contains the code for the three numerical examples presented in the paper.


# How To Run
The code is written in Python 3.9. The parameters in the code are chosen in consistent with the paper. 

Example of running the rod swimmer example with 50 cores:

mpirun -np 50 python Parareal_DMD_5ds.py


## RodSwimmerExample
- The folder contains the code for the simulation of a single rod swimmer solved using the Method of Regularized Stokeslets (MRS) with regularization parameter $\epsilon=3\sigma$ and $\epsilon=5\sigma$. The parameters of the examples in this folder are set to run with 50 cores.

## SphereSwimmerExample
- The folder contains the code for the simulation of an elastic sphere in shear flow solved using the Method of Regularized Stokeslets (MRS). The parameters of the examples in this folder are set to run with 50 cores.

## ThinFilmExample
- The folder contains the code for the simulation of thin liquid film on a chemically heterogeneous substrate using the approximate Newton-ADI method. The parameters of the examples in this folder are set to run with 10 cores.

## Credits
- The code for the Method of Regularized Stokeslets is adapted from the code provided by Minghao W. Rostami (SUNY Binghamton).
- The code for the sphere is adapted from the code for the Immersed Boundary Method at https://github.com/ModelingSimulation/IB-MATLAB, which orignates from Prof. Charles Peskin's class: "Advanced Topics In Numerical Analysis: Immersed Boundary Method For Fluid Structure Interaction" in Spring 2019, and coauthored by Guanhua Sun and Tristan Goodwill.
- The code for LU decomposition is adapted from the book "Numerical Recipes in C" by Press et al.
- The code for the thin film example is partly based on the code provided by Thomas P. Witelski (Duke University).

## References
- [1] Cortez, R. 2001 The method of regularized Stokeslets. SIAM J. Sci. Comput.23, 1204-1225.
- [2] Cortez, R., Fauci, L. & Medovikov, A. 2005 The method of regularized Stokeslets in three dimensions: Analysis, validation, and application to helical swimming. Physics of fluids 17, 1-14.
- [3] Ainley, J., Durkin, S., Embid, R., Boindala, P., & Cortez, R. 2008 The method of images for regularized Stokeslets. J. Comput. Phys., 227, 4600-4616.
- [4] Liu, W and Witelski, T.P., 2020 Steady states of thin film droplets on chemically heterogeneous substrates, IMA Journal of Applied Mathematics, Volume 85, Issue 6, December 2020, Pages 980–1020.
- [5] Witelski, T. P. and Bowen, M., 2003 ADI schemes for higher-order nonlinear diffusion equations, Applied Numerical Mathematics 45 (2003): 331-351.
