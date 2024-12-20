IPJDSVD Documentation
******************************

IPJDSVD: Inner preconditioned Jacobi--Davidson SVD method. 
********************************************************************* 

"ipjdsvd()" and "ipjdsvd_hybrid ()" find k singular values of the matrix A closest to a given target 
and/or the corresponding left and right singular vectors using the standard 
extraction version of the inner preconditoned Jacobi-Davidson type 
SVD algorithms and its two stage variant that first works on the cross-product of A  and then switches to the augmented matrices of A, 
such that the computed approximate partial SVD (T,U,V) of A satisfies 
      ||AV-UT||_F^2+||A^TU-VT||_F^2 <= k||A||_2^2*tol^2, 
where the diagonal elements of the diagonal matrix T save the approximate 
singular values of A and the columns of U and V save the corresponding left 
and right singular vectors, respectively. 


License Information
===================

IPJDSVD is licensed under the 3-clause license BSD.  

   Copyright (c) 2018, Tsinghua University and Soochow University. 
   All rights reserved.


Contact Information
===================

For reporting bugs or questions about functionality contact Zhongxiao 
Jia by email, *jiazx* at *tsinghua.edu.cn* or Jinzhi Huang by email 
*jzhuang21* at *suda.edu.cn*.  


Support
===================

* National Science Foundation of China No.12171273

* Youth Fund of the National Science Foundation of China No. 12301485

* Youth Program of the Natural Science Foundation of Jiangsu Province 
  No. BK20220482


Pre-process 
===================
Before using IPJDSVD, add the folder (without subfolders) "IPJDSVD-Master" 
to the MATLAB search path list on the MATLAB homepage. 

 
Implementation information on ipjdsvd, and the same for ipjdsvd_hybrid. 
===================

function [varargout] = ipjdsvd(varargin)

T = IPJDSVD(A) returns 6 largest singular values of A.

T = IPJDSVD(A,K) returns K largest singular values of A.

T = IPJDSVD(A,K,SIGMA) returns K singular values of A depending on SIGMA:

       'largest'   - compute K largest singular values. This is the default.
       'smallest' - compute K smallest singular values.
       numeric   - compute K singular values nearest to SIGMA.

 T = IPJDSVD(A,K,SIGMA,NAME,VALUE) configures additional options specified 
       by one or more name-value pair arguments:

                                 'Tolerance' - Convergence tolerance
                           'MaxIterations' - Maximum number of iterations
          'MaxSubspaceDimension' - Maximum size of subspaces
          'MinSubspaceDimension' - Minimum size of subspaces
                  'SwitchingTolerance' - Switching tolerance for correction equations
                         'InnerTolerance' - Parameter for inner stopping tolerance
                         'LeftStartVector' - Left starting vector
                       'RightStartVector' - Right starting vector
   'InnerPreconditionTolerance1' - 1st parameter for inner preconditioning
   'InnerPreconditionTolerance1' - 2nd parameter for inner preconditioning
                                      'Display' - Display diagnostic messages

 T = IPJDSVD(A,K,SIGMA,OPTIONS) alternatively configures the additional options 
       using a structure. See the documentation below for more information.

[T,U,V] = IPJDSVD(A,...) computes the singular vectors as well. If A is  M-by-N 
       K singular values are computed, then T is K-by-K diagonal, U and V are 
       M-by-K and N-By-K orthonormal, respectively.

[T,U,V,RELRES] = IPJDSVD(A,...) also returns a matrix each column of which 
       contains all the relative residual norms of the approximate singular triplets 
       during computing the relavant desired exact singular triplet.

[T,U,V,RELRES,OUTIT] = IPJDSVD(A,...) also returns a vector of numbers of outer 
       iterations uesd to compute each singular triplet.

[T,U,V,RELRES,OUTIT,INNIT] = IPJDSVD(A,...) also returns a vector of numbers of 
       total inner iterations uesd to compute each singular triplet.

[T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP] = IPJDSVD(A,...) also returns a matrix 
       each of whose column contains the numbers of inner iterations during each 
       outer iteration when compute the relavant singular triplet.

[T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME] = IPJDSVD(A,...) also returns 
       a vector of CPU time used to compute each singular triplet.

[T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME,FLAG] = IPJDSVD(A,...) also 
       returns a convergence flag. If the method has converged, then FLAG = 0; 
       If the maximun number of outer iterations have been used before convergence, 
       then FLAG = 1.

[...] = IPJDSVD(AFUN,MN, ...) accepts function handle AFUN instead of the 
       matrix A. AFUN(X,'notransp') must accept a vector input X and return the 
       matrix-vector product A*X, while AFUN(U,'transp') must return A'*U. MN 
       is a 1-by-2 row vector [M N] where M and N are the numbers of rows and 
       columns of A respectively.

Description for the parameters contained in the structure OPTS.

PARAMETER           DESCRIPTION  

  OPTS.TOL            Convergence tolerance, same as the name 'Tolerance'. 
                              Convergence is determined when 
                                    ||[A^Tu-θv;Av-θu|| <= ||A||·TOL,  
                              where (θ,u,v) is the current approximate singular 
                      	      triplet of A and ||A|| is the 1-norm of A when A is 
                              given as a matrix or estimate 2-norm of A if the 
                              function handle Afun is given.
                              DEFAULT VALUE    TOL = 1e-8 

  OPTS.MAXIT        Maximum number of outer iterations, same as the 
                              parameter 'MaxIterations'.                           
                              DEFAULT VALUE    MAXIT = N 

  OPTS.MAXSIZE    Maximum size of searching subspaces, same as 
                              'MaxSubspaceDimension'.                          
                              DEFAULT VALUE    MAXSIZE = 30

  OPTS.MINSIZE     Minimum size of searching subspaces, same as 
                              'MinSubspaceDimension'.                         
                              DEFAULT VALUE    MINSIZE = 3

  OPTS.FIXTOL       Switching tolerance for the inner correction euqations, 
                              same as the parameter 'SwitchingTolerance'.
                              DEFAULT VALUE    FIXTOL = 1e-4

  OPTS.INNTOL      Accuracy requirement for the apprixmate solution of 
                              the inner correction equations, same as the 
                              'SwitchingTolerance'. 
                              DEFAULT VALUE    FIXTOL = 1e-4  

  OPTS.U0             Left starting vector, same as the parameter 
                             'LeftStartVector'. 
                             DEFAULT VALUE  U0 = randn(M,1);

  OPTS.V0             Right starting vector, same as the parameter 
                             'RightStartVector'. 
                             DEFAULT VALUE  V0 = randn(N,1);

  OPTS.INNPRECTOL1    The first inner preconditioning parameter, same as
                                       the parameter 'InnerPreconditionTolerance1'.
                             DEFAULT VALUE  INNPRECTOL1 = 0.05;

  OPTS.INNPRECTOL2    The second inner preconditioning parameter, same as
                                       the parameter 'InnerPreconditionTolerance2'.
                             DEFAULT VALUE  INNPRECTOL1 = 0.01;

  OPTS.DISPS        Indicates if K approximate singular values are to be 
                             displayed during the computation. Set DISPS > 1 to 
                             display the values at each outer iteration or when 
                             the algorithm is stopped. If 0 < DISPS <= 1, then the 
                             results are only display after the overall convergence.
                             DEFAULT VALUE   DISPS = 0


Test Example
===================
M = 10000; 
sigA = [0.8:0.0001:0.9979,1.0001:0.0001:1.002];
N = size(sigA,2); 
dense = 0.2; 
A = sprand(M,N,dense,sigA);

k = 20;
target = 1;
outtol = 1e-8;
maxit = min(M,N);
maxdim = 30;
mindim = 3;
fixtol = 0;
inntol = 1e-4;
u0 = ones(M,1);
v0 = ones(N,1);

% Tests with matrices;
innprectol1 = 0.05;     innprectol2 = 0.01;
[Tp,Up,Vp,RELRESp,OUTITp,INNITp,INNITPERSTEPp,CPUTIMEp,FLAGp,PERCp] = ...
    ipjdsvd(A,k,target,'Tolerance',outtol,'MaxIterations',maxit,...
    'MaxSubspaceDimension',maxdim,'MinSubspaceDimension',mindim,...
    'SwitchingTolerance',fixtol,'InnerTolerance',inntol,...
    'LeftStartVector',u0,'RightStartVector',v0,...
    'InnerPreconditionTolerance1',innprectol1,...
    'InnerPreconditionTolerance2',innprectol2,'Display',1);
 
innprectol1 = 0;        innprectol2 = 0;
[To,Uo,Vo,RELRESo,OUTITo,INNITo,INNITPERSTEPo,CPUTIMEo,FLAGo,PERCo]=...
    ipjdsvd(A,k,target,'Tolerance',outtol,'MaxIterations',maxit,...
    'MaxSubspaceDimension',maxdim,'MinSubspaceDimension',mindim,...
    'SwitchingTolerance',fixtol,'InnerTolerance',inntol,...
    'LeftStartVector',u0,'RightStartVector',v0,...
    'InnerPreconditionTolerance1',innprectol1,...
    'InnerPreconditionTolerance2',innprectol2,'Display',1);

% Tests with function handles;
afun = MatrixFunction(A);
innprectol1 = 0.05;     innprectol2 = 0.01;
[Tp,Up,Vp,RELRESp,OUTITp,INNITp,INNITPERSTEPp,CPUTIMEp,FLAGp,PERCp]=...
    ipjdsvd(afun,[M N],k,target,'Tolerance',outtol,'MaxIterations',maxit,...
    'MaxSubspaceDimension',maxdim,'MinSubspaceDimension',mindim,...
    'SwitchingTolerance',fixtol,'InnerTolerance',inntol,...
    'LeftStartVector',u0,'RightStartVector',v0,...
    'InnerPreconditionTolerance1',innprectol1,...
    'InnerPreconditionTolerance2',innprectol2,'Display',displayornot);

innprectol1 = 0;        innprectol2 = 0;
[To,Uo,Vo,RELRESo,OUTITo,INNITo,INNITPERSTEPo,CPUTIMEo,FLAGo,PERCo]=...
    ipjdsvd(afun,[M N],k,target,'Tolerance',outtol,'MaxIterations',maxit,...
    'MaxSubspaceDimension',maxdim,'MinSubspaceDimension',mindim,...
    'SwitchingTolerance',fixtol,'InnerTolerance',inntol,...
    'LeftStartVector',u0,'RightStartVector',v0,...
    'InnerPreconditionTolerance1',innprectol1,...
    'InnerPreconditionTolerance2',innprectol2,'Display',displayornot);

% MatrixFunction turns a matrix to a function handle
% For experimental purpose only
function varargout = MatrixFunction(C)
% Creat the function handle with a given matrix C.

varargout{1} = @matrixfun;
    function y = matrixfun(x,transpornot)
        if strcmp(transpornot,'notransp')
            y = C*x;
        elseif strcmp(transpornot,'transp')
            y = C'*x;
        else
            error('wrong input for transpornot');
        end
    end
end
See also: *testrand.m*.

REFERENCES:
 [1] Jinzhi Huang and Zhongxiao Jia, On inner iterations of Jacobi-Davidson
      type methods for large SVD computations, SIAM J. SCI. COMPUT., 41,3 
      (2019), pp. A1574–A1603.

 [2] Jinzhi Huang and Zhongxiao Jia, Preconditioning correction equations 
      in Jacobi--Davidson type methods for computing partial singular value 
      decompositions of large matrices, Numerical Algorithms,  (2024), 26 pages.
 