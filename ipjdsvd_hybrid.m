function varargout = ipjdsvd_hybrid(A,varargin)
% IPJDSVD_HYBRID finds k singular values of the matrix A closest to a given 
% target and/or the corresponding left and right singular vectors using the 
% standard extraction version of the preconditoned inexact Jacobi-Davidson 
% type hybrid SVD algorithm such that the computed approximate partial SVD 
% (T,U,V) of A satisfies
%       ||AV-UT||_F^2+||A^TU-VT||_F^2 <= k||A||_2^2*tol^2,
% where the diagonal elements of the diagonal matrix T save the approximate
% singular values of A and the columns of U and V save the corresponding
% left and right singular vectors, respectively.
%
% T = IPJDSVD_HYBRID(A) returns 6 largest singular values of A.
%
% T = IPJDSVD_HYBRID(A,K) returns K largest singular values of A.
%
% T = IPJDSVD_HYBRID(A,K,SIGMA) returns K singular values of A depending on 
% SIGMA:
%
%        'largest' - compute K largest singular values. This is the default.
%       'smallest' - compute K smallest singular values.
%         numeric  - compute K singular values nearest to SIGMA.
%
% T = IPJDSVD_HYBRID(A,K,SIGMA,NAME,VALUE) configures additional options 
% specified by one or more name-value pair arguments:
%
%                     'Tolerance' - Convergence tolerance
%                 'MaxIterations' - Maximum number of iterations
%          'MaxSubspaceDimension' - Maximum size of subspaces
%          'MinSubspaceDimension' - Minimum size of subspaces
%            'SwitchingTolerance' - Switching tolerance for correction equations
%                'InnerTolerance' - Parameter for inner stopping tolerance
%               'LeftStartVector' - Left starting vector
%              'RightStartVector' - Right starting vector
%   'InnerPreconditionTolerance1' - 1st parameter for inner preconditioning
%   'InnerPreconditionTolerance1' - 2nd parameter for inner preconditioning
%                       'Display' - Display diagnostic messages
%
% T = IPJDSVD_HYBRID(A,K,SIGMA,OPTIONS) alternatively configures the additional
% options using a structure. See the documentation below for more information.
%
% [T,U,V] = IPJDSVD_HYBRID(A,...) computes the singular vectors as well. 
% If A is M-by-N K singular values are computed, then T is K-by-K diagonal, 
% U and V are M-by-K and N-By-K orthonormal, respectively.
%
% [T,U,V,RELRES] = IPJDSVD_HYBRID(A,...) also returns a matrix each column 
% of which contains all the relative residual norms of the approximate singular
% triplets during computing the relavant desired exact singular triplet.
%
% [T,U,V,RELRES,OUTIT] = IPJDSVD_HYBRID(A,...) also returns a vector of 
% numbers of outer iterations uesd to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT] = IPJDSVD_HYBRID(A,...) also returns a vector 
% of numbers of total inner iterations uesd to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP] = IPJDSVD_HYBRID(A,...) also 
% returns a matrix each of whose column contains the numbers of inner 
% iterations during each outer iteration when compute the relavant 
% singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME] = IPJDSVD_HYBRID(A,...) 
% also returns a vector of CPU time used to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME,FLAG] = IPJDSVD_HYBRID(A,...)
% also returns a convergence flag. If the method has converged, then
% FLAG = 0; If the maximun number of outer iterations have been used before
% convergence, then FLAG = 1.
%
% [...] = IPJDSVD_HYBRID(AFUN,MN, ...) accepts function handle AFUN instead 
% of the matrix A. AFUN(X,'notransp') must accept a vector input X and return
% the matrix-vector product A*X, while AFUN(U,'transp') must return A'*U.
% MN is a 1-by-2 row vector [M N] where M and N are the numbers of rows and
% columns of A respectively.
%
% Description for the parameters contained in the structure OPTS.
%
%  PARAMETER           DESCRIPTION
%
%  OPTS.TOL            Convergence tolerance, same as the name 'Tolerance'.
%                      Convergence is determined when
%                               ||[A^Tu-θv;Av-θu|| <= ||A||·TOL,
%                      where (θ,u,v) is the current approximate singular
%                      triplet of A and ||A|| is the 2-norm of A.
%                      DEFAULT VALUE    TOL = 1e-8.
%
%  OPTS.MAXIT          Maximum number of outer iterations, same as the
%                      parameter 'MaxIterations'.
%                      DEFAULT VALUE    MAXIT = N.
%
%  OPTS.MAXSIZE        Maximum size of searching subspaces, same as
%                      'MaxSubspaceDimension'.
%                      DEFAULT VALUE    MAXSIZE = 30.
%
%  OPTS.MINSIZE        Minimum size of searching subspaces, same as
%                      'MinSubspaceDimension'.
%                      DEFAULT VALUE    MINSIZE = 3.
%
%  OPTS.FIXTOL         Switching tolerance for the inner correction euqations,
%                      same as the parameter 'SwitchingTolerance'.
%                      DEFAULT VALUE    FIXTOL = 1e-4.
%
%  OPTS.INNTOL         Accuracy requirement for the apprixmate solution of
%                      the inner correction equations, same as the
%                      'SwitchingTolerance'.
%                      DEFAULT VALUE    FIXTOL = 1e-4.
%
%  OPTS.U0             Left starting vector, same as the parameter
%                      'LeftStartVector'.
%                      DEFAULT VALUE  U0 = randn(M,1).
%
%  OPTS.V0             Right starting vector, same as the parameter
%                      'RightStartVector'.
%                      DEFAULT VALUE  V0 = randn(N,1).
%
%  OPTS.INNPRECTOL1    The first inner preconditioning parameter, same as
%                      the parameter 'InnerPreconditionTolerance1'.
%                      DEFAULT VALUE  INNPRECTOL1 = 0.05.
%
%  OPTS.INNPRECTOL2    The second inner preconditioning parameter, same as
%                      the parameter 'InnerPreconditionTolerance2'.
%                      DEFAULT VALUE  INNPRECTOL1 = 0.01.
%
%  OPTS.DISPS          Indicates if K approximate singular values are to be
%                      displayed during the computation. Set DISPS > 1 to
%                      display the values at each outer iteration or when
%                      the algorithm is stopped. If 0 < DISPS <= 1, then the
%                      results are only display after the overall convergence.
%                      DEFAULT VALUE   DISPS = 0.
%
% REFERENCES:
% [1] Jinzhi Huang and Zhongxiao Jia, On inner iterations of Jacobi-Davidson
%     type methods for large SVD computations, SIAM J. SCI. COMPUT., 41,3
%     (2019), pp. A1574–A1603.
% [2] Jinzhi Huang and Zhongxiao Jia, Preconditioning correction equations
%     in Jacobi--Davidson type methods for computing partial singular value
%     decompositions of large matrices, (2024), 26 pages.

% Check the number of outputs.
if nargout == 2  || nargout >= 11
    error('Incorrect number of output arguments.');
end
% Initialize a dedicated randstream, to make output reproducible.
randStr = RandStream('dsfmt19937','Seed',0);
% Get inputs and check the items.
VarIn = varargin;
[A,M,N,k,target,u0,v0,Options] = tools.chekInputs(A,VarIn,randStr);
% Reset the stream.
reset(randStr,1);
% Use the function handles if given, otherwise build them from the matrices.
if isa(A,'function_handle')
    Afun = A;
    if ~isfield(Options,'normAest') 
        normAest = svds(Afun,[M,N],1,"largest","Tolerance",1e-2,"MaxIterations",30);
        Options.normAest = normAest;
    end
    if M < N
        AfunT = tools.FunctionT(Afun);
    end
elseif ismatrix(A)
    Afun = tools.MatrixFunction(A);
    if ~isfield(Options,'normAest') 
        normAest = sqrt(norm(A,1)*norm(A,'inf'));
        Options.normAest = normAest;
    end  
    if M < N
        AfunT = tools.MatrixFunction(A');
    end
end
% If target = 0, replace the left and right initial vectors u0 and v0 as
% A^Tu0 and Av0, respectively, in order to discard the information on the
% zero singular value of A and nontrivial singular triplets corresponding
% to nonzero singular values of A are computed.
if (isscalar(target) && target == 0) || (isstring(target) && strcmp(target,'smallest'))
    if ~isfield(Options,'u0') && ~isfield(Options,'v0')
        ut = Afun(v0,'notransp');
        v0 = Afun(u0,'transp');
        u0 = ut;
    end
end
% Fill the fields u0 and v0 in Options if it is empty.
if ~isfield(Options,'u0')
    Options.u0 = u0;
end
if ~isfield(Options,'v0')
    Options.v0 = v0;
end
%--------------------------------%
% BEGIN: CHECK FOR EMPTY MATRIX. %
%--------------------------------%
if M==0 || N==0
    % Nothing need to been done so no singular values to be displayed.
    if nargout == 0
        fprintf('\nAlgorithm: the Standard IPJDSVD_HYBRID algorithm;\n');
        fprintf('Nothing need to be done since the given matrices is empty');
        fprintf('Total number of outer iterations = %d;\n',0);
        fprintf('Total number of inner iterations = %d;\n',0);
        fprintf('Total CPU time in seconds = %s;\n\n',0);
    end
    % Empty vector for computed singular values.
    if nargout == 1
        varargout{1} = [];
    end
    % Empty matrices for computed partial SVD.
    if nargout > 1
        varargout{1} = [];
        varargout{2} = [];
        varargout{3} = [];
    end
    % Empty matrix for the relative residual norms.
    if nargout >= 4
        varargout{4} = [];
    end
    % Empty vector for numbers of outer iterations.
    if nargout >= 5
        varargout{5}= [];
    end
    % Empty vector for numbers of inner iterations.
    if nargout >= 6
        varargout{6} = [];
    end
    % Empty matrix for numbers of inner iterations.
    if nargout >= 7
        varargout{7} = [];
    end
    % Emtpy vector for CPU time.
    if nargout >= 8
        varargout{8} = [];
    end
    % The flag is set as zero for empty matrix.
    if nargout >= 9
        varargout{9} = 0;
    end
    % The percentage of inner convergence is set 1 for empty matrix.
    if nargout >= 10
        varargout{10} = 1;
    end
    return;
end
%------------------------------%
% END: CHECK FOR EMPTY MATRIX. %
%------------------------------% 
% % Implement the Standard IPJDSVD method based on the eigenproblem of the 
% cross product matrix A^TA at the first stage, with a relatively large 
% outer stopping tolerance
if M >= N
    Options_cp = Options;
    [T_convecp,U_convecp,V_convecp,RelatResiMatcp,OuterIterVeccp,InnerIterVeccp,...
        InnerIterMatcp,CPUTimeseVeccp,FLAGofIPJDSVDcp,PercentConvcp] = ...
        ipjdsvd_cp(Afun,[M N],k,target,Options_cp);
else
    Options_cp = Options;
    Options_cp.u0 = Options.v0;
    Options_cp.v0 = Options.u0;
    [T_convecp,V_convecp,U_convecp,RelatResiMatcp,OuterIterVeccp,InnerIterVeccp,...
        InnerIterMatcp,CPUTimeseVeccp,FLAGofIPJDSVDcp,PercentConvcp] = ...
        ipjdsvd_cp(AfunT,[N M],k,target,Options_cp);
end
k_convecp = size(T_convecp,1) - FLAGofIPJDSVDcp;
if strcmp(target,'largest') && k_convecp >= 1
    Options.normAest = T_convecp(1,1);
end
% % If the total outer stopping tolerance is small, use the results
% computed at the first stage as the initial left and right searching
% subspaces for the second stage and perform the Standard IPJDSVD method
% based on the eigenproblem of the augmented matrix [0 A;A^T 0] yet with
% the left and right searching subspaces dealt separately.
Options_am = Options;
Options_am.u0 = U_convecp;
Options_am.v0 = V_convecp;
[T_conveam,U_conveam,V_conveam,RelatResiMatam,OuterIterVecam,InnerIterVecam,...
    InnerIterMatam,CPUTimeseVecam,FLAGofIPJDSVDam,PercentConvam] = ...
    ipjdsvd_am(Afun,[M N],k,target,Options_am);
k_conveam = size(T_conveam,1) - FLAGofIPJDSVDam;
% Handle the outputs.
RelatResiMat = [RelatResiMatcp zeros(size(RelatResiMatcp,1),max(k_conveam-k_convecp,0));...
                RelatResiMatam zeros(size(RelatResiMatam,1),max(k_convecp-k_conveam,0))];
InnerIterMat = [InnerIterMatcp zeros(size(InnerIterMatcp,1),max(k_conveam-k_convecp,0));...
                InnerIterMatam zeros(size(InnerIterMatam,1),max(k_convecp-k_conveam,0))];
OuterIterVec = [[OuterIterVeccp; zeros(max(k_conveam-k_convecp,0),1)] ...
                [OuterIterVecam; zeros(max(k_conveam-k_convecp,0),1)]];
InnerIterVec = [[InnerIterVeccp; zeros(max(k_conveam-k_convecp,0),1)] ...
                [InnerIterVecam; zeros(max(k_conveam-k_convecp,0),1)]];
CPUTimeseVec = [[CPUTimeseVeccp; zeros(max(k_conveam-k_convecp,0),1)] ... 
                [CPUTimeseVecam; zeros(max(k_conveam-k_convecp,0),1)]];
FLAGofIPJDSVD = FLAGofIPJDSVDam;
if sum(OuterIterVecam) == 0
    PercentConv = PercentConvcp;
else
    PercentConv = (PercentConvcp*sum(OuterIterVeccp)+...
        PercentConvam*sum(OuterIterVecam))/sum(sum(OuterIterVec));
end
% Compute the results when needed.
if nargout == 0 || Options.disp > 0
    k_result = length(OuterIterVecam); 
    Results=zeros(k_result,9);
    Results(:,1) = (1:k_result)';
    Results(:,2) = diag(T_conveam(1:k_result,1:k_result));
    for j=1:k_result
        Results(j,3) = min(nonzeros(RelatResiMat(:,j)));
    end
    Results(:,4:5) = OuterIterVec(1:k_result,:);
    Results(:,6:7) = InnerIterVec(1:k_result,:);
    Results(:,8:9) = CPUTimeseVec(1:k_result,:);
    k_conve = sum(Results(:,3) <= Options.tol);
end
%------------------------%
% BEGIN: OUTPUT RESULTS. %
%------------------------%
% Output option I: Display the singular values.
if nargout == 0 || Options.disp > 0
    fprintf('\nSize of A: %d×%d; \n',M,N);
    fprintf('Number of desired singular triplets: %d;\n',k);
    fprintf('Target of desired singular triplets: %7.5e;\n',target);
    fprintf('Maximum number of outer iterations: %d;\n',Options.maxit);
    fprintf('Maximum dimension of searching subspaces: %d;\n',Options.maxsize);
    fprintf('Minimum dimension of searching subspaces: %d;\n',Options.minsize);
    fprintf('Switching tolerance of the correction equations: %7.5e;\n',Options.fixtol);
    fprintf('Stopping tolerance of the inner iterations: %7.5e;\n',Options.inntol);
    fprintf('Algorithm: the Standard IPJDSVD_HYBRID algorithm;\n');
    fprintf('Number of converged singular triplets: %d;\n',k_conve);
    fprintf('  i:     SingularValues;     RelResNorms;    out1-its;    out2-its;    inn1-its;    inn2-its;     CPUtime1;     CPUtime2; \n')
    fprintf('%3d:     %14.5e;     %11.5e;     %7d;     %7d;     %7d;     %7d;     %7.2e;     %7.2e;\n',Results');
    fprintf('Total number of outer iterations = %d;\n',sum(OuterIterVec));
    fprintf('Total number of inner iterations = %d;\n',sum(InnerIterVec));
    fprintf('Total CPU time in seconds = %s;\n',sum(CPUTimeseVec));
    fprintf('Percentage of converged outer iterations = %4.2f%%;\n',sum(PercentConv*100));
end
% Output option II: Output the singualr values.
if nargout == 1
    varargout{1} = diag(T_conveam);
end
% Output option III: Output the partial GSVD.
if nargout > 1
    varargout{1} = T_conveam;
    varargout{2} = U_conveam;
    varargout{3} = V_conveam;
end
% Output option IV: Output the relative residual norm of the approximate 
% singular triplets.
if nargout >= 4
    varargout{4} = RelatResiMat;
end
% Output option V: Output the numbers of outer iterations the algorithm 
% takes to computed each approximate singular triplet.
if nargout >= 5
    varargout{5} = OuterIterVec;
end
% Output option VI: Output the numbers of total inner iterations the 
% algorithm takes to computed each approximate singular triplet.
if nargout >= 6
    varargout{6} = InnerIterVec;
end
% Output option VII: Output the numbers of inner iterations during each 
% outer iteration.
if nargout >= 7
    varargout{7} = InnerIterMat;
end
% Output option VIII: Output the total CPU time the algorithm takes to 
% compute each approximate singular triplet of (A,B).
if nargout >= 8
    varargout{8} = CPUTimeseVec;
end
% Output Options IX: Oupput the flag: flag=0 if all the desired singular 
% triplets have been successfully computed; flag=1 if the maximum number 
% of outer iterations have been used.
if nargout >= 9
    varargout{9} = FLAGofIPJDSVD;
end
% Output Options X: Oupput the percentage of total outer iterations during
% which the inner iterations of solving the correction equation have
% converged to the desired inner accuracy.
if nargout >= 10
    varargout{10} = PercentConv;
end
%----------------------%
% END: OUTPUT RESULTS. %
%----------------------%
end     