function varargout = ipjdsvd(A,varargin)
% IPJDSVD finds k singular values of the matrix A closest to a given target 
% and/or the corresponding left and right singular vectors using the standard 
% extraction version of the preconditoned inexact Jacobi-Davidson type SVD 
% algorithm such that the computed approximate partial SVD (T,U,V) of A satisfies 
%       ||AV-UT||_F^2+||A^TU-VT||_F^2 <= k||A||_2^2*tol^2,
% where the diagonal elements of the diagonal matrix T save the approximate
% singular values of A and the columns of U and V save the corresponding
% left and right singular vectors, respectively. 
%
% T = IPJDSVD(A) returns 6 largest singular values of A.
%
% T = IPJDSVD(A,K) returns K largest singular values of A.
%
% T = IPJDSVD(A,K,SIGMA) returns K singular values of A depending on SIGMA:
%
%        'largest' - compute K largest singular values. This is the default.
%       'smallest' - compute K smallest singular values.
%         numeric  - compute K singular values nearest to SIGMA.
%
% T = IPJDSVD(A,K,SIGMA,NAME,VALUE) configures additional options specified
% by one or more name-value pair arguments:
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
% T = IPJDSVD(A,K,SIGMA,OPTIONS) alternatively configures the additional 
% options using a structure. See the documentation below for more information.
%
% [T,U,V] = IPJDSVD(A,...) computes the singular vectors as well. If A is 
% M-by-N K singular values are computed, then T is K-by-K diagonal, U and V 
% are M-by-K and N-By-K orthonormal, respectively.
%
% [T,U,V,RELRES] = IPJDSVD(A,...) also returns a matrix each column of 
% which contains all the relative residual norms of the approximate singular
% triplets during computing the relavant desired exact singular triplet.
%
% [T,U,V,RELRES,OUTIT] = IPJDSVD(A,...) also returns a vector of numbers 
% of outer iterations uesd to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT] = IPJDSVD(A,...) also returns a vector of 
% numbers of total inner iterations uesd to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP] = IPJDSVD(A,...) also returns a 
% matrix each of whose column contains the numbers of inner iterations during 
% each outer iteration when compute the relavant singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME] = IPJDSVD(A,...) also 
% returns a vector of CPU time used to compute each singular triplet.
%
% [T,U,V,RELRES,OUTIT,INNIT,INNITPERSTEP,CPUTIME,FLAG] = IPJDSVD(A,...) 
% also returns a convergence flag. If the method has converged, then 
% FLAG = 0; If the maximun number of outer iterations have been used before 
% convergence, then FLAG = 1.
%
% [...] = IPJDSVD(AFUN,MN, ...) accepts function handle AFUN instead of 
% the matrix A. AFUN(X,'notransp') must accept a vector input X and return 
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
%                      triplet of A and ||A|| is the 1-norms of A is A is 
%                      given as a matrix or estimate 2-norm of A if the 
%                      function handle Afun is given.
%                      DEFAULT VALUE    TOL = 1e-8 
%
%  OPTS.MAXIT          Maximum number of outer iterations, same as the 
%                      parameter 'MaxIterations'.                           
%                      DEFAULT VALUE    MAXIT = N 
%
%  OPTS.MAXSIZE        Maximum size of searching subspaces, same as 
%                      'MaxSubspaceDimension'.                          
%                      DEFAULT VALUE    MAXSIZE = 30
%
%  OPTS.MINSIZE        Minimum size of searching subspaces, same as 
%                      'MinSubspaceDimension'.                         
%                      DEFAULT VALUE    MINSIZE = 3
%
%  OPTS.FIXTOL         Switching tolerance for the inner correction euqations, 
%                      same as the parameter 'SwitchingTolerance'.
%                      DEFAULT VALUE    FIXTOL = 1e-4
%
%  OPTS.INNTOL         Accuracy requirement for the apprixmate solution of 
%                      the inner correction equations, same as the 
%                      'SwitchingTolerance'. 
%                      DEFAULT VALUE    FIXTOL = 1e-4  
%
%  OPTS.U0             Left starting vector, same as the parameter 
%                      'LeftStartVector'. 
%                      DEFAULT VALUE  U0 = randn(M,1);
%
%  OPTS.V0             Right starting vector, same as the parameter 
%                      'RightStartVector'. 
%                      DEFAULT VALUE  V0 = randn(N,1);
%
%  OPTS.INNPRECTOL1    The first inner preconditioning parameter, same as
%                      the parameter 'InnerPreconditionTolerance1'.
%                      DEFAULT VALUE  INNPRECTOL1 = 0.05;
%
%  OPTS.INNPRECTOL2    The second inner preconditioning parameter, same as
%                      the parameter 'InnerPreconditionTolerance2'.
%                      DEFAULT VALUE  INNPRECTOL1 = 0.01;
%
%  OPTS.DISPS          Indicates if K approximate singular values are to be 
%                      displayed during the computation. Set DISPS > 1 to 
%                      display the values at each outer iteration or when 
%                      the algorithm is stopped. If 0 < DISPS <= 1, then the 
%                      results are only display after the overall convergence.
%                      DEFAULT VALUE   DISPS = 0
%
% REFERENCES:
% [1] Jinzhi Huang and Zhongxiao Jia, On inner iterations of Jacobi-Davidson
%     type methods for large SVD computations, SIAM J. SCI. COMPUT., 41,3 (2019), 
%     pp. A1574–A1603.
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
[A,M,N,k,target,u0,v0,Options] = chekInputs(A,VarIn,randStr);

% Reset the stream.
reset(randStr,1);

%klanczos = 30;

% Use the function handles if given, otherwise build them from the matrices.
if isa(A,'function_handle')
    Afun = A;
    %normAest = svds(@Afun,[M,N],1,'largest','MaxIterations',klanczos,'Tolerance',1e-3);
    normAest = tools.normestimation(A,M,N);
elseif ismatrix(A)
    Afun = MatrixFunction(A);
    %normAest = svds(A,1,'largest','MaxIterations',klanczos,'Tolerance',1e-3); 
    normAest = sqrt(norm(A,1)*norm(A,"inf"));
end

%--------------------------------%
% BEGIN: CHECK FOR EMPTY MATRIX. %
%--------------------------------%
if M==0 || N==0
    % Nothing need to been done so no singular values to be displayed.
    if nargout == 0
        fprintf('\nAlgorithm: the Standard IPJDSVD algorithm;\n');
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

%-------------------------------------------------%
% BEGIN: DISPLAY THE KEY INFORMATION IF REQUIRED. %
%-------------------------------------------------%
if Options.disp > 1
    fprintf('Size of A: %d×%d; \n',M,N);
    fprintf('Number of desired SVD singular triplets: %d;\n',k);
    fprintf('Target of desired SVD singular triplets: %7.5e;\n',target);
    fprintf('Maximum number of outer iterations: %d;\n',Options.maxit);
    fprintf('Maximum dimension of searching subspaces: %d;\n',Options.maxsize);
    fprintf('Minimum dimension of searching subspaces: %d;\n',Options.minsize);
    fprintf('Switching tolerance of the correction equations: %7.5e;\n',Options.fixtol);
    fprintf('Stopping tolerance of the inner iterations: %7.5e;\n',Options.inntol);
    fprintf('Algorithm: the Standard IPJDSVD algorithm;\n');
end
%-----------------------------------------------%
% END: DISPLAY THE KEY INFORMATION IF REQUIRED. %
%-----------------------------------------------%

%-----------------------------------------------------------%
% BEGIN: DESCRIPTION AND INITIALIZATION OF LOCAL VARIABLES. %
%-----------------------------------------------------------%
% Initialization and description of local variables.

% Converged partial SVD of A that is desired by the users.
T_conve = zeros(k,k);   
U_conve = zeros(M,k);   
V_conve = zeros(N,k);

% Computational results recorded to illstrate the performence of the algorithm.
RelatResiMat = [];
InnerIterMat = [];
OuterIterVec = zeros(k,1);
InnerIterVec = zeros(k,1);    
CPUTimeseVec = zeros(k,1);

% Auxiliary items to help recording performance.
RelatResiAccumul=[];        
InnerIterAccumul=[];
OuterIterCounter=0;         
InnerIterCounter=0;

% Intermediate matrices to extract approximate singular triplets.
U = zeros(M,Options.maxsize);
V = zeros(N,Options.maxsize);
AV = zeros(M,Options.maxsize);
ATU = zeros(N,Options.maxsize);
H =zeros(Options.maxsize,Options.maxsize);
%---------------------------------------------------------%
% END: DESCRIPTION AND INITIALIZATION OF LOCAL VARIABLES. %
%---------------------------------------------------------%

tic;

% Number of converged singular triplets.
k_conve = 0;

% Record the size of the searching subspaces.
m = 0;

% Count the overall outer iterations.
OuterIter = 0;

% Count the overall convergence of inner iterations.
InnerConv = 0;

% Normalize the starting left and right vectors.
uplus = tools.normalizing(u0);
vplus = tools.normalizing(v0);

%-----------------------------%
% BEGIN: MAIN ITERATION LOOP. %
%-----------------------------%

while k_conve < k && OuterIter < Options.maxit
    m = m+1;
    
    %-------------------------------%
    % BEGIN: Updating the matrices. %
    %-------------------------------%
    U(:,m) = uplus;     
    V(:,m) = vplus;     
    ATU(:,m) = Afun(uplus,'transp');
    AV(:,m) = Afun(vplus,'notransp');
    H(1:m,m) = ATU(:,1:m)'*V(:,m);
    H(m,1:m-1) = U(:,m)'*AV(:,1:m-1);
    
    %-----------------------------%
    % END: Updating the matrices. %
    %-----------------------------%
    
    %--------------------------------------------------------%
    % BEGIN: SEARCH AND EXTRACT APPROXIMATE SINGULAR VALUES. %
    %--------------------------------------------------------%
    search=1;
    while search
        % Compute the SVD of the small projected matrix.
        [Theta,Ctemp,Dtemp] = StandardExtraction(H,target,m);       

        % Compute the approximate singular triplet and the corresponding residual.
        theta = Theta(1,1);
        ufound = U(:,1:m)*Ctemp(:,1);
        vfound = V(:,1:m)*Dtemp(:,1);
        router = [AV(:,1:m)*Dtemp(:,1)-theta*ufound;...
                  ATU(:,1:m)*Ctemp(:,1)-theta*vfound];
        
        % Compute the relative residual norm and add it to the auxiliary item
        % RelatResiAccumel which accumulately stores the relative residual
        % norms of every step of extraction.
        RelResidual = norm(router)/normAest;
        RelatResiAccumul = [RelatResiAccumul;RelResidual];
        
        % Check for convergence.
        if RelResidual <= Options.tol
            % Update the number of converged singular triplets
            k_conve=k_conve+1;
            
            % Update the computational results.
            CPUTimeseVec(k_conve) = toc;
            RelatResiMat = tools.addvec(RelatResiMat,RelatResiAccumul);
            OuterIterVec(k_conve) = OuterIterCounter;
            InnerIterMat = tools.addvec(InnerIterMat,InnerIterAccumul);
            InnerIterVec(k_conve) = InnerIterCounter;
            
            % Update the converged partial SVD.
            T_conve(k_conve,k_conve) = theta;
            U_conve(:,k_conve) = ufound;
            V_conve(:,k_conve) = vfound;
            
            % If all desired singular triplets have converged, then break the
            % loop; else discard the newly converged one from the current
            % searching subspace and restart the algorithm.
            if k_conve == k
                break;
            else
                % Deflat and restart the algorithm efficiently.
                [U,V,ATU,AV,H]=StandardDeflation(U,V,ATU,AV,Ctemp,Dtemp,Theta,m);
                m=m-1;
                
                % Reset the auxiliary items and the timer.
                RelatResiAccumul = [];        
                InnerIterAccumul = [];
                OuterIterCounter = 0;         
                InnerIterCounter = 0;
                tic;
            end
        else
            search = 0;
        end
        
        %-------------------------------------------------%
        % BEGIN: DISPLAY THE KEY INFORMATION IF REQUIRED. %
        %-------------------------------------------------%
        if Options.disp > 1
            fprintf('Number of Converged singular triplets: %d;\n',k_conve);
            if k_conve >= 1
                ConvergedSVD = [[1:1:k_conve]' (diag(T_conve(1:k_conve,1:k_conve)))]';
                fprintf('Converged singular values:\n %d %15.5e;\n',ConvergedSVD);
            end
            fprintf('Relativel residual of the next approximate singular triplet: %s;\n',RelResidual);
        end
        %-----------------------------------------------%
        % END: DISPLAY THE KEY INFORMATION IF REQUIRED. %
        %-----------------------------------------------%
    end
    %------------------------------------------------------%
    % END: SEARCH AND EXTRACT APPROXIMATE SINGULAR VALUES. %
    %------------------------------------------------------%
    
    % Determine the parameter targetin to be used for the correction equation
    targetin = GetInnerTarget(Theta,target,RelResidual,Options);
    
    % Determine the indexes of the singular vectors to be exploited in the
    % correction equations for the first time.
    iexten = StandardExtendIndexI(Theta,targetin,Options,m);

    if isempty(iexten)
        Uexten = [];
        Vexten = [];
    else
        % Compute the singular triplets associated with the extending singular
        % vectors as well as the residual norms.
        Texten = Theta(iexten,iexten);
        Cexten = Ctemp(:,iexten);
        Dexten = Dtemp(:,iexten);
        Uexten = U(:,1:m)*Cexten;
        Vexten = V(:,1:m)*Dexten;
        Rexten = [AV(:,1:m)*Dexten-Uexten*Texten;...
            ATU(:,1:m)*Cexten-Vexten*Texten];
        Resexten = vecnorm(Rexten);

        % Determine the indexes of the singular vectors to be finally exploited
        % in the correction equations.
        jexten = StandardExtendIndexII(Resexten,normAest,Options);

        % Extract the final singular vectors to be used to precondition the
        % correction equations.
        iexten = iexten(jexten);
        Uexten = Uexten(:,jexten);    
        Vexten = Vexten(:,jexten);
    end

    % Determine the stopping cretiria epsilonin for inner iterations.
    epsilonin = StandardParamterN(Theta,targetin,iexten,Options,m);

    if Options.exact > 0
        % Directly solve the correction equation, for only experimental purpose.
        [s,t,iter,~,inflag] = tools.InnerExactSolverPreJDSVD(A,router,...
            U_conve(:,1:k_conve),V_conve(:,1:k_conve),ufound,vfound,Uexten,Vexten,targetin);
    else
        % Iteratively solve the correction equations approximately.
        [s,t,iter,~,inflag] = tools.InnerSolverPreJDSVD(Afun,[M N],router,...
            U_conve(:,1:k_conve),V_conve(:,1:k_conve),ufound,vfound,...
            Uexten,Vexten,targetin,epsilonin);
    end

    % Update the computational results.
    OuterIter = OuterIter+1;
    InnerConv = InnerConv+(inflag==0);
    OuterIterCounter = OuterIterCounter+1;
    InnerIterCounter = InnerIterCounter+iter;
    InnerIterAccumul = [InnerIterAccumul;iter];
       
    % If the dimension of the searching subspaces reaches the given maximum
    % number, then efficiently restart the algorithm. 
    if m == Options.maxsize
        [U,V,ATU,AV,H,m] = StandardRestart(U,V,ATU,AV,Ctemp,Dtemp,Theta,...
            m,Options.minsize,iexten);
    end
    
    % Orthogonalize the approximate solution of the correction equation to
    % the orthonormal basis matrix of the left and right searching subspaces.
    uplus = tools.orthog(U(:,1:m),s);
    vplus = tools.orthog(V(:,1:m),t);
end

%---------------------------%
% END: MAIN ITERATION LOOP. %
%---------------------------%

% Restore the results about the last approximate singular triplet if the
% maximun number of outerations have been used.

if OuterIter >= Options.maxit
    % Set the flag as 1.
    FLAGofIPJDSVD = 1;
    
    % Update the number of converged singular triplets temporarily.
    k_conve=k_conve+1;
    
    % Update the computational results.
    CPUTimeseVec(k_conve) = toc;
    RelatResiMat = tools.addvec(RelatResiMat,RelatResiAccumul);
    OuterIterVec(k_conve) = OuterIterCounter;
    InnerIterMat = tools.addvec(InnerIterMat,InnerIterAccumul);
    InnerIterVec(k_conve) = InnerIterCounter;
    
    % Update the converged partial GSVD.
    T_conve(k_conve,k_conve) = theta;      
    U_conve(:,k_conve) = ufound;          
    V_conve(:,k_conve) = vfound;  
    
    % Reset the number of converged singular triplets. 
    k_conve=k_conve-1;
    
    % Warn the users.
    warning('Maximum number of outer iterations have been used before computing all the desired singular triplets')
else 
   FLAGofIPJDSVD = 0;
end

% Compute the results when needed.
if nargout == 0 || Options.disp > 0
    if FLAGofIPJDSVD==1
        k_result = k_conve+1;
    elseif FLAGofIPJDSVD==0
        k_result = k_conve;
    end 
    Results=zeros(k_result,6);  
    Results(:,1) = (1:k_result)';
    Results(:,2) = diag(T_conve(1:k_result,1:k_result));
    for j=1:k_result
        Results(j,3) = min(nonzeros(RelatResiMat(:,j)));
    end
    Results(:,4) = OuterIterVec(1:k_result,:);
    Results(:,5) = InnerIterVec(1:k_result,:);
    Results(:,6) = CPUTimeseVec(1:k_result,:);
end

%------------------------%
% BEGIN: OUTPUT RESULTS. %
%------------------------%

% Output option I: Display the singular values.
if nargout == 0 || Options.disp > 0
    fprintf('\n\nSize of A: %d×%d; \n',M,N);
    fprintf('Number of desired singular triplets: %d;\n',k);
    fprintf('Target of desired singular triplets: %7.5e;\n',target);
    fprintf('Maximum number of outer iterations: %d;\n',Options.maxit);
    fprintf('Maximum dimension of searching subspaces: %d;\n',Options.maxsize);
    fprintf('Minimum dimension of searching subspaces: %d;\n',Options.minsize);
    fprintf('Switching tolerance of the correction equations: %7.5e;\n',Options.fixtol);
    fprintf('Stopping tolerance of the inner iterations: %7.5e;\n',Options.inntol);
    fprintf('Algorithm: the Standard IPJDSVD algorithm;\n');
    fprintf('Number of converged singular triplets: %d;\n',k_conve);
    fprintf('  i:     SingularValues;     RelResNorms;     out-its;     inn-its;     CPUtime; \n')
    fprintf('%3d:     %14.5e;     %11.5e;     %7d;     %7d;     %7.5e;\n',Results');
    fprintf('Total number of outer iterations = %d;\n',sum(OuterIterVec));
    fprintf('Total number of inner iterations = %d;\n',sum(InnerIterVec));
    fprintf('Total CPU time in seconds = %s;\n',sum(CPUTimeseVec)); 
    fprintf('Percentage of converged outer iterations = %4.2f%%;\n\n',sum(InnerConv/OuterIter*100)); 
end

% Output option II: Output the singualr values.
if nargout == 1
    varargout{1}=diag(T_conve);
end

% Output option III: Output the partial GSVD.
if nargout > 1
varargout{1} = T_conve;   
varargout{2} = U_conve;
varargout{3} = V_conve;  
end

% Output option IV: Output the relative residual norm of the approximate singular triplets. 
if nargout >= 4
    varargout{4} = RelatResiMat;
end

% Output option V: Output the numbers of outer iterations the algorithm takes 
% to computed each approximate singular triplet. 
if nargout >= 5
    varargout{5} = OuterIterVec;
end

% Output option VI: Output the numbers of total inner iterations the algorithm 
% takes to computed each approximate singular triplet.
if nargout >= 6
    varargout{6} = InnerIterVec;
end

% Output option VII: Output the numbers of inner iterations during each outer
% iteration. 
if nargout >= 7
    varargout{7} = InnerIterMat;
end

% Output option VIII: Output the total CPU time the algorithm takes to compute
% each approximate singular triplet of (A,B).
if nargout >= 8
    varargout{8} = CPUTimeseVec;
end

% Output Options IX: Oupput the flag: flag=0 if all the desired singular triplets 
% have been successfully computed; flag=1 if the maximum number of outer 
% iterations have been used. 
if nargout >= 9
    varargout{9} = FLAGofIPJDSVD;
end

if nargout >= 10
    varargout{10} = InnerConv/OuterIter;
end

%----------------------%
% END: OUTPUT RESULTS. %
%----------------------%
end

function [A,M,N,k,target,u0,v0,Options] = chekInputs(A,VarIn,randStr)
% Get the inputs and do check errors.

% Get A and M, N.
[M,N,VarIn] = getSizeA(A,VarIn);

% Get the number k of desired singular triplets.
if numel(VarIn)<1
    k = 6;
else
    k = VarIn{1};
    if ~isPosInt(k) || ~isscalar(k)
        error('Wrong input of k');
    end
    k = double(full(k));
end
k = min([M,N,k]);

% Get the target of the desired singular values.
target = getTarget(VarIn);
if isnumeric(target) && target == 0
    warning('target should not be zero');
end

% Get the starting vector x0 and optional parameters.
[u0,v0,Options] = getOptions(VarIn,M,N,k,randStr);
end

function [M,N,VarIn] = getSizeA(A,VarIn)
% Error check and get the size of A in checkInputs.

if isa(A,'function_handle')
    if numel(VarIn) < 1
        error('For a function handle A, its sizes should be given');
    end
    
    % MN gives the sizes of A.
    MN=VarIn{1};
    
    % Error check M and N.
    if  ~isPosInt(MN) || ~isrow(MN) || length(MN) ~= 2 || ~all(isfinite(MN))
        error('Wrong inputs for the size numbers M, N');
    else
        M = double(full(MN(1)));
        N = double(full(MN(2)));
    end
    
    % Remove MN from VarIn. The remaining entries are k, sigma, and Options
    % which matches VarIn when A is given as a matrix.
    VarIn=VarIn(2:numel(VarIn));
elseif ismatrix(A)
    % Save size of A in [M,N].
    [M,N] = size(A);
else
    error('A should be either a function handle or a matrix');
end
end

function target = getTarget(VarIn)
% Error check and get the target in checkInputs.

if length(VarIn) < 2
    target = 'largest';
else
    target = VarIn{2};
    
    % Error Check sigma.
    if (ischar(target) && isrow(target)) || (isstring(target) && isscalar(target))
        ValidTarget = {'largest','smallest'};
        match = startsWith(ValidTarget, target, 'IgnoreCase', true);
        j = find(match,1);
        if isempty(j) || (strlength(target) == 0)
            error('The target should be one of the strings: "largest" and "smallest"');
        else
            % Reset sigma to the correct valid sigma for cheaper checking.
            target = ValidTarget{j};
        end
    elseif isfloat(target)
        if ~isreal(target) || ~isscalar(target) || ~isfinite(target)
            error('The target should be a real finite scalar');
        end
    end
end
end

function [u0,v0,Options] = getOptions(VarIn,M,N,k,randStr)
% Get x0 and options, and set defaults if they are not provided.

Options = struct;
nVarIn = numel(VarIn);
NameValueFlag = false;

% Get Options, if provided.
if nVarIn >=3
    if isstruct(VarIn{3})
        Options = VarIn{3};
        
        % VarIn should be {k, sigma, Options}.
        if nVarIn > 3
            error('Only one of the structure "opts" and name-value pairs pattern can be used to specify the parameters' );
        end
    else
        % Convert the Name-Value pairs to a struct for ease of error checking.
        NameValueFlag = true;
        for j=3:2:nVarIn
            name = VarIn{j};
            if (~(ischar(name) && isrow(name)) && ~(isstring(name) && isscalar(name))) ...
                    || (isstring(name) && strlength(name) == 0)
                error('Wtong input for the names');
            end
            nvNames=["Tolerance","MaxIterations","MaxSubspaceDimension","MinSubspaceDimension",...
                "SwitchingTolerance","InnerTolerance","LeftStartVector","RightStartVector",...
                "InnerPreconditionTolerance1","InnerPreconditionTolerance2","Display","ExactInnerSolver"];
            ind = matches(nvNames, name, 'IgnoreCase', true);
            if nnz(ind) ~= 1
                error('')
            end
            if j+1 > nVarIn
                error('');
            end
            structNames = {'tol','maxit','maxsize','minsize','fixtol','inntol','u0','v0','innprectol1','innprectol2','disp','exact'};
            Options.(structNames{ind}) = VarIn{j+1};
        end
    end
end

% Error check Options.tol if provided or set the default value for it if not.
if isfield(Options,'tol')
    if ~isnumeric(Options.tol) || ~isscalar(Options.tol) ...
            || ~isreal(Options.tol) || ~(Options.tol >= 0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.tol = 1e-8;
end

% Error check Options.maxit if provided or set the default value for it if not.
if isfield(Options,'maxit')
    if ~isPosInt(Options.maxit) || ~isscalar(Options.maxit)...
            || Options.maxit == 0
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.maxit = min([M,N]);
end

% Error check Options.maxsize if provided or set the default value for it if not.
if isfield(Options,'maxsize')
    if ~isPosInt(Options.maxsize) || ~isscalar(Options.maxsize)...
            || Options.maxsize == 0
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
    if Options.maxsize < k+2
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.maxsize = max([30 k+3]);
end

% Error check Options.minsize if provided or set the default value for it if not.
if isfield(Options,'minsize')
    if ~isPosInt(Options.minsize) || ~isscalar(Options.minsize)...
            || Options.minsize == 0
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.minsize = 3;
end

% Error check Options.fixtol if provided or set the default value for it if not.
if isfield(Options,'fixtol')
    if ~isnumeric(Options.fixtol) || ~isscalar(Options.fixtol) ...
            || ~isreal(Options.fixtol) || ~(Options.fixtol >= 0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.fixtol = 1e-4;
end

% Error check Options.inntol if provided or set the default value for it if not.
if isfield(Options,'inntol')
    if ~isnumeric(Options.inntol) || ~isscalar(Options.inntol) ...
            || ~isreal(Options.inntol) || ~(Options.inntol >= 0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.inntol = 1e-3;
end

% Error check Options.u0 if provided or set the default value for it if not.
if isfield(Options,'u0')
    u0 = Options.u0;
    if ~iscolumn(u0) || length(u0) ~= M || ~isfloat(u0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    u0 = randn(randStr,M,1);
end

% Error check Options.v0 if provided or set the default value for it if not.
if isfield(Options,'v0')
    v0 = Options.v0;
    if ~iscolumn(v0) || length(v0) ~= N || ~isfloat(v0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    v0 = randn(randStr,N,1);
end

% Error check Options.innprec if provided or set the default value for it if not.
if isfield(Options,'innprectol1')
    if ~isnumeric(Options.innprectol1) || ~isscalar(Options.innprectol1) || ~isreal(Options.innprectol1)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.innprectol1 = 0.05;
end

% Error check Options.inntol if provided or set the default value for it if not.
if isfield(Options,'innprectol2')
    if ~isnumeric(Options.innprectol2) || ~isscalar(Options.innprectol2) ...
            || ~isreal(Options.innprectol2) || ~(Options.innprectol2 >= 0)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
elseif Options.innprec > 0
    Options.innprectol2 = 0.01;
end

% Error check Options.disp if provided or set the default value for it if not.
if isfield(Options,'disp')
    if ~isnumeric(Options.disp) || ~isscalar(Options.disp) || ~isreal(Options.disp)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.disp = 0;
end

% Error check Options.exact if provided or set the default value for it if not.
if isfield(Options,'exact')
    if ~isnumeric(Options.exact) || ~isscalar(Options.exact) || ~isreal(Options.exact)
        if NameValueFlag
            error('Wrong input in the name-value pairs');
        else
            error('Wrong input in the structure "opts"');
        end
    end
else
    Options.exact = 0;
end
end

function  [tf] = isPosInt(X)
% Check if X is a non-negative integer vector.
tf = isnumeric(X) && isreal(X) && all(X(:) >= 0) && all(fix(X(:)) == X(:));
end

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

function [Theta,Ctemp,Dtemp] = StandardExtraction(H,target,m)
% Compute the SVD of the small projected matrix and resort the approximate
%singular values in the desired oder.

% Compute the small scaled SVD.
[Ctemp,Theta,Dtemp] = svd(H(1:m,1:m));

% Resort the approximate singular values and the Ritz vectors.
Isort = StandardSortSigmas(Theta,target,m);
Theta = Theta(Isort,Isort);      
Ctemp = Ctemp(:,Isort);     
Dtemp = Dtemp(:,Isort);
end

function Isort = StandardSortSigmas(Theta,target,m)
% Reorder the approximate  singular values in StandardExtraction.

if (isscalar(target) && target<10^(-16)) || strcmp(target,'smallest')
    [~,Isort] = sort(diag(Theta));
elseif (isscalar(target) && target>10^(16)) || strcmp(target,'largest')
    [~,Isort] = sort(diag(Theta),'descend');
elseif isscalar(target) && target>=10^(-16) && target<=10^(16)
    [~,Isort] = sort(abs(diag(Theta)-target*ones(m,1)));
else
    error('wrong input for target');
end
end

function [U,V,ATU,AV,H]=StandardDeflation(U,V,ATU,AV,Ctemp,Dtemp,Theta,m)
% Perform one step of restart after the deflation.

U(:,1:m-1) = U(:,1:m)*Ctemp(:,2:m);
V(:,1:m-1) = V(:,1:m)*Dtemp(:,2:m);
ATU(:,1:m-1) = ATU(:,1:m)*Ctemp(:,2:m);
AV(:,1:m-1) = AV(:,1:m)*Dtemp(:,2:m);
H(1:m-1,1:m-1) = Theta(2:m,2:m);
end

function targetin = GetInnerTarget(Theta,target,RelResidual,Options)
% Determine the parameter targetin used in the followed correction equation 

% Determine the inner target.
theta = Theta(1,1);
if isscalar(target) && RelResidual > Options.fixtol
    targetin = target;
else
    targetin = theta;
end
end

function iexten = StandardExtendIndexI(Theta,targetin,Options,m)
% Determine the indexes of the singular vectors to be extended to the
% correction equations for the first time.
if m > 1
    theta = diag(Theta(2:m,2:m)); 
    iexten = 1+find(abs(theta-targetin)./max(theta,1) <= Options.innprectol1);
else
    iexten = [];
end
iexten=iexten';
end

function jexten = StandardExtendIndexII(Resexten,normAest,Options)
% Determine the final indexes of the singular vectors to be extended to the
% correction equations.
jexten = find(Resexten <= normAest*Options.innprectol2);
end
 
function epsilonin = StandardParamterN(Theta,targetin,iexten,Options,m)
% Determine the stopping tolerance epsilonin for the inner iterations to 
% solve the correction equation.

% Compute the accuracy requirement for solution. 
ConstantParameter = 2*sqrt(2);
if m>1
    iselected = [1 iexten];
    iunselect = setdiff(1:1:m,iselected);
    if ~isempty(iunselect)
    Theta = diag(Theta);
    relativesep = zeros(length(iselected),1);
    for i=iselected
        relativesep(i) = min(abs((Theta(i)-Theta(iunselect))./(Theta(iunselect)-targetin)));
    end
    if relativesep(i)~=0
        ConstantParameter = ConstantParameter/min(abs(relativesep));
    end
    end
end

% Compute inner stopping criteria.
epsilonin = min([Options.inntol*ConstantParameter 0.01]);
end

function [U,V,ATU,AV,H,m] = StandardRestart(U,V,ATU,AV,Ctemp,Dtemp,Theta,maxsize,minsize,iexten)
% Perform the thick restart techenique for the standard extraction approach.
% iselected = [1 iexten];
% iunselected=setdiff(1:maxsize,[1,iexten]);
% inew = [iselected iunselected(1:(minsize-1-length(iexten)))];

if length(iexten)+1<=minsize
    inew = 1:1:minsize;
else
    inew = [1 iexten];
end

m=length(inew);

U(:,1:m) = U(:,1:maxsize)*Ctemp(:,inew);
V(:,1:m) = V(:,1:maxsize)*Dtemp(:,inew);
ATU(:,1:m) = ATU(:,1:maxsize)*Ctemp(:,inew);
AV(:,1:m) = AV(:,1:maxsize)*Dtemp(:,inew);
H(1:m,1:m) = Theta(inew,inew);
end