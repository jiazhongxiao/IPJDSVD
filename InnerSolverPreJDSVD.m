function varargout = InnerSolverPreJDSVD(varargin)
% Solve the correction equations involved in the JD type GSVD algorithms.
%Afun,MN,router,U_conve,V_conve,ufound,vfound,targetin,epsilonin

% Get or creat the function handles Afun and Bfun.
if isa(varargin{1},'function_handle')
    Afun = varargin{1};
elseif ismatrix(varargin{1})
    Afun = MatrixFunction(varargin{1});
end

% Get the size information on A.
MN = varargin{2};
M = MN(1);
N = MN(2);

% Get the right hand side vector of the equation.
router = varargin{3};

% Get the converged and approximate right generalized singular vectors.
Uconve = varargin{4};               Vconve = varargin{5};
ufound = varargin{6};               vfound = varargin{7};
Uexten = varargin{8};               Vexten = varargin{9};
Up = [Uconve ufound Uexten];        Vp = [Vconve vfound Vexten];

% norm(Up'*Up-eye(size(Up,2),size(Up,2)))
% norm(Vp'*Vp-eye(size(Vp,2),size(Vp,2)))

% Get the parameters target and inner stopping tolerance.
target = varargin{10};               inepsilon = varargin{11};

% Calculate the projected right hand side to make the equation consistant.
router_projection(1:M,:) = router(1:M,:) - Uconve*(Uconve'*router(1:M,:));
router_projection((M+1):(M+N),:) = router((M+1):(M+N),:) - Vconve*(Vconve'*router((M+1):(M+N),:));

%norm(Up'*router_projection(1:M,:))+norm(Vp'*router_projection((M+1):(M+N),:))

% Creat the function handle of applying matrix-vector multiplication with
% the coefficient matrix.
mvpwithB = MatricesFunction(Afun,M,N,Up,Vp,target);

% Turn off the warning.
warning off;

% Solve the correction equation using the MINRES algorithm.
[st,flag,relres,iter] = minres(mvpwithB,router_projection,inepsilon,M+N);
 

% Turn on the warning.
warning on;

% Output option I: Return the approximate solution of this equation.
varargout{1} = st(1:M,:) - Up*(Up'*st(1:M,:));
varargout{2} = st((1+M):(M+N),:) - Vp*(Vp'*st((1+M):(M+N),:));

% Output option II: Return the number of inner iterations of solving this equation.
if nargout>=3
    varargout{3} = iter;
end

% Output option III: Return the relative residual norm of the equation.
if nargout>=4
    varargout{4} = relres;
end

% Output option IV: Return the convergence flag.
if nargout>=5
    varargout{5} = flag;
end
end

function varargout=MatrixFunction(C)
% Creat the function handle with a given matrix C.
varargout{1}=@matrixfun;
    function y = matrixfun(x,transpornot)
        if strcmp(transpornot,'notransp')
            y = C*x;
        elseif strcmp(transpornot,'transp')
            y = C'*x;
        end
    end
end

function varargout = MatricesFunction(Afun,M,N,Up,Vp,target)
% Creat the function handle involved in innersolver.
varargout{1}=@matricesfunction;
function y = matricesfunction(x)
   xupper = x(1:M,:)-Up*(Up'*x(1:M,:));
   xlower = x((M+1):(M+N),:)-Vp*(Vp'*x((M+1):(M+N),:));
   yupper = Afun(xlower,'notransp')-target*xupper;
   ylower = Afun(xupper,'transp')-target*xlower;
   y(1:M,:) = yupper-Up*(Up'*yupper);
   y((M+1):(M+N),:) = ylower-Vp*(Vp'*ylower);
end
end

