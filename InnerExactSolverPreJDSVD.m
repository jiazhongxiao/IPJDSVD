function varargout = InnerExactSolverPreJDSVD(A,varargin)
% Solve the correction equations involved in the JD type SVD algorithms
% exactly using a direct method.
% This function is used for no other but experimental purpose only.

% Check the number of outputs.
nargoutchk(1,5);

VarIn = varargin;

% Get the size information on A and B.
if isa(A,'function_handle')  
    MN = VarIn{1};
    if ~isPosInt(MN) || ~isrow(MN) || length(MN) ~= 2 || ~all(isfinite(MN))
        error('Wrong input MPN');
    end
    M = MN(1);
    N = MN(2);
    VarIn{1} = [];
end

% Get the matrix from function handles Afun and Bfun.
if isa(A,'function_handle')
    Amat = MatrixfromFunction(A,M,N);
elseif ismatrix(A)
    Amat = A;
end

% Get the right hand side vector of the equation.
router = VarIn{1};

% Get the converged and approximate right generalized singular vectors.
Uconve = varargin{2};               Vconve = varargin{3};
ufound = varargin{4};               vfound = varargin{5};
Uexten = varargin{6};               Vexten = varargin{7};
Up = [Uconve ufound Uexten];        Vp = [Vconve vfound Vexten];
 

% Get the parameters target and inner stopping tolerance.
target = varargin{8};        

% Calculate the projected right hand side to make the equation consistant.
router_projection(1:M,:) = router(1:M,:) - Up*(Up'*router(1:M,:));
router_projection((M+1):(M+N),:) = router((M+1):(M+N),:) - Vp*(Vp'*router((M+1):(M+N),:));

% For the middle matrix in the coefficient matrix of the correction equation.
Mmiddle = [-target*speye(M,M) Amat;Amat' -target*speye(N,N)];
UVp = [Up zeros(M,size(Vp,2));zeros(N,size(Up,2)) Vp];

% Turn off the warning.
warning off;

% Compute Mmiddle^{-1}*router_projection.
MinvRouter = linsolve(full(Mmiddle),router_projection);

% Compute Mmiddle^{-1}*UVp.
MinvUVp = linsolve(full(Mmiddle),UVp);

% Compute the coefficient vector.
Ypcoefficient = linsolve(UVp'*MinvUVp,UVp'*MinvRouter);

% Turn on the warning.
warning on;

% Calculate the solution.
st = MinvUVp*Ypcoefficient - MinvRouter;
st(1:M,:) = st(1:M,:) - Up*(Up'*st(1:M,:));
st((1+M):(M+N),:) = st((1+M):(M+N),:) - Vp*(Vp'*st((1+M):(M+N),:));

% Output option I: Return the approximate solution of this equation.
varargout{1} = st(1:M,:);
varargout{2} = st((1+M):(M+N),:);

% Output option II: Return the number of inner iterations of solving this equation.
if nargout>=3
    varargout{3} = 1;
end

% Output option III: Return the relative residual norm of the equation.
if nargout>=4
    Mst = Mmiddle*st;
    relres = norm(Mst-UVp*(UVp'*Mst)+router_projection)/norm(router_projection);
    varargout{4} = relres;
end

% Output option IV: Return the convergence flag.
if nargout>=5
    varargout{5} = 0;
end
end

function  [tf] = isPosInt(X)
% Check if X is a non-negative integer vector.
tf = isnumeric(X) && isreal(X) && all(X(:) >= 0) && all(fix(X(:)) == X(:));
end

function vargout = MatrixfromFunction(Cfun,m,n)
% Creat the matrix from a function handle.

% Creat an m-by-n sparse zero matrix.
C = sparse(m,n);

% Creat an m-by-1 sparse zero vector.
e = sparse(m,1);

for i=1:n
    % Turn the zero vector to the ith unit vector.
    e(i,1) = 1;

    % Compute the ith column of C.
    C(:,i) = Cfun(e,'nottransp');

    % Turn the ith unit vector back to zero vector;
    e(i,1)=0;
end

vargout{1} = C;
end 

