function varargout=normalizing(X)
% Normalize a vector or the columns of a matrix.

% Check the number of outputs.
nargoutchk(1,2);

% Get the number of columns of X.
n=size(X,2);

% Get the norm of each columns of X.
normX=vecnorm(X,2);

% Normalize each column of X.
for i=1:n
    if normX(1,i) ~= 0
        X(:,i)=X(:,i)/normX(1,i);
    end
end

% Output Option I: Return the normalized X whose columns are of unit-length.
varargout{1} = X;

% Output Option II: Return the norm of each column of the old X.
if nargout > 1
    varargout{2} = normX; 
end
end