function z = orthog(X,y)
% Orthogonalize a vector to a orthonormal matrix.

if ~isempty(X)
    z = y - X*(X'*y);
    if norm(z)<norm(y)/2
        z=z-X*(X'*z);
    end
else
    z=y;
end

z=tools.normalizing(z);
end