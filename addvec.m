function C = addvec(A,b)
% Add a vector a matrix b to the right of A;
% A and b can have different numbers of rows and columns.

% Get the sizes of A and B
[M,N]=size(A);  
[m,n]=size(b);

% If b is not empty, then add b to the right of A.
C=[[A;zeros(max([m-M,0]),N)] [b;zeros(max([M-m,0]),n)]];

% If b is empty, then add a zero vector to the right of A.
if isempty(b)
   C=[A zeros(size(A,1),1)]; 
end
end