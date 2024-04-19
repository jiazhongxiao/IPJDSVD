function normest = normestimation (Afun,M,N)
% Estimate the 2-norm of a given matrix function handle A by
%        ||A||_2 ≈ sqrt(||A||_1*||A||_inf).

EyeM = speye(M,M);
EyeN = speye(N,N);

normA1 = 0;
for i=1:N
    normA1 = max([normA1;norm(Afun(EyeN(:,i),'notransp'),1)]);
end

normAinf = 0;
for i=1:M
    normAinf = max([normAinf;norm(Afun(EyeM(:,i),'transp'),1)]);
end

normest = sqrt(normA1*normAinf);
end