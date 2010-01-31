function [B,C]=nmf(A,B,C,maxiter,tol)
% NMF from Long
% function [B,C]=nmf(A,B,C,maxiter,tol)
[n,m]=size(B);
[m,k]=size(C);
olderror=-1;
for iter=1:maxiter
    D=log(A)-log(B*C);
    %D(isnan(D)==-Inf);
    if mod(iter,2)==1 % optimize B
        sumC=C*ones(k,1);
        for i=1:n
            factor=(C*D(i,:)')./sumC;
            B(i,:)=B(i,:).*exp(factor)';
        end
    else % optimize C
        sumB=B'*ones(n,1);
        for i=1:k
            factor=(B'*D(:,i))./sumB;
            C(:,i)=C(:,i).*exp(factor);
        end
    end
    eA=B*C;
    error=sum(sum((A.*log(A)-A)-(eA.*log(eA)-eA)-log(eA).*(A-eA)));
    if error<tol || abs(olderror-error)<tol
        disp(sprintf('Converged after %d iterations', iter));
        break;
    end
    disp(sprintf('Iteration %d error %f norm = %f',iter,error,norm(A-B*C,'fro')));
    olderror=error;
end
