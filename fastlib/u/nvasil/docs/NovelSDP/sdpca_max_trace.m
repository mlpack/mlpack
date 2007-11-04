function [newx]=sdpca_max_trace(x, knn)
%This function computes the maximum trace embedding with
%the most obvious SDP formulation
%  min Tr(-X)
%  \Sum\Sum X=0
%  X_ii + X_jj -X_ij -X_ji =d_ij
%  X \geq 0

[n, m]=size(x);
dist=zeros(n,1);
A=zeros(n,n);
for i=1:n
   for j=1:n
      dist(j)=x(i,:)*x(i,:)'+ x(j,:)*x(j,:)' -2*x(j,:)*x(i,:)' ;
   end
   [dist I]=sort(dist);
   A(i,I(1:knn+1))=dist(1:knn+1);
end
A=max(A,A');
num_of_neighbors=length(find(A(:)~=0));

At=sparse(num_of_neighbors+1,n^2);
b=zeros(num_of_neighbors+1,1);
count=0;
for i=1:n
   [I]=find(A(i,:)~=0);
   dist=A(i,I);
   for j=1:length(I)
     count=count+1;
     At(count, sub2ind([n n], i,i)) = 1;
     At(count, sub2ind([n n], I(j),I(j))) = 1;
     At(count, sub2ind([n n], I(j),i)) = -1;
     At(count, sub2ind([n n], i, I(j))) = -1;
     b(count)=dist(j);
   end
end
At(end,:)=1;
b(end)=0;
c=-eye(n);
c=c(:);
K.s=n;
pars.maxiter=90;
pars.eps=10^-3;
pars.sdp=1;
[s,y,info]=sedumi(At,b,10^-5*c,K,pars);
newx=mat(s,n);
