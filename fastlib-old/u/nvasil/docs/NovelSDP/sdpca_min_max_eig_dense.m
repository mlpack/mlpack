function [newx]=sdpca_min_max_eig_dense(x, knn)

[n, m]=size(x);
dist=zeros(n,1);
A=zeros(n,n)+inf;
At = sparse(n,n);
for i=1:n
   for j=1:n
      dist(j)=sum((x(i,:)-x(j,:)).^2);
   end
   [dist I]=sort(dist);
   A(i,I(1:knn+1))=dist(1:knn+1);
end
A=min(A,A');
I=find(vec(triu(A,1))~=inf & vec(triu(A,1))~=0);
D=A(I);
J=find(A(:)==0);
K=find(vec(triu(A,1))==inf);
At=sparse(length(I),length(I)+length(J));
for i=1:length(I);
    [k,l]=ind2sub([n n],I(i));
    At(i, find(J==sub2ind([n,n],k,k)))=1;
    At(i, find(J==sub2ind([n,n],l,l)))=1;
    At(i,i+length(J))=-2;
end

Ak=sparse(n^2, length(I)+length(J)+length(K)+1);
for i=1:length(J)
    Ak(J(i),i)=1;
end

for i=1:length(I)
    Ak(I(i),i+length(J))=1;
    [k,l]=ind2sub([n n],I(i));
    Ak(sub2ind([n,n],l,k),i+length(J))=1;
end

for i=1:length(K)
    Ak(K(i),i+length(J)+length(I))=1;
    [k,l]=ind2sub([n n],K(i));
    Ak(sub2ind([n,n],l,k),i+length(J)+length(I))=1;
end

Ak(:,end)=-vec(speye(n));
A=[ At zeros(length(I), length(K)+1) ;...
    -At zeros(length(I), length(K)+1);...
    -[eye(length(J)) zeros(length(J), ...
      length(I)+length(K)+1)];...
    [ones(1, length(I)+length(K)+length(J)) 0]; ...
    -[ones(1, length(I)+length(K)+length(J)) 0]; ...
    Ak];
A=sparse(A);
c=[ D(:)+10^10;...
    -D(:)+10^-10;...
    zeros(length(J),1);...
    0;...
    0;...
    zeros(n^2,1)]';
c=sparse(c);
b=[zeros(1,length(I)+length(J)+length(K)) -1]';
b=sparse(b);
Kt.l=2*length(I)+length(J)+2;
Kt.s=n;
pars.maxiter=80;
pars.eps=10^-5;
%pars.sdp=1;
[s,y,info]=sedumi(A, b, c, Kt,pars);
info
newx=mat(Ak(:,1:end-1)*y(1:end-1),n);

