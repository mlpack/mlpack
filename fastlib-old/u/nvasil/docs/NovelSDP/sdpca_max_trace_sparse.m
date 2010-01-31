function [newx]=sdpca_max_trace_sparse(x, knn)

[n, m]=size(x);
dist=zeros(n,1);
A=zeros(n,n)+inf;
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
At=sparse(length(I),length(I)+length(J));
for i=1:length(I);
    [k,l]=ind2sub([n n],I(i));
    At(i, find(J==sub2ind([n,n],k,k)))=1;
    At(i, find(J==sub2ind([n,n],l,l)))=1;
    At(i,i+length(J))=-2;
end

Ak=sparse(n^2, length(I)+length(J));
for i=1:length(J)
    Ak(J(i),i)=1;
end

for i=1:length(I)
    Ak(I(i),i+length(J))=1;
    [k,l]=ind2sub([n n],I(i));
    Ak(sub2ind([n,n],l,k),i+length(J))=1;
end
[n_At m_At]=size(At);
A=[  At ;...
    -[eye(length(J)) zeros(length(J), length(I))];...  
    [ones(1,length(I)+length(J))]; ...
    -[ones(1,length(I)+length(J))]; ...
    -Ak];
A=sparse(A);
c=[  D(:)+10^-15;...
     zeros(length(J),1);...
     0;...
     0;...
     zeros(n^2,1)]';
c=sparse(c);
b=[ones(1,length(J)) zeros(1,length(I))]';
b=sparse(b);
K.l=length(I)+length(J)+2;
K.s=n;
pars.maxiter=80;
pars.eps=10^-5;
%pars.sdp=1;
[s,y,info]=sedumi(A', b, c, K,pars);
info
newx=mat(Ak*y,n);
