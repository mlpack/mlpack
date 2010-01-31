function [newx]=sdpca_min_max_eig_trad(x, knn)
%This function computes the  minimum maximum eigenvalue 
%embedding with the most primal SDP formulation
%  min X(1)
%  \Sum\Sum X=0
%  2*X(1)- X_ii  -X_jj +X_ij +X_ji =d_ij
%  X(1)*I-X \geq 0
[n, m]=size(x);
dist=zeros(n,1);
At=sparse(knn*n+1,n^2);
A=sparse(knn*n+n,n^2);
AA=speye(n^2);
b=zeros(knn*n+1,1);
for i=1:n
   for j=1:n
      dist(j)=sum((x(i,:)-x(j,:)).^2);
   end
   [dist I]=sort(dist);
   for j=1:knn
     At((i-1)*knn+j, (i-1)*n+i) = 1;
     At((i-1)*knn+j, (I(j+1)-1)*n+I(j+1)) = 1;
     At((i-1)*knn+j, (I(j+1)-1)*n+i) = -1;
     At((i-1)*knn+j, (i-1)*n+ I(j+1)) = -1;
     A(n+(i-1)*knn+j, (I(j+1)-1)*n+i) = 1;
   end
   A(i, (i-1)*n+i)=1;
   b((i-1)*knn+1:i*knn)=dist(2:knn+1);
end
At(end,:)=1;
%Att=[[2*ones(knn*n,1); n] -At];

Att=[sparse(knn*n+1,1) At sparse(knn*n+1,n^2) ;...
%     [-ones(n,1); sparse(knn*n,1)] A A];
     -vec(speye(n)) AA AA];
 
Att=sparse(Att);
b(end)=0;
b = [b ; -sparse(n^2, 1)];
c=[1 zeros(1,n^2) sparse(1,n^2)];
c=sparse(c);
K.l=1;
K.s=[n n];
pars.maxiter=60;
pars.eps=10^-4;
pars.sdp=1;
[s,y,info]=sedumi(Att,b,c,K,pars);
info
newx=mat(s(2:2+n^2-1),n);