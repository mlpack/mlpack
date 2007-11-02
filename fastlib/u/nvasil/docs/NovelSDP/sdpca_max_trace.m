function [newx]=sdpca_max_trace(x, knn)
%This function computes the maximum trace embedding with
%the most obvious SDP formulation
%  min Tr(-X)
%  \Sum\Sum X=0
%  X_ii + X_jj -X_ij -X_ji =d_ij
%  X \geq 0
[n, m]=size(x);
dist=zeros(n,1);
At=sparse(knn*n+1,n^2);
b=zeros(knn*n+1,1);
for i=1:n
   for j=1:n
      dist(j)=sum((x(i,:)-x(j,:)).^2);
   end
   [dist I]=sort(dist);
   for j=1:knn
     At((i-1)*knn+j, sub2ind([n n], i,i)) = 1;
     At((i-1)*knn+j, sub2ind([n n], I(j+1),I(j+1))) = 1;
     At((i-1)*knn+j, sub2ind([n n], I(j+1),i)) = -1;
     At((i-1)*knn+j, sub2ind([n n], i, I(j+1))) = -1;
   end
   b((i-1)*knn+1:i*knn)=dist(2:knn+1);
end
At(end,:)=1;
b(end)=0;
c=-eye(n);
c=c(:);
K.s=n;
pars.maxiter=90;
pars.eps=10^-2;
pars.sdp=1;
[s,y,info]=sedumi(At,b,c,K,pars);
newx=mat(s,n);
