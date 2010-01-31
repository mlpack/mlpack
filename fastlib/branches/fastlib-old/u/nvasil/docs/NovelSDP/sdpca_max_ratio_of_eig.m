function [newx]=sdpca_max_ratio_of_eig(x, knn, neig, ratio)
% This tries to maximize the ratio between the maximum and minimum
% eigenvalue of the kernel matrix

[newx]=sdpca_max_trace(x, knn);
[n m]=size(x);
% Preservation of distances
dist=zeros(n,1);
At=sparse(knn*n+1+neig-1, n^2);
b=zeros(knn*n+1+neig-1, 1);
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
   end
   b((i-1)*knn+1:i*knn)=dist(2:knn+1);
end
At(end-1,:)=1;

[v d]=eig(newx);
v=fliplr(v);
d=-sort(diag(-d));
d=d/max(d);
d=d(1:neig);
mu=d(1:end-1)./d(2:end);
  
info.feasratio=1;
plot(d,'b');
for i=1:50
  [v d]=eig(newx);
  v=fliplr(v);
  d=-sort(diag(-d));
  d=d(1:neig);
  d=d/max(d);
  v=v(:,1:neig);
  mu=d(1:end-1)./d(2:end);
  mu=mu*ratio;

%Ratio of eigenvalues condition
  A=zeros(neig, n^2);
  for k=1:neig-1
     A(k,:)=vec(v(:,k)*v(:,k)' - mu(k)*v(:,k+1)*v(:,k+1)');
  end
  At(end-neig+1:end,:)=A;
  K.s=n;
  pars.maxiter=30;
  pars.eps=10^-5;
  [x, y, info]=sedumi(At(1:end,:),b(1:end),0,K,pars);
  if (info.feasratio<0)
      break;
  end
  newx=mat(x,n);
end
hold on
plot(d, 'r');