N=500;
t=rand(N,1)*3*pi+3*pi/2;
h=rand(N,1)*3*pi+3*pi/2;
x=[t.*cos(t) h, t.*sin(t)];
%x=rand(N,5);
x=x-repmat(mean(x),N,1);
[newx]=sdpca_max_trace(x, 6);
[newx]=sdpca_max_trace_sparse(x,7);
[newx]=sdpca_max_trace_dense(x,5);
[newx]=sdpca_min_max_eig_trad(x, 5);
[newx]=sdpca_min_sum_max_eig_trad(x, 5, 5);
[newx]=sdpca_min_max_eig(x, 8);
[newx]=sdpca_min_max_eig_dense(x, 8);
[newx]=sdpca_min_max_sum_eig(x, 5, 12);
[newx]=sdpca_max_ratio_of_eig(x, 8, 4, 1.0);
[v d]=eig(newx);
v=fliplr(v);
d=-sort(diag(-d));
d=d/max(d);
plot((d(1:6)),'k')
plot(v(:,1),v(:,2),'.')
plot3(v(:,1),v(:,2),v(:,3),'.')
plot3(x(:,1), x(:,2), x(:,3),'.');