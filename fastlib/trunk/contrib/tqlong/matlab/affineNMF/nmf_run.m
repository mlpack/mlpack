function [W,H] = nmf_run(V, r, Winit, Hinit)
[m, n] = size(V);
if nargin == 2,
    Winit = rand(m, r);
    Hinit = rand(r, n);
elseif nargin == 3,
    Hinit = rand(r, n);
end
tol = 1e-6;
timelimit = 10;
maxiter = 1000;
[W,H] = nmf(V,Winit,Hinit,tol,timelimit,maxiter);
end