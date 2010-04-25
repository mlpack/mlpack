%% solve min x'Kx subject to x in canonical simplex
function [xout, I] = min_quad_on_simplex_optimal(K)
tic
n_iter = 20;
n = size(K,1);
x = ones(n,1);
w = x;
I = false(n,1);
Iw = false(n,1);
alpha = 1;
L = 2;
iter = 0;
tol = 1e-8
while (1)
    iter = iter + 1;
    a = K(:,~Iw)*w(~Iw);
    L1 = norm(a)/norm(w);
    if (L < L1) L = L1; end
    x_old = x;
    I_old = I;
    
    [x, I] = simplex_project(a, L, w);
    
    alpha = 1/2*(1+sqrt(1+alpha^2));
    w = x + (alpha-1)/(alpha+1)*(x-x_old);
    Iw = I_old | I;

    change = norm(x-x_old)/n;
    %disp(sprintf('iter = %d (change = %f < tol = %d) SVs = %d',iter, change, ...
    %    change < tol, n-sum(I)))
    if change < tol
        disp('Converged')
        break;
    end
end
xout = x;
L
t = toc
disp(sprintf('times = %f, iter = %d', t, iter))
end