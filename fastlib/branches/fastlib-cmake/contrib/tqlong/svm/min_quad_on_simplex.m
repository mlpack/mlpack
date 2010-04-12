%% solve min x'Kx subject to x in canonical simplex
function [xout, I] = min_quad_on_simplex(K)
tic
n_iter = 20;
n = size(K,1);
x = ones(n,1);
I = false(n,1);
L = 2;
iter = 0;
tol = 1e-8
while (1)
    iter = iter + 1;
    a = K(:,~I)*x(~I);
    L1 = norm(a)/norm(x);
    if (L < L1) L = L1; end
    x_old = x;
    [x, I] = simplex_project(a, L, x);
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