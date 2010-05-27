%% Minimize t(a'x+\gamma \|x-\bar{x}\|^2)+\sum(-\ln x_i)
%% subject to \sum(x_i) = 1
%% Barrier version
function x = min_barrier(t, a, gamma, bx, x0)
x = x0;
n_iter = 5;
for iter = 1:n_iter
    A = 1./(x.^2)+t*gamma;
    g = a+gamma*(x-bx);
    b = -t*g+1./x;
    [dx, w] = quick_solve(A, b);
    
    I = (dx<0);
    if any(I)
        step = 0.995*min(-x(I)./dx(I));
    else
        I = dx>0;
        if any(I)
            step = 0.995*min((1-x(I))./dx(I));
        else
            break;
        end
    end
    step1 = -dot(g,dx)/dot(dx, dx)/gamma;
    step;
    x = x+min(step1,step)*dx;
end
end