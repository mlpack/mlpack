function [a, b] = update_sparse_variational(laplace_var, exp_param)

options = optimset('GradObj', 'on', 'Hessian', 'on', 'TolFun', 1e-12);
%Gamma(k=3,theta=2) has a heavy tail, so let's start with that
a = fmincon(@(x) sparse_variational_objective(x, laplace_var, exp_param), ...
            3, [], [], [], [], 0, [], [], options);
b = (1 + sqrt(1 + 8 * exp_param * laplace_var^2 * a / (a - 1))) ...
    / (4 * exp_param * a);
