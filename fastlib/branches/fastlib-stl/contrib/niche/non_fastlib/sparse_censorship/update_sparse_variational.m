function [a, b] = update_sparse_variational(laplace_var, exp_param)

options = optimset('GradObj', 'on', 'Hessian', 'on', 'TolFun', 1e-12);
%Gamma(k=3,theta=2) has a heavy tail, so let's start with that
x = fminunc(@(x) sparse_variational_objective(x, laplace_var, exp_param), ...
            [3; 2], options);
[a, b] = x;
