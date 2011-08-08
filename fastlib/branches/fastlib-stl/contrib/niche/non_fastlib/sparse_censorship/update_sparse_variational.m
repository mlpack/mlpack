function [a, b] = update_sparse_variational(laplace_var, exp_param)

options = optimset('GradObj', 'on', 'Hessian', 'on', 'TolFun', 1e-12);
fminunc(@(x) sparse_variational_objective(x, laplace_var, exp_param), ...
        0, options);
