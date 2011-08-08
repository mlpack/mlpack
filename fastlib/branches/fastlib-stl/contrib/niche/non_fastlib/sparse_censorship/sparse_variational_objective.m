function [f,g,h] = sparse_variational_objective(x, laplace_var, exp_param)

a = x(1);
b = x(2);

gamma_a = gamma(a);

f = (2 * a * (1 - b * exp_param) + (laplace_var^2) / (b * (1 - a)) ...
     + log(b) + 2 * log(gamma_a) + (1 - 2 * a) * gamma_a) / 2;

g = (2 - 2 * b * exp_param + (laplace_var^2) / ((a - 1)^2 * b) ...
     + (1 - 2 * a) * psi(1, a)) / 2;

h = -(laplace_var^2 / ((a - 1)^3 * b)) - psi(1, a) ...
    + (1/2 - a) * psi(2, a);

     
