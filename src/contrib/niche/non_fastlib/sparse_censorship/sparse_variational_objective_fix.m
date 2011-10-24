function [f,g] = sparse_variational_objective_fix(x, laplace_var, exp_param)

a = x(1);
b = x(2);

fprintf('f([%f %f) = ', a, b);

gamma_a = gamma(a);

f = -(2 * a * (1 - b * exp_param) + (laplace_var^2) / (b * (1 - a)) ...
     + log(b) + 2 * log(gamma_a) + (1 - 2 * a) * psi(a)) / 2;

fprintf('%f\n', f);

g_a = -(2 - 2 * b * exp_param + (laplace_var^2) / ((a - 1)^2 * b) ...
     + (1 - 2 * a) * psi(1, a)) / 2;

g_b = -(-1 / (2 * b) - (a - 1) / b + a / b - a * exp_param ...
      + laplace_var^2 / (2 * (a - 1) * b^2));

g = [g_a; g_b];

fprintf('g = [%f %f]\n', g(1), g(2));

