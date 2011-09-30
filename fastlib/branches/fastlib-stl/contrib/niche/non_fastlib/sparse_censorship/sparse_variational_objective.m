function [f,g] = sparse_variational_objective(a, laplace_var, exp_param)
%function [f,g,h] = sparse_variational_objective(a, laplace_var, exp_param)

fprintf('f(%f) = ', a);

b = (1 + sqrt(1 + 8 * exp_param * laplace_var^2 * a / (a - 1))) ...
    / (4 * exp_param * a);

gamma_a = gamma(a);

f = -(2 * a * (1 - b * exp_param) + (laplace_var^2) / (b * (1 - a)) ...
     + log(b) + 2 * log(gamma_a) + (1 - 2 * a) * psi(a)) / 2;

if isreal(f)
  fprintf('%f\n', f);
else
  fprintf('%f + %fi\n', real(f), imag(f));
end

g = -(2 - 2 * b * exp_param + (laplace_var^2) / ((a - 1)^2 * b) ...
     + (1 - 2 * a) * psi(1, a)) / 2;

fprintf('g = %f\n', g);

%h = -(-(laplace_var^2 / ((a - 1)^3 * b)) - psi(1, a) ...
%    + (1/2 - a) * psi(2, a));

%fprintf('h = %f\n', h);