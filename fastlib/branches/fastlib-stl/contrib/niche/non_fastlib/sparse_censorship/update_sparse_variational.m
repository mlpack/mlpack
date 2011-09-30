function [a, b] = update_sparse_variational(laplace_var, exp_param, ...
					    v)

%options = optimset('GradObj', 'on', 'Hessian', 'on', 'TolFun', 1e-12);
options = optimset('GradObj', 'on', 'TolFun', 1e-12);
%Gamma(k=3,theta=2) has a heavy tail, so let's start with that
if v == 1
  fprintf('indirect\n');
  %a = fminunc(@(x) sparse_variational_objective(x, laplace_var, exp_param), ...
  %	      3, options);
  initial_guess = 1 / (1 + 8 * exp_param * laplace_var^2);
  a = fmincon(@(x) sparse_variational_objective(x, laplace_var, exp_param), ...
	      0.99999 * initial_guess, [], [], [], [], 0, 1, [], options);
elseif v == 2
  fprintf('direct\n');
  %a = fminunc(@(x) sparse_variational_objective_direct(x, laplace_var, exp_param), ...
  %            3, options);
  a = fmincon(@(x) sparse_variational_objective_direct(x, laplace_var, exp_param), ...
	      3, [], [], [], [], 0, [], [], options);
else
  fprintf('fix\n');
  
  %  initial_guess = [2; 2];
  %initial_guess = [1 / (1 + 8 * exp_param * laplace_var^2); 1];
  initial_guess = [0.9909; 0.1]
  %x = fminunc(@(x) sparse_variational_objective_fix(x, laplace_var, exp_param), ...
  %            initial_guess, options);
  x = fmincon(@(x) sparse_variational_objective_fix(x, laplace_var, exp_param), ...
	      initial_guess, [], [], [], [], 0, [], [], options);

end


%b = (1 + sqrt(1 + 8 * exp_param * laplace_var^2 * a / (a - 1))) ...
%    / (4 * exp_param * a);

a = x(1);
b = x(2);