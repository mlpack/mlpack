function [W, Z] = TaskCoding(X, Y, n_atoms, lambda_w, lambda_z, tol)
%function [W, Z] = TaskCoding(X, Y, n_atoms, lambda_w, lambda_z, tol)
%
% X is tensor in R^{n_dims \times n_points \times n_tasks}
% Y is a matrix in R^{n_points \times n_tasks}
% n_atoms is the number of atoms in the dictionary of estimators
% lambda_w is the regularization term for the penalty on the SVM parameters
% lambda_z is the regularization term for l1-norm penalties on each Z_t
% tol is the tolerance within which the objective must have converged to stop the algorithm

if nargin < 6
  tol = 1e-6;
end

[n_dims n_points n_tasks] = size(X);

% initialize W somehow
W = normrnd(0,1,n_dims, n_atoms);

last_obj = 1e99;
converged = false;
n_max_iterations = 50;
iteration_num = 0;
while ~converged && iteration_num < n_max_iterations
  iteration_num = iteration_num + 1;
  Z = TaskCodingZStep(X, Y, W, lambda_z);
  obj = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z);
  W = TaskCodingWStep(X, Y, Z, lambda_w);
  obj = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z);
  
  %converged = (last_obj - obj) < tol;
  last_obj = obj;
end

% final coding step
Z = TaskCodingZStep(X, Y, W, lambda_z);
