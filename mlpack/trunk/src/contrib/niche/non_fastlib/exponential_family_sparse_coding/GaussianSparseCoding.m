function [D, S] = GaussianSparseCoding(X, n_atoms, lambda, max_iterations, D_initial)
%function [D, S] = GaussianSparseCoding(X, n_atoms, lambda, max_iterations, D_initial)
% 
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j

if nargin < 5
  D_initial = [];
end

if lambda < 0
  lambda_for_obj = 0;
else
  lambda_for_obj = lambda;
end

[n_dims n_points] = size(X);

% hardcoded for now
atom_l2_norm = 1; %100;

% Set Initial Dictionary
if isempty(D_initial)
  D = atom_l2_norm * normcols(normrnd(0,1,n_dims,n_atoms));
else
  D = D_initial;
end

% Sparse codes update via feature-sign
fprintf('INITIAL SPARSE CODING STEP\n');
DT_D = D' * D;
S = zeros(n_atoms, n_points);
for i = 1:n_points
  X_i = X(:,i);
  
  l1_prob_sol = lars(D, X_i, 'LASSO', -lambda, true, DT_D)';
  S(:,i) = l1_prob_sol(:,end);
end

fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
%fprintf('DONE SPARSE CODING\n');
%pause;

fprintf('\t\t\tObjective value: %f\n', ...
	ComputeGaussianFullObjective(D, S, X, lambda_for_obj));

converged = false;
iteration_num = 0;

if iteration_num == max_iterations
  converged = true;
end

while ~converged
  iteration_num = iteration_num + 1;
  fprintf('PSC Iteration %d\n', iteration_num);
  
  % Dictionary update
  fprintf('DICTIONARY LEARNING STEP\n');
  D = l2ls_learn_basis_dual(X, S, atom_l2_norm, D);
  %fprintf('DONE LEARNING DICTIONARY\n');
  %pause;

  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeGaussianFullObjective(D, S, X, lambda_for_obj));
    

  % Sparse codes update via feature-sign
  fprintf('SPARSE CODING STEP\n');
  DT_D = D' * D;
  for i = 1:n_points
    X_i = X(:,i);
    
    l1_prob_sol = lars(D, X_i, 'LASSO', -lambda, true, DT_D)';
    S(:,i) = l1_prob_sol(:,end);
  end
  fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	  norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
  %fprintf('DONE SPARSE CODING\n');
  %pause;

  fprintf('\t\t\tObjective value: %Cf\n', ...
	  ComputeGaussianFullObjective(D, S, X, lambda_for_obj));
  
  
  % check convergence criterion - temporarily just 10 iterations
  % for debugging
  if iteration_num == max_iterations
    converged = true;
  end
end

fprintf('Done learning\n');
