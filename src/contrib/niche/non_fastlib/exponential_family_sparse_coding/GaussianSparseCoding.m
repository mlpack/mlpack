function [D, S] = GaussianSparseCoding(X, n_atoms, lambda, max_iterations, D_initial)
%function [D, S] = GaussianSparseCoding(X, n_atoms, lambda, max_iterations, D_initial)
% 
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j

if nargin < 5
  D_initial = [];
end

verbosity = 1;

obj_tol = 1e-6;

%if lambda < 0
%  lambda_for_obj = 0;
%else
%  lambda_for_obj = lambda;
%end

[n_dims n_points] = size(X);

% hardcoded for now
atom_l2_norm = 1; %100;

% Set Initial Dictionary
if isempty(D_initial)
  D = atom_l2_norm * normcols(normrnd(0,1,n_dims,n_atoms));
else
  D = D_initial;
end

% Sparse codes update
if verbosity == 2
  fprintf('INITIAL SPARSE CODING STEP\n');
end
DT_D = D' * D;
S = zeros(n_atoms, n_points);
parfor i = 1:n_points
  X_i = X(:,i);
  
  l1_prob_sol = lars(D, X_i, 'LASSO', -lambda, true, DT_D)';
  S(:,i) = l1_prob_sol(:,end);
end

if verbosity == 2
  fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	  norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
end

fprintf('\t\t\tObjective value: %f\n', ...
	ComputeGaussianFullObjective(D, S, X, lambda));

converged = false;
iteration_num = 0;

last_obj_val = 1e99;

n_times_little_improv = 0;

if iteration_num == max_iterations
  converged = true;
end

while ~converged
  iteration_num = iteration_num + 1;
  if verbosity >= 1
    fprintf('PSC Iteration %d\n', iteration_num);
  end
  
  % Dictionary update
  if verbosity == 2
    fprintf('D Step\n');
  end
  D = l2ls_learn_basis_dual(X, S, atom_l2_norm, D);
  %fprintf('DONE LEARNING DICTIONARY\n');
  %pause;

  if verbosity == 2
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjective(D, S, X, lambda))
  end
    

  % Sparse codes update
  if verbosity == 2
    fprintf('S Step\n');
  end
  DT_D = D' * D;
  parfor i = 1:n_points
    X_i = X(:,i);
    
    l1_prob_sol = lars(D, X_i, 'LASSO', -lambda, true, DT_D)';
    S(:,i) = l1_prob_sol(:,end);
  end
  if verbosity == 2
    fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	    norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
  end

  if verbosity == 2
    fprintf('\t\t\tObjective value: %Cf\n', ...
	    ComputeGaussianFullObjective(D, S, X, lambda));
  end
  

  cur_obj_val = ComputeGaussianFullObjective(D, S, X, lambda);
  
  obj_val_improv = last_obj_val - cur_obj_val;
  if verbosity >= 1
    fprintf('\t\tIMPROVEMENT: %e\n', ...
	    obj_val_improv);
  end
  if obj_val_improv < obj_tol
    n_times_little_improv = n_times_little_improv + 1;
  end
  
  
  last_obj_val = cur_obj_val;
  
  
    % check convergence criterion
  if n_times_little_improv >= 1
    fprintf('Converged after %d iterations\n', iteration_num);
    converged = true;
  elseif (iteration_num == max_iterations)
    if n_times_little_improv == 0
      fprintf('Early termination after hitting max iterations\n');
    end
    converged = true;
  end
end

fprintf('Done learning\n');
