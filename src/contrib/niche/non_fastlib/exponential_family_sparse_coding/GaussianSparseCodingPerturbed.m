function [D Q S] = GaussianSparseCodingPerturbed(X, n_atoms, lambda_S, lambda_Q, max_iterations, D_initial)
%function [D Q S] = GaussianSparseCodingPerturbed(X, n_atoms, lambda_S, lambda_Q, max_iterations, D_initial)
% 
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j

if nargin < 6
  D_initial = [];
end

verbosity = 1;

obj_tol = 1e-6;


%if lambda_S < 0
%  lambda_for_obj = 0;
%else
%  lambda_for_obj = lambda_S;
%end

[n_dims n_points] = size(X);

% hardcoded for now
atom_l2_norm = 1; %100;

% Set Initial Dictionary
if isempty(D_initial)
  D = atom_l2_norm * normcols(normrnd(0, 1, n_dims, n_atoms));
else
  D = D_initial;
end

Q = zeros(size(D));
Dmod = D + Q;

% Sparse codes update
if verbosity == 2
  fprintf('Initial S Step\n');
end
DmodT_Dmod = Dmod' * Dmod;
S = zeros(n_atoms, n_points);
parfor i = 1:n_points
  X_i = X(:,i);
  
  l1_prob_sol = lars(Dmod, X_i, 'LASSO', -lambda_S, true, DmodT_Dmod)';
  S(:,i) = l1_prob_sol(:,end);
end

if verbosity == 2
  fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	  norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q));
end

converged = false;
iteration_num = 0;

if iteration_num == max_iterations
  converged = true;
end

last_obj_val = 1e99;

n_times_little_improv = 0;

while ~converged
  iteration_num = iteration_num + 1;
  if verbosity >= 1
    fprintf('PSC Iteration %d\n', iteration_num);
  end
  
  % 1) Dictionary update
  if verbosity == 2
    fprintf('D Step\n');
  end
  D = l2ls_learn_basis_dual(X - Q * S, S, atom_l2_norm, D);
  Dmod = D + Q;
  if verbosity == 2
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q));
  end
  

  % 2) Sparse codes update
  if verbosity == 2
    fprintf('S Step\n');
  end
  DmodT_Dmod = Dmod' * Dmod;
  parfor i = 1:n_points
    X_i = X(:,i);
    
    l1_prob_sol = lars(Dmod, X_i, 'LASSO', -lambda_S, true, DmodT_Dmod)';
    S(:,i) = l1_prob_sol(:,end);
  end
  if verbosity == 2
    fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	    norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q));
  end

  
  % 3) Dictionary Perturbation update
  if verbosity == 2
    fprintf('Q Step\n');
  end
  X_residT = (X - D * S)';
  S_ST = S * S';
  parfor i = 1:n_dims
    X_residT_i = X_residT(:,i);

    % LASSO
    %l1_prob_sol = lars(S', X_residT_i, 'LASSO', -lambda_Q, true, S_ST)';
    %Q(i,:) = l1_prob_sol(:,end)';

    % least squares with negativity constraints and penalization by
    % sum of negative components
    Q(i,:) = sparse_neg(S', X_residT_i, lambda_Q);
    
  end
  Dmod = D + Q;
  if verbosity == 2
    fprintf('norm(Q) = %f\t||Q||_1 = %f\t%f%% sparsity\n', ...
	    norm(Q), sum(sum(abs(Q))), (nnz(Q) / prod(size(Q))) * 100);
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q));
  end
  

  % 4) Sparse codes update
  if verbosity == 2
    fprintf('S Step\n');
  end
  DmodT_Dmod = Dmod' * Dmod;
  parfor i = 1:n_points
    X_i = X(:,i);
    
    l1_prob_sol = lars(Dmod, X_i, 'LASSO', -lambda_S, true, DmodT_Dmod)';
    S(:,i) = l1_prob_sol(:,end);
  end
  if verbosity == 2
    fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	    norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q));
  end

  
  cur_obj_val = ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q);

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
