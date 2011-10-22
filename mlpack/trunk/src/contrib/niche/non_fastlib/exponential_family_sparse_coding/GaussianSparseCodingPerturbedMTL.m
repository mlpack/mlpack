function [D Q S] = GaussianSparseCodingPerturbedMTL(X, n_points, n_atoms, ...
						  lambda_S, lambda_Q, ...
						  max_iterations, D_initial)
%function [D Q S] = GaussianSparseCodingPerturbedMTL(X, n_points, n_atoms, ...
%						  lambda_S, lambda_Q, ...
%						  max_iterations, D_initial)
% 
% Given: X - a matrix X = [X^1 ... X^n] where each column of X^t is a document,
% and each row corresponds to vocabulary word. 
% X^t_{i,j} is the number of times word i occurs in document j of task t.
%        n_points is vector of length n_tasks where n_points(t)
%        indicates the number of documents in task t

if nargin < 7
  D_initial = [];
end

verbosity = 2;

obj_tol = 1e-6;




%if lambda_S < 0
%  lambda_for_obj = 0;
%else
%  lambda_for_obj = lambda_S;
%end

n_tasks = length(n_points);
n_dims = size(X, 1);
n_points_total = sum(n_points);

point_inds = cell(n_tasks, 1);
cur_ind = 1;
for t = 1:n_tasks
  point_inds{t} = cur_ind:(cur_ind + n_points(t) - 1);
  cur_ind = cur_ind + n_points(t);
end


% hardcoded for now
atom_l2_norm = 1; %100;

% Set Initial Dictionary
if isempty(D_initial)
  D = atom_l2_norm * normcols(normrnd(0, 1, n_dims, n_atoms));
else
  D = D_initial;
end

Q = cell(n_tasks, 1);
Dmod = cell(n_tasks, 1);
for t = 1:n_tasks
  Q{t} = zeros(size(D));
  Dmod{t} = D + Q{t};
end

% Sparse codes update
if verbosity == 2
  fprintf('Initial S Step\n');
end

DmodT_Dmod = cell(n_tasks, 1);
for t = 1:n_tasks
  DmodT_Dmod{t} = Dmod{t}' * Dmod{t};
end

S = zeros(n_atoms, n_points_total);
for t = 1:n_tasks
  parfor ind = point_inds{t}(1:n_points(t))
    l1_prob_sol = ...
	lars(Dmod{t}, X(:, ind),'LASSO', -lambda_S, true, DmodT_Dmod{t})';
    S(:,ind) = l1_prob_sol(:,end);
  end
end

if verbosity == 2
  for t = 1:n_tasks
    fprintf('Task %d, S %f%% sparsity\n', ...
	    t, (nnz(S(:,point_inds{t})) / (n_atoms * n_points(t))) * 100);
  end
  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						   lambda_S, lambda_Q, ...
						   point_inds));
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
  Xmod = zeros(size(X));
  for t = 1:n_tasks
    inds = point_inds{t};
    Xmod(:,inds) = X(:,inds) - Q{t} * S(:,inds);
  end
  D = l2ls_learn_basis_dual(Xmod, S, atom_l2_norm, D);
  for t = 1:n_tasks
    Dmod{t} = D + Q{t};
  end
  if verbosity == 2
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						     lambda_S, lambda_Q, ...
						     point_inds));
  end
  
  
  % 2) Sparse codes update
  if verbosity == 2
    fprintf('S Step\n');
  end
  for t = 1:n_tasks
    DmodT_Dmod{t} = Dmod{t}' * Dmod{t};
  end
  
  for t = 1:n_tasks
    parfor ind = point_inds{t}(1:n_points(t))
      l1_prob_sol = ...
	  lars(Dmod{t}, X(:, ind),'LASSO', -lambda_S, true, DmodT_Dmod{t})';
      S(:,ind) = l1_prob_sol(:,end);
    end
  end
  
  if verbosity == 2
    for t = 1:n_tasks
      fprintf('Task %d, S %f%% sparsity\n', ...
	      t, (nnz(S(:,point_inds{t})) / (n_atoms * n_points(t))) * 100);
    end
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						     lambda_S, lambda_Q, ...
						     point_inds));
  end
  
  
  % 3) Dictionary Perturbation update
  if verbosity == 2
    fprintf('Q Step\n');
  end
  for t = 1:n_tasks
    inds = point_inds{t};
    XmodT = (X(:,inds) - D * S(:,inds))';
    S_ST = S(:,inds) * S(:,inds)';
    Q_t = zeros(size(Q{t})); % we deal with Q_t instead of Q{t} 
                             % to make MATLAB not complain about
                             % the parfor statement below
    parfor i = 1:n_dims
      % LASSO
      %l1_prob_sol = lars(S(:,inds)', XmodT(:,i), 'LASSO', -lambda_Q, true, S_ST)';
      %Q_t(i,:) = l1_prob_sol(:,end)';

      % least squares with negativity constraints and penalization
      % by sum of negative components
      Q_t(i,:) = sparse_neg(S(:,inds)', XmodT(:,i), lambda_Q);
    end
    Q{t} = Q_t;
    Dmod{t} = D + Q{t};
  end
  if verbosity == 2
    for t = 1:n_tasks
      fprintf('Task %d, Q %f%% sparsity\n', ...
	      t, (nnz(Q{t}) / (n_dims * n_atoms)) * 100);
    end
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						     lambda_S, lambda_Q, ...
						     point_inds));
  end
  
  
  % 4) Sparse codes update
  if verbosity == 2
    fprintf('S Step\n');
  end
  for t = 1:n_tasks
    DmodT_Dmod{t} = Dmod{t}' * Dmod{t};
  end
  
  for t = 1:n_tasks
    parfor ind = point_inds{t}(1:n_points(t))
      l1_prob_sol = ...
	  lars(Dmod{t}, X(:, ind),'LASSO', -lambda_S, true, DmodT_Dmod{t})';
      S(:,ind) = l1_prob_sol(:,end);
    end
  end
  
  if verbosity == 2
    for t = 1:n_tasks
      fprintf('Task %d, S %f%% sparsity\n', ...
	      t, (nnz(S(:,point_inds{t})) / (n_atoms * n_points(t))) * 100);
    end
    fprintf('\t\t\tObjective value: %f\n', ...
	    ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						     lambda_S, lambda_Q, ...
						     point_inds));
  end

  % check for convergence  
  
  cur_obj_val = ComputeGaussianFullObjectivePerturbedMTL(D, Q, S, X, ...
						  lambda_S, lambda_Q, ...
						  point_inds);

  obj_val_improv = last_obj_val - cur_obj_val;
  if verbosity >= 1
    fprintf('\t\tIMPROVEMENT: %e\n', ...
	    obj_val_improv);
  end
  if obj_val_improv < obj_tol
    n_times_little_improv = n_times_little_improv + 1;
  end
  
  
  last_obj_val = cur_obj_val;
  
  
  % check convergence criterion - temporarily just 10 iterations
  % for debugging
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
