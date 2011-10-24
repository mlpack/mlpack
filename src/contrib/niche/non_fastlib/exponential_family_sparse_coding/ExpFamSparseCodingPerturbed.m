function [Q, S] = ExpFamSparseCodingPerturbed(type, X, k, lambda_S, lambda_Q, ...
					      max_iterations, warm_start, D)
%function [Q, S] = ExpFamSparseCodingPerturbed(type, X, k, lambda_S, lambda_Q, ...
%					      max_iterations, warm_start, D)
%
% ONLY IMPLEMENTED FOR type='g'  (GAUSSIAN CASE)
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j
% Good news: warm start is faster
% Bad news:  warm start yields poorer solutions than starting at 0

if type == 'p'
  return;
  ComputeFullObjective = ...
      @(D, S, X, lambda) ...
      ComputePoissonFullObjective(D, S, X, lambda);
  
elseif type == 'b'
  return;
  ComputeFullObjective = ...
      @(D, S, X, lambda) ...
      ComputeBernoulliFullObjective(D, S, X, lambda);

elseif type == 'g'
  ComputeFullObjective = ...
      @(D, Q, S, X, lambda_S, lambda_Q) ...
      ComputeGaussianFullObjectivePerturbed(D, Q, S, X, lambda_S, lambda_Q);

end


[d n] = size(X);

% hardcoded for now
alpha = 1e-4;
beta = 0.9;

c = 1; %100;

% Set Initial Perturbed Dictionary
Q = zeros(size(D));

% Sparse codes update via feature-sign
fprintf('INITIAL SPARSE CODING STEP\n');
S = UpdateSparseCodes(type, X, D + Q, lambda_S, [], alpha, beta);
fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
%fprintf('DONE SPARSE CODING\n');
%pause;

fprintf('\t\t\tObjective value: %f\n', ...
	ComputeFullObjective(D, Q, S, X, lambda_S, lambda_Q));


converged = false;
iteration_num = 0;

if iteration_num == max_iterations
  converged = true;
end

while ~converged
  iteration_num = iteration_num + 1;
  fprintf('PSC Iteration %d\n', iteration_num);
  
  % update of Perturbation to Dictionary
  fprintf('DICTIONARY PERTURBATION LEARNING STEP\n');
  Q = UpdateSparseCodes(type, (X - D * S)', S', lambda_Q, Q', alpha, beta)';

  fprintf('norm(Q) = %f\t||Q||_1 = %f\t%f%% sparsity\n', ...
	  norm(Q), sum(sum(abs(Q))), (nnz(Q) / prod(size(Q))) * 100);


  %fprintf('DONE LEARNING DICTIONARY\n');
  %pause;

  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeFullObjective(D, Q, S, X, lambda_S, lambda_Q));
    

  % Sparse codes update via feature-sign
  fprintf('SPARSE CODING STEP\n');
  if warm_start
    S = UpdateSparseCodes(type, X, D + Q, lambda_S, S, alpha, beta);
  else
    S = UpdateSparseCodes(type, X, D + Q, lambda_S, [], alpha, beta);
  end
  fprintf('norm(S) = %f\t||S||_1 = %f\t%f%% sparsity\n', ...
	  norm(S), sum(sum(abs(S))), (nnz(S) / prod(size(S))) * 100);
  %fprintf('DONE SPARSE CODING\n');
  %pause;

  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeFullObjective(D, Q, S, X, lambda_S, lambda_Q));
  
  
  % check convergence criterion - temporarily just 10 iterations
  % for debugging
  if iteration_num == max_iterations
    converged = true;
  end
end

fprintf('Done learning\n');
