function [D, S] = ExpFamSparseCoding(type, X, k, lambda, max_iterations, ...
				     warm_start)
%function [D, S] = ExpFamSparseCoding(type, X, k, lambda, max_iterations, ...
%				     warm_start)
%
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j
% Good news: warm start is faster
% Bad news:  warm start yields poorer solutions than starting at 0

if nargin < 6
  warm_start = false;
end

if type == 'p'
  @(D, S, X, lambda) ComputeFullObjective = ...
      ComputePoissonFullObjective(D, S, X, lambda);
  
elseif type == 'b'
  @(D, S, X, lambda) ComputeFullObjective = ...
      ComputeBernoulliFullObjective(D, S, X, lambda);
  
end


[d n] = size(X);

% hardcoded for now
alpha = 1e-4;
beta = 0.9;

% Set Initial Dictionary
D = normcols(normrnd(0,1,d,k));

% Sparse codes update via feature-sign
fprintf('INITIAL SPARSE CODING STEP\n');
S = UpdateSparseCodes(X, D, lambda, [], alpha, beta);
fprintf('norm(S) = %f\n', norm(S));
%fprintf('DONE SPARSE CODING\n');
%pause;

fprintf('\t\t\tObjective value: %f\n', ...
	ComputeFullObjective(D, S, X, lambda));


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
  D = DictionaryProjectedGradient(D, S, X, alpha, beta, 2);
  %fprintf('DONE LEARNING DICTIONARY\n');
  %pause;

  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeFullObjective(D, S, X, lambda));
    

  % Sparse codes update via feature-sign
  fprintf('SPARSE CODING STEP\n');
  if warm_start
    S = UpdateSparseCodes(X, D, lambda, S, alpha, beta);
  else
    S = UpdateSparseCodes(X, D, lambda, [], alpha, beta);
  end
  fprintf('norm(S) = %f\n', norm(S));
  %fprintf('DONE SPARSE CODING\n');
  %pause;

  fprintf('\t\t\tObjective value: %f\n', ...
	  ComputeFullObjective(D, S, X, lambda));
  
  
  % check convergence criterion - temporarily just 10 iterations
  % for debugging
  if iteration_num == max_iterations
    converged = true;
  end
end

fprintf('Done learning\n');
