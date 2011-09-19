function [] = PoissonSparseCoding(X, k, lambda)
%function [] = poisson_sparse_coding(X, lambda)
%
% Given: X - a matrix where each column is a document, and each row
% corresponds to vocabulary word. X_{i,j} is the number of times
% word i occurs in document j

[d n] = size(X);

% Set Initial Dictionary
D = rand(d, k);


converged = false;

while ~converged
  % Dictionary update
  D = DictionaryProjectedGradient(D, S, X, 1e-4, 0.99);
  
  % Sparse codes update via feature-sign
  if exist('S')
    S = UpdateSparseCodes(D, lambda, S);
  else
    S = UpdateSparseCodes(D, lambda);
  end


  % check convergence criterion
end