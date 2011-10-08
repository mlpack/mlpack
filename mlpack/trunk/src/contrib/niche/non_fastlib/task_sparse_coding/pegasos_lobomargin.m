function W = pegasos_lobomargin(X, Y, mini_batch_size, n_iterations, n_atoms, W_initial)
%function W = pegasos_lobomargin(X, Y, mini_batch_size, n_iterations, n_atoms, W_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m

if nargin < 7
  W_initial = zeros(size(X, 1), 1);
end

if mini_batch_size == 1
  W = pegasos_lobomargin_trivial_minibatch_bottou(X, Y, n_iterations, n_atoms, W_initial);
else
  W = pegasos_lobomargin_minibatch_bottou(X, Y, mini_batch_size, ...
					  n_iterations, n_atoms, W_initial);
end

%n_errors = sum(Y .* (X' * w) <= 0);

%fprintf('%f%% error\n', 100 * n_errors / size(X, 2));
