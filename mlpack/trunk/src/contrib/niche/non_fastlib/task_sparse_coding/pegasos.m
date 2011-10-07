function w = pegasos(X, Y, lambda, mini_batch_size, n_iterations, w_initial)
%function w = pegasos(X, Y, lambda, mini_batch_size, n_iterations, w_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m

if nargin < 6
  w_initial = zeros(size(X, 1), 1);
end

if mini_batch_size == 1
  w = pegasos_trivial_minibatch(X, Y, lambda, n_iterations, w_initial);
else
  w = pegasos_minibatch_bottou(X, Y, lambda, mini_batch_size, n_iterations, w_initial);
end

n_errors = sum(Y .* (X' * w) <= 0);

fprintf('%f%% error\n', 100 * n_errors / size(X, 2));
