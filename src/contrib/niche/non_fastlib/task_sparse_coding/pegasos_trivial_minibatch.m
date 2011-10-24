function w = pegasos_trivial_minibatch(X, Y, lambda, n_iterations, w_initial)
%function w = pegasos_trivial_minibatch(X, Y, lambda, n_iterations, w_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m

[n_dims n_points] = size(X);

w = w_initial;

% trivial mini-batch case, when mini_batch_size == 1
for t = 1:n_iterations
  ind = randint(1,1,n_points) + 1;
  X_t = X(:,ind);
  Y_t = Y(ind);
  
  step_size = 1 / (lambda * t);
  if Y_t * w' * X_t < 1
    w = (1 - step_size * lambda) * w + step_size * Y_t * X_t;
  else
    w = (1 - step_size * lambda) * w;
  end
  
  % project onto the (1 / sqrt(lambda)) ball - is this step needed
  % for the best version of Pegasos?
  norm_w = norm(w);
  if norm_w > 1 / sqrt(lambda)
    w = w * (1 / sqrt(lambda)) / norm_w;
  end
end
