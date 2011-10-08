function w = pegasos_minibatch_bottou2(X, Y, lambda, minibatch_size, n_iterations, w_initial)
%function w = pegasos_minibatch_bottou2(X, Y, lambda, minibatch_size, n_iterations, w_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m
%
% Sampling without replacement by:
%   1) Exhausting all examples sequentially from a random permutation
%   2) Creating a new random permutation
%   3) Repeating

[n_dims n_points] = size(X);


inds = randperm(n_points);

w = w_initial;

cur_ind = 1;

for t = 1:n_iterations
  step_size = 1 / (lambda * t);

  
  if cur_ind == (n_points + 1)
    cur_ind = 1;
    inds = randperm(n_points);
  end
  
  draws = cur_ind:min((cur_ind + minibatch_size - 1), n_points);
  rand_inds = inds(draws);
  cur_ind = draws(end) + 1;
  subgrad = zeros(n_dims, 1);
  for i = 1:length(draws)
    
    X_t = X(:,rand_inds(i));
    Y_t = Y(rand_inds(i));
  
    if Y_t * w' * X_t < 1
      subgrad = subgrad + Y_t * X_t;
    end
  end
  w = (1 - step_size * lambda) * w + (step_size / minibatch_size) * subgrad;  

  % project onto the (1 / sqrt(lambda)) ball - is this step needed
  % for the best version of Pegasos?
  norm_w = norm(w);
  if norm_w > 1 / sqrt(lambda)
    w = w * (1 / sqrt(lambda)) / norm_w;
  end
end
