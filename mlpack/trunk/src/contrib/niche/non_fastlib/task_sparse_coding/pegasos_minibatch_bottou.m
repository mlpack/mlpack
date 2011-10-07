function w = pegasos_minibatch_bottou(X, Y, lambda, minibatch_size, n_iterations, w_initial)
%function w = pegasos_minibatch_bottou(X, Y, lambda, minibatch_size, n_iterations, w_initial)
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
X = X(:,inds); % perhaps its faster to reorder the data now rather
               % than picking random columns later
Y = Y(inds);

w = w_initial;

cur_ind = 1;

for t = 1:n_iterations
  step_size = 1 / (lambda * t);

  
  if cur_ind == (n_points + 1)
    cur_ind = 1;
    inds = randperm(n_points);
    X = X(:,inds);
    Y = Y(inds);
  end
  
  draws = cur_ind:min((cur_ind + minibatch_size - 1), n_points);
  cur_ind = draws(end) + 1;
  
  subgrad = zeros(n_dims, 1);
  
  % slow subgradient computation, shown for code readability
  %for i = 1:length(draws)
  %  X_t = X(:,draws(i));
  %  Y_t = Y(draws(i));
  %  
  %  if Y_t * w' * X_t < 1
  %    subgrad = subgrad + Y_t * X_t;
  %  end
  %end
  
  % fast subgradient computation
  margin_error_inds = draws(find(Y(draws)' .* (w' * X(:,draws)) < 1));
  subgrad = subgrad + ...
	    sum(bsxfun(@times, X(:,margin_error_inds), Y(margin_error_inds)'), 2);
  
  w = (1 - step_size * lambda) * w + (step_size / minibatch_size) * subgrad;  

  % project onto the (1 / sqrt(lambda)) ball - is this step needed
  % for the best version of Pegasos?
  norm_w = norm(w);
  if norm_w > 1 / sqrt(lambda)
    w = w * (1 / sqrt(lambda)) / norm_w;
  end
end
