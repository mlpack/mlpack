function w = pegasos_minibatch(X, Y, lambda, minibatch_size, n_iterations, w_initial)
%function w = pegasos_minibatch(X, Y, lambda, minibatch_size, n_iterations, w_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m

[n_dims n_points] = size(X);


inds = 1:n_points;

w = w_initial;

for t = 1:n_iterations
  step_size = 1 / (lambda * t);

  subgrad = zeros(n_dims, 1);
  for i = 0:(minibatch_size-1)
    draw = randint(1,1,n_points - i) + 1;
    ind = inds(draw);
    inds(draw) = inds(n_points - i);
    inds(n_points - i) = ind;
    
    X_t = X(:,ind);
    Y_t = Y(ind);
  
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
