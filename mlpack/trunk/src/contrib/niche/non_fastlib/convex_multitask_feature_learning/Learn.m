function [D, W] = Learn(X, Y, lambda, epsilon, n_iterations)
%function [D, W] = Learn(X, Y, lambda, epsilon, n_iterations)
%
% X is tensor in R^{n_dims \times n_points \times n_tasks}
% Y is a matrix in R^{n_points \times n_tasks}
% lambda is the regularization parameter for the squared 2,1 norm
% penalty on A (A being from the original problem formulation)
% epsilon - initial perturbation magnitude


[n_dims, n_points, n_tasks] = size(X);

% for pegasos
%minibatch_size = 10;
%n_iterations = 10 * n_points / minibatch_size;

%tol = 1e-9;

D = eye(n_dims) / n_dims;

%epsilon_I = epsilon * eye(n_dims);

W = zeros(n_dims, n_tasks);
W_prev = W + 1;

%sqrt_D = D^0.5;

run_once = false;

while epsilon > eps || ~run_once
  
  run_once = true;

  %epsilon_I = epsilon * eye(n_dims);

  % after epsilon reaches min_epsilon, solve the unperturbed problem
  %if epsilon < min_epsilon
  %  epsilon = 0;
  %end
  
  %fprintf('OUTER ITERATION: epsilon = %e\n', epsilon);

  
  [U,S,V] = svd(D);
  S = diag(S);

  sqrt_S = zeros(size(S));
  nz_inds = find(S > eps);
  sqrt_S(nz_inds) = 1 ./ S(nz_inds);
  sqrt_S = sqrt(sqrt_S);
  sqrt_S(nz_inds) = 1 ./ sqrt_S(nz_inds);
  
  sqrt_D = U * diag(sqrt_S) * U';
  
  
  for iteration_num = 1:n_iterations
    for t = 1:n_tasks
      % solve SVM problem
      %W(:,t) = pegasos(sqrt_D * X(:,:,t), Y(:,t), lambda, minibatch_size, n_iterations);
      
      % solve least squares problem
      X_t = sqrt_D * X(:,:,t);
      % next two lines equivalent via Woodbury identity
      %W(:,t) = (X_t * X_t' + lambda * eye(n_dims)) \ (X_t * Y(:,t));
      %W(:,t) = (X_t / (X_t' * X_t + lambda * eye(n_dims))) * Y(:,t);
      W(:,t) = (X_t * inv(X_t' * X_t + lambda * eye(n_dims))) * Y(:,t);
    end
    W = sqrt_D * W;
    
    [U,S,V] = svd(W);
    S = diag(S);
    S = sqrt(S.^2 + epsilon);
    S = S / sum(S);
    
    D = U * diag(S) * U';

    sqrt_S = zeros(size(S));
    nz_inds = find(S > eps);
    sqrt_S(nz_inds) = 1 ./ S(nz_inds);
    sqrt_S = sqrt(sqrt_S);
    sqrt_S(nz_inds) = 1 ./ sqrt_S(nz_inds);

    sqrt_D = U * diag(sqrt_S) * U'; % compute sqrt_D for use in next round
    %save mehta sqrt_D;
  end
  
  epsilon = epsilon / 10;
end