function [D, W] = Learn(X, Y, lambda, epsilon, n_iterations, loss_type)
%function [D, W] = Learn(X, Y, lambda, epsilon, n_iterations, loss_type)
%
% X is tensor in R^{n_dims \times n_points \times n_tasks}
% Y is a matrix in R^{n_points \times n_tasks}
% lambda is the regularization parameter for the squared 2,1 norm
% penalty on A (A being from the original problem formulation)
% epsilon - initial perturbation magnitude
% loss_type - either 'squared' for squared loss or 'hinge' for hinge loss (svm)
%
% Notes: Currently, the code just chooses the W corresponding to
% the smallest epsilon such that epsilon > eps (where we
% geometrically decrease the initial value of epsilon by a factor
% of 10 in outer iteration). Hence, for now, you might as well just call
% this program with epsilon = eps * 2.

[n_dims, n_points, n_tasks] = size(X);

HINGE = 0;
SQUARED = 1;
loss_code = -1;
if isequal(loss_type, 'hinge')
  loss_code = HINGE;

  % for pegasos
  minibatch_size = 10;
  n_iterations = 10 * n_points / minibatch_size;
elseif isequal(loss_type, 'squared')
  loss_code = SQUARED;
else
  error('Error: ''%s'' is not a valid setting for loss_type\n', ...
	loss_type);
end

W = zeros(n_dims, n_tasks);

while epsilon > eps

  D = eye(n_dims) / n_dims;

  %epsilon_I = epsilon * eye(n_dims);

  % after epsilon reaches min_epsilon, solve the unperturbed problem
  %if epsilon < eps
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
      if loss_code == HINGE
	% solve SVM problem
	W(:,t) = pegasos(sqrt_D * X(:,:,t), Y(:,t), lambda, ...
			 minibatch_size, n_iterations);
      elseif loss_code == SQUARED
	% solve least squares problem
	X_t = sqrt_D * X(:,:,t);
	% next two lines equivalent via Woodbury identity
	%W(:,t) = (X_t * X_t' + lambda * eye(n_dims)) \ (X_t * Y(:,t));
	W(:,t) = (X_t / (X_t' * X_t + lambda * eye(n_points))) * Y(:,t);
	%W(:,t) = (X_t * inv(X_t' * X_t + lambda * eye(n_points))) * Y(:,t);
      end
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
  end
  
  epsilon = epsilon / 10;
end
