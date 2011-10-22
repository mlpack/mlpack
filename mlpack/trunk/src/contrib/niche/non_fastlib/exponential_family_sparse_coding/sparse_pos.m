function x = sparse_pos(A, y, lambda)
%function x = sparse_pos(A, y, lambda)
%
% ||A x - y||_F^2 + lambda sum(x)
% s.t. x >= 0

n_dims = size(A, 2);

x = zeros(n_dims, 1);
active_set = zeros(1, n_dims);
active_inds = [];
A_active = [];

converged = false;

while ~converged

  max_i = -1;
  max_val = 0;
  for i = find(active_set == 0)
    val = - 2 * A(:,i)' * (A * x - y);
    if val > max_val
      max_val = val;
      max_i = i;
    end
  end
  if max_val > lambda
    active_set(max_i) = 1;
    active_inds = [active_inds max_i];
    A_active = [A_active A(:,max_i)];
  else
    return;
  end 
  
  x_active = (A_active' * A_active) \ (A_active' * y - lambda / 2);

  x(active_inds) = x_active;
  
  converged = true;
  for i = 1:n_dims
    val = 2 * A(:,i)' * (A * x - y);
    if active_set(i)
      if abs(val + lambda) > 1000 * eps
	error('condition 1 failed with %e', val + lambda);
      end
    else
      if -val > lambda
	converged = false;
      end
    end
  end
end
