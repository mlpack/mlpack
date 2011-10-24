function W = svm_lobomargin_block(X, y, n_atoms, W)

n_dims = size(X, 1) / n_atoms;
fprintf('n_dims = %f\n', n_dims);
n_points = length(y);
fprintf('n_points = %f\n', n_points);

A = X * diag(y);

n_iterations = 0;

converged = false;
while ~converged
  fprintf('n_iterations = %d\n', n_iterations);
  for j = 1:2:n_atoms
    %active_set = ((j - 1) * n_dims + 1):(j * n_dims);
    %inactive_set = [1:((j - 1) * n_dims)  ...
%		    (j * n_dims + 1):(n_atoms * n_dims)];

    active_set = ((j - 1) * n_dims + 1):((j+2) * n_dims);
    inactive_set = [1:((j - 1) * n_dims)  ...
		    ((j+2) * n_dims + 1):(n_atoms * n_dims)];
    
    W_inactive = W(inactive_set);
    A_active = A(active_set,:);
    A_inactive = A(inactive_set,:);
    AW_inactive = A_inactive' * W_inactive;

    cvx_begin
      variable W_active1(n_dims)
      variable W_active2(n_dims)
      variable W_active3(n_dims)
      variable xi(n_points)
      minimize sum(xi) / n_points
      subject to
        AW_inactive + A_active' * [W_active1; W_active2; W_active3] >= 1 - xi
        xi >= 0
        W_active1 == norm_ball(n_dims)
        W_active2 == norm_ball(n_dims)
        W_active3 == norm_ball(n_dims)
    cvx_end
    if isequal(cvx_status, 'Solved')
      W(active_set) = [W_active1; W_active2; W_active3];
    else
      %error('cvx_status = ''%s''\n', cvx_status);
      % do nothing
    end

  end

  n_iterations = n_iterations + 1;
  if n_iterations == 2
    converged = true;
  end
end
