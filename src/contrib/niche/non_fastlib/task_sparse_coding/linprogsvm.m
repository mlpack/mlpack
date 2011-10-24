function w = linprogsvm(X, y, lambda)
%function w = linprogsvm(X, y, lambda)
%
% Enough of this ill-behaved LPSVM shit, time for good old simplex


[n_dims n_points] = size(X);

c = [zeros(n_dims, 1); ...
     ones(n_dims, 1); ...
     ones(n_points, 1) / lambda];

A = [[-bsxfun(@times, X, y')' zeros(n_points, n_dims) -eye(n_points)]; ...
     [-eye(n_dims) -eye(n_dims) zeros(n_dims, n_points)]; ...
     [eye(n_dims) -eye(n_dims) zeros(n_dims, n_points)]];

b = [-ones(n_points, 1); ...
     zeros(n_dims, 1); ...
     zeros(n_dims, 1)];

% the zero solution
%x = [zeros(n_dims, 1); ...
%     zeros(n_dims, 1); ...
%     ones(n_points, 1)];
%A * x - b

lower_bound = [-Inf * ones(n_dims, 1); ...
	       zeros(n_dims, 1); ...
	       zeros(n_points, 1)];

options = optimset('Display', 'notify', 'LargeScale', 'off', 'Simplex', 'on');
%options = optimset('LargeScale', 'off');
solution = linprog(c,A,b,[],[],lower_bound, [],[],options);

w = solution(1:n_dims);
%alpha = solution(n_dims+1:2*n_dims);
%xi = solution(2*n_dims+1:end);
%disp(xi);
