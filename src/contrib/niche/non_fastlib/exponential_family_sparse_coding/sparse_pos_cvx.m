function x = sparse_neg_cvx(A, y, lambda)
%function x = sparse_neg_cvx(A, y, lambda)
%
% ||A x - y||_F^2 + lambda sum(x)
% s.t. x >= 0

n_dims = size(A, 2);

cvx_precision best
cvx_begin
  variable x(n_dims, 1)
  minimize square_pos(norm(A * x - y, 'fro')) + lambda * sum(x)
  subject to
    x >= zeros(n_dims, 1)
cvx_end
