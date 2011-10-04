function Z = TaskCodingZStep(X, Y, W, lambda_z)
%function Z = TaskCodingZStep(X, Y, W, lambda_z)
%

n_tasks = size(Y, 2);
n_atoms = size(W, 2);

Z = zeros(n_atoms, n_tasks);
for t = 1:n_tasks
  V = W' * X(:,:,t);
  Z(:,t) = lpsvm(V', Y(:,t), 0, 1 / lambda_z, 0);
end
