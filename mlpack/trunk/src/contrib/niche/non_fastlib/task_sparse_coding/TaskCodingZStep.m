function Z = TaskCodingZStep(X, Y, W, lambda_z)
%function Z = TaskCodingZStep(X, Y, W, lambda_z)
%

n_tasks = size(Y, 2);
n_atoms = size(W, 2);

output = 0;
nu = 1 / lambda_z;
Z = zeros(n_atoms, n_tasks);
for t = 1:n_tasks
  V = W' * X(:,:,t);
  [Z(:,t), gamma] = lpsvm(V', Y(:,t), 1, nu, output, 10);
  %fprintf('gamma = %f\n', gamma);
end
