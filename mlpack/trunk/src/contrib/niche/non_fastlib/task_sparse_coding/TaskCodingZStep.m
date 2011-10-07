function Z = TaskCodingZStep(X, Y, W, lambda_z)
%function Z = TaskCodingZStep(X, Y, W, lambda_z)
%

n_tasks = size(Y, 2);
n_atoms = size(W, 2);

delta= 100;
output = 0;
nu = 1 / lambda_z;
Z = zeros(n_atoms, n_tasks);
parfor t = 1:n_tasks
  fprintf('task %d\n', t);
  V = W' * X(:,:,t);
  %[Z(:,t), gamma, trainCorr, testCorr, cpu_time, anu] = lpsvm(V', Y(:,t), 1, nu, output, delta);
  Z(:,t) = linprogsvm2(V, Y(:,t), lambda_z);
  %fprintf('gamma = %f\n', gamma);
end
