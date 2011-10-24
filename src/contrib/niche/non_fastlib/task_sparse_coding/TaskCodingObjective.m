function f = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z)
%function f = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z)
%

n_points = size(Y, 1);
n_tasks = size(Z, 2);

hinge_loss = 0;
n_errors = 0;
for t = 1:n_tasks
  predictions = Y(:,t) .* (X(:,:,t)' * (W * Z(:,t)));
  hinge_loss = hinge_loss + sum(max(1 - predictions, 0));
  n_errors = n_errors + sum(predictions <= 0);
end

fprintf('\t\t\t\t\t\t\t\t\t\tTraining Error: %f%%\n', 100 * n_errors / (n_tasks * n_points));
f = hinge_loss + lambda_z * sum(sum(abs(Z))) + 0.5 * lambda_w * norm(W, 'fro')^2;
fprintf('\t\t\t\tObjective Value: %f\n', f);
