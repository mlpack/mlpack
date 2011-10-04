function f = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z)
%function f = TaskCodingObjective(X, Y, W, Z, lambda_w, lambda_z)
%

n_tasks = size(Z, 2);

hinge_loss = 0;
for t = 1:n_tasks
  hinge_loss = hinge_loss ...
      + sum(max(1 - Y(:,t) .* (X(:,:,t)' * (W * Z(:,t))), 0));
end

f = hinge_loss + lambda_z * sum(sum(abs(Z))) + 0.5 * lambda_w * norm(W, 'fro')^2;
fprintf('Objective Value: %f\n', f);
