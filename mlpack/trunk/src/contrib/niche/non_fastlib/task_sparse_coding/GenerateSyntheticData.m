function [X,Y,W,Z] = GenerateSyntheticData(n_dims, n_points, n_tasks, n_atoms)
%function [X,Y,W,Z] = GenerateSyntheticData(n_dims, n_points, n_tasks, n_atoms)

% generate the dictionary
W = normrnd(0, 1, n_dims, n_atoms);

% generate sparse codes
Z = normrnd(0, 1, n_atoms, n_tasks);
Z(find(abs(Z) < 0.5)) = 0;

% generate data
X = normrnd(0,1, n_dims, n_points, n_tasks);
Y = zeros(n_points, n_tasks);
for t = 1:n_tasks
  Y(:,t) = sign(X(:,:,t)' * W * Z(:,t));
end
