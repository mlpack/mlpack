function [X_train,Y_train,X_test,Y_test,W,Z] = ...
    GenerateSyntheticData(n_dims, n_points, n_tasks, n_atoms)
%function [X_train,Y_train,X_test,Y_test,W,Z] = ...
%    GenerateSyntheticData(n_dims, n_points, n_tasks, n_atoms)

% generate the dictionary
W = normrnd(0, 1, n_dims, n_atoms);
W = W ./ repmat(sqrt(sum(W.^2)), n_dims, 1);

% generate sparse codes where non-zero entries lie 
% uniformly in [-2, -0.5] and [0.5, 2] with equal probability
a = 0.5;
b = 2;
Z = (1 - 2 * (rand(n_atoms, n_tasks) > 0.5)) ...
    .* (a + (b - a) * rand(n_atoms, n_tasks));
%Z = normrnd(0, 1, n_atoms, n_tasks);
for t = 1:n_tasks
  inds = randperm(n_atoms);
  selected_inds = inds(1:floor(0.8 * n_atoms));
  Z(selected_inds, t) = 0;
end

%Z(find(abs(Z) < 2)) = 0;

% generate data
X = normrnd(0,1, n_dims, 2 * n_points, n_tasks);
Y = zeros(2 * n_points, n_tasks);
for t = 1:n_tasks
  Y(:,t) = sign(X(:,:,t)' * W * Z(:,t));
end

X_train = X(:,1:n_points,:);
Y_train = Y(1:n_points,:);

X_test = X(:,(n_points + 1):end,:);
Y_test = Y((n_points + 1):end,:);
