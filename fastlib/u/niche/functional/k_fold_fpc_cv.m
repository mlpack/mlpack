function [alpha] = k_fold_fpc_cv(data, t, k);
% USAGE: data is a vector of size num_dims by num_points

%t = 1/240:1/240:1;

p = 120;

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));


total_count = size(data, 2);

%shuffled_indices = shuffle(1:total_count);
%shuffled_indices = shuffled_indices(1:round(end/10));
%total_count = length(shuffled_indices);
shuffled_indices = 1:total_count;

fold_counts = floor(total_count / k) * ones(k, 1);

overflow_counts = total_count - sum(fold_counts);

fold_counts(1:overflow_counts) = ...
    fold_counts(1:overflow_counts) + 1;

fold_indices = cell(k,1);

last = 0;
for i = 1:k
  fold_indices{i} = shuffled_indices((last+1):(last + fold_counts(i)));
  last = last + fold_counts(i);
end


for i = 1:k
  fprintf('preparing fold %d\n', i);
  
  training_indices = [];
  for j = [1:(i-1) (i+1):k]
    training_indices = [training_indices fold_indices{j}];
  end
  
  training_data = data(:, training_indices);
  
  training_fd = data2fd(training_data, t, mybasis);
  
  mean_result = pca_fd(training_fd, 0);
  centered_training_data_coef = ...
      getcoef(training_fd) - ...
    repmat(getcoef(mean_result.meanfd), ...
	   1, size(getcoef(training_fd), 2));
  centered_training_fd_cells{i} = ...
      fd(centered_training_data_coef, mybasis);
  
  
  test_indices = fold_indices{i};
  
  test_data = data(:, test_indices);
  
  test_fd = data2fd(test_data, t, mybasis);
  
  mean_result = pca_fd(test_fd, 0);
  centered_test_data_coef = ...
      getcoef(test_fd) - ...
    repmat(getcoef(mean_result.meanfd), ...
	   1, size(getcoef(test_fd), 2));
  centered_test_fd_cells{i} = ...
      fd(centered_test_data_coef, mybasis);

end


lambda_set = [0 1e-9 1e-8 1e-7 1e-6 1e-5];

alpha = zeros(length(lambda_set), k);

for lambda_i = 1:length(lambda_set)
  
  lambda = lambda_set(lambda_i);
  
  disp(lambda);

  myfdPar = fdPar(mybasis, 2, lambda);
  
  for i = 1:k
    fprintf('processing fold %d\n', i);
    
    pca_results = pca_fd(centered_training_fd_cells{i}, p, myfdPar);
    pc_coef = getcoef(pca_results.harmfd);
    
    for j = 1:p
      pc_coef_j = pc_coef(:,j);
      pc_j_norm = sqrt(sum(sum((pc_coef_j * pc_coef_j') .* basis_inner_products)));
      pc_coef(:,j) = pc_coef(:,j) / pc_j_norm;
    end

    disp('calculating scores');
    test_scores = get_scores(getcoef(centered_test_fd_cells{i}), ...
			     pc_coef, basis_inner_products);
    disp('done');
    
    alpha(lambda_i, i) = norm(test_scores, 'fro');

  end
end

% [y,i] = sort(alpha, 'descend')