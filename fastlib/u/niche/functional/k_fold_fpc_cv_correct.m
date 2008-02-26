%function [alpha] = k_fold_fpc_cv_correct(data, t, k);
% USAGE: data is a vector of size num_dims by num_points

t = 1/240:1/240:1;
k = 10;
m = 120;
p = 12;

mybasis = create_bspline_basis([min(t) max(t)], m, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));
basis_inner_products_diag = diag(basis_inner_products);


total_count = size(data, 2);

shuffled_indices = shuffle(1:total_count);
%shuffled_indices = shuffled_indices(1:round(end/10));
%shuffled_indices = 1:round(total_count);
total_count = length(shuffled_indices);

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


fprintf('preparing folds');
for i = 1:k
  fprintf(', %d', i);

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
  
  test_fd_cells{i} = data2fd(test_data, t, mybasis);

end

fprintf('\n');
return;
% use matlab's built-in bounded local optimizer
[lambda_opt, alpha_opt, exitflag] = ...
    fminbnd(@(lambda) ...
	    k_fold_fpc_cv_correct_opt(lambda, k, mybasis, m, p, ...
				      centered_training_fd_cells, ...
				      test_fd_cells, basis_inner_products, ...
				      basis_inner_products_diag), ...
	    0, 1e-4, optimset('TolX', 1e-9));


% variance of all test data (not just residuals) is 1.6280E6
% variance of alpha_opt (residual for optimal lambda) is 0.0947
% 10% cutoff point is 1.6280E5
% 6.6E-5 -> alpha = 1.62780E5