function alpha = ...
    k_fold_fpc_cv_correct_opt(lambda, k, mybasis, m, p, ...
			      centered_training_fd_cells, ...
			      test_fd_cells, basis_inner_products, ...
			      basis_inner_products_diag);
% Objective function to minimize

myfdPar = fdPar(mybasis, 2, lambda);

alpha = 0;

fprintf('processing folds');
for i = 1:k

  fprintf(', %d', i);
  
  pca_results = pca_fd(centered_training_fd_cells{i}, p, myfdPar);
  pc_coef = getcoef(pca_results.harmfd);
  
  for j = 1:p
    pc_coef_j = pc_coef(:,j);
    pc_j_norm = sqrt(sum(sum((pc_coef_j * pc_coef_j') .* basis_inner_products)));
    pc_coef(:,j) = pc_coef(:,j) / pc_j_norm;
  end
  
  test_scores = get_scores(getcoef(test_fd_cells{i}), ...
			   pc_coef, basis_inner_products);
  
  residual_coef = ...
      getcoef(test_fd_cells{i})' - test_scores * pc_coef';
  
  %  alpha = alpha + ...
  %	  sum((residual_coef * basis_inner_products_diag).^2);

  for l = 1:size(residual_coef, 1)
    alpha = alpha + ...
	    sum(sum((residual_coef(l,:)' * residual_coef(l,:)) .* ...
		    basis_inner_products));
    end
  
end

fprintf('\n');

fprintf('lambda = %f\talpha = %f\n', lambda, alpha);
