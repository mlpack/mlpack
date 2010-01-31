function scores = get_scores(data_coef, components_coef, ...
					basis_inner_products);
% this function is for use with the fdaM toolbox
% USAGE: scores = get_scores(data_coef, components_coef, basis_inner_products)
% data_coef is a matrix of size num_basis by N
% components_coef is a matrix of size num_basis by num_components
% basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)))
% scores is a matrix of size N by num_components


[num_basis, num_components] = size(components_coef);
N = size(data_coef, 2);

scores = zeros(N, num_components);

for j = 1:num_components
  components_coef_j = components_coef(:,j)';
  for i = 1:N
    scores(i,j) = ...
	sum(sum((data_coef(:,i) * ...
		 components_coef_j) .* basis_inner_products));
  end
end
