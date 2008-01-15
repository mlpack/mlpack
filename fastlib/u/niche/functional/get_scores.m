function scores = get_scores(data_coef, pc_coef, ...
					basis_inner_products);
% USAGE: scores = get_scores(data_coef, pc_coef, basis_inner_products)
% data_coef is a matrix of size num_pc by N
% pc_coef is a matrix of num_pc by num_basis
% basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0))) 
% scores is a matrix of size N by num_pc


[num_pc, num_basis] = size(pc_coef);
N = size(data_coef, 2);

scores = zeros(N, num_pc);

for j = 1:num_pc
  pc_coef_j = pc_coef(j,:);
  for i = 1:N
    scores(i,j) = ...
	sum(sum((data_coef(:,i) * ...
		 pc_coef_j) .* basis_inner_products));
  end
end
