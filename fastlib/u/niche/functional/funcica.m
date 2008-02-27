function [ic_curves_pos, ic_coef_pos, ic_scores, ...
	  sub_pc_coef, sub_pc_curves, sub_pc_scores, W] = ...
    funcica(t, myfd_data, p, basis_curves, myfdPar, basis_inner_products);
% funcica() - functional ICA
% first call prelim_funcica
% USAGE: [ic_curves_pos, ic_coef_pos, h_Y_pos] = funcica(t, s, data)




data_coef = getcoef(myfd_data);
pca_results = pca_fd(myfd_data, 2, myfdPar);
pc_coef = getcoef(pca_results.harmfd);
pc_curves = basis_curves * pc_coef;

% pc_scores = pca_results.harmscr;% this doesn't work if lambda > 0
% instead, we do:
pc_scores = get_scores(data_coef, pc_coef, basis_inner_products);

%mean_coef = getcoef(pca_results.meanfd);



% p_small should be automatically selected according to some
% reconstruction error threshold
threshold = 0.9;
p_small = min(find((cumsum(pca_results.varprop) >= threshold) == 1))
%p_small = p;

%total_sum_var = 0;
%for i = 1:p
%  total_sum_var = total_sum_var + sum(pc_scores(:,i).^2);
%end

%sum_var = 0;
%for p_small = 1:p
%  sum_var = sum_var + sum(pc_scores(:,p_small).^2);
%  disp(sprintf('i = %d, sum_var = %f', p_small, sum_var / total_sum_var));
%  if sum_var / total_sum_var > 0.9
%    break
%  end
%end

fprintf('p_small = %d\n', p_small);


sub_pc_coef = pc_coef(:,1:p_small);
sub_pc_curves = basis_curves * sub_pc_coef;
sub_pc_scores = pc_scores(:,1:p_small)';

%{
inv_pc_coef = inv(pc_coef);
calc_pc_scores = ...
    inv_pc_coef * (data_coef - ...
		   repmat(getcoef(pca_results.meanfd), 1, ...
			  size(data_coef, 2)));

disp(sprintf('the difference is %f', maxall(calc_pc_scores' - pc_scores)));
%}



% check to ensure covariance matrix is orthogonal
cov_sub_pc_scores = cov(sub_pc_scores');
off_diag_max = maxall(cov_sub_pc_scores - diag(diag(cov_sub_pc_scores)));
if off_diag_max > eps
  fprintf('covariance matrix is not orthogonal!\n');
  fprintf('off_diag_max = %f\n', off_diag_max);
  [V,D] = eig(cov_sub_pc_scores);
  whitening_transform = V * (D^(-.5)) * V';
else
  % the covariance matrix of sub_pc_scores is already diagonal, but we need to
  % scale it so that cov(sub_pc_scores') is white
  whitening_transform = diag(1./std(sub_pc_scores'));
end


white_sub_pc_scores = whitening_transform * sub_pc_scores;

% Use my simplified version of RADICAL
%[Y_pos, Y_neg, post_whitening_W_pos, post_whitening_W_neg] = ...
%    find_opt_unmixing_matrix(white_E);

% Use RADICAL
[ic_scores, W] = RADICAL(white_sub_pc_scores);

W = W * whitening_transform;

ic_coef_pos = (W * sub_pc_coef')';
ic_curves_pos = basis_curves * ic_coef_pos;
