%function [lambda, diff_sum_H, pre_ICA_sum_H, post_ICA_sum_H] ...
function [lambda, ...
	  train_ICA_sum_H, test_ICA_sum_H, diff_sum_H, ...
	  train_ICA_sum_H_std, test_ICA_sum_H_std, diff_sum_H_std] ...
    = get_ICA_improvement(filename, ...
			  centered_rest_of_data_coef, ...
			  basis_inner_products);
% USAGE: [diff_sum_H, pre_ICA_sum_H, post_ICA_sum_H] =
% get_ICA_improvement(pc_scores, ic_scores)

%pwd
%fprintf('\n|%s|\n', filename);
%class(filename)
S = load(filename);
lambda = S.lambda;
ic_scores = S.ic_scores;
ic_coef = S.ic_coef;
pc_scores = S.pc_scores;

rest_ic_scores = ...
    get_scores(centered_rest_of_data_coef, ...
	       ic_coef, ...
	       basis_inner_products)';



used_ic_scores = rest_ic_scores;

%{
% check to ensure covariance matrix is orthogonal
cov_pc_scores = cov(pc_scores');
off_diag_max = maxall(cov_pc_scores - diag(diag(cov_pc_scores)));
if off_diag_max > eps
  [V,D] = eig(cov_pc_scores);
  whitening_transform = V * (D^(-.5)) * V';
else
  % the covariance matrix of pc_scores is already diagonal, but we need to
  % scale it so that cov(pc_scores') is white
  whitening_transform = diag(1./std(pc_scores'));
end

white_pc_scores= whitening_transform * pc_scores;
%}
%pre_ICA_sum_H = vasicek_sum(white_pc_scores);
%post_ICA_sum_H = vasicek_sum(used_ic_scores);
%diff_sum_H = pre_ICA_sum_H - post_ICA_sum_H;

train_ICA_sum_H = vasicek_sum(ic_scores);
test_ICA_sum_H = vasicek_sum(used_ic_scores);
diff_sum_H = test_ICA_sum_H - train_ICA_sum_H;

train_ICA_sum_H_std = vasicek_sum_std(ic_scores);
test_ICA_sum_H_std = vasicek_sum_std(used_ic_scores);
diff_sum_H_std = test_ICA_sum_H_std - train_ICA_sum_H_std;
