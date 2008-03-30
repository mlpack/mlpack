%function [lambda, ...
%	  train_ICA_sum_H, test_ICA_sum_H, diff_sum_H, ...
%	  train_ICA_sum_H_std, test_ICA_sum_H_std, diff_sum_H_std] ...
function [lambda, Q] ...
    = get_ICA_improvement(filename, ...
			  centered_rest_of_data_coef, ...
			  basis_inner_products);
% USAGE: [diff_sum_H, pre_ICA_sum_H, post_ICA_sum_H] =
% get_ICA_improvement(pc_scores, ic_scores)

S = load(filename);
lambda = S.lambda;
ic_scores = S.ic_scores;
ic_coef = S.ic_coef;
pc_scores = S.pc_scores;
pc_coef = S.pc_coef;

%{
disp('calculating rest of ic scores');

rest_ic_scores = ...
    get_scores(centered_rest_of_data_coef, ...
	       ic_coef, ...
	       basis_inner_products)';
%}
%disp('calculating rest of pc scores');

%rest_pc_scores = ...
%    get_scores(centered_rest_of_data_coef, ...
%	       pc_coef, ...
%	       basis_inner_products)';

disp('calculating entropies');

%used_ic_scores = rest_ic_scores;
used_ic_scores = ic_scores;
%used_pc_scores = rest_pc_scores;

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

h_norm = log(sqrt(2*pi*exp(1)));

num_ics = size(used_ic_scores, 1);

h_ic = zeros(1, num_ics);
h_ic_std = zeros(1, num_ics);

for ic_num = 1:num_ics
  h_ic(ic_num) = ...
      get_vasicek_entropy_estimate(used_ic_scores(ic_num,:));
  h_ic_std(ic_num) = ...
      get_vasicek_entropy_estimate_std(used_ic_scores(ic_num,:));
end

Q = sum(1 ./ (h_norm - h_ic_std));

