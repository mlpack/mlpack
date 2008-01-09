function [ic_curves_pos, ic_coef_pos, h_Y_pos, pc_coef, pc_curves, mean_coef, W_pos] = ...
    funcica(t, s, myfd_data_train, p, basis_curves, myfdPar);
% funcica() - functional ICA
% first call prelim_funcica
% USAGE: [ic_curves_pos, ic_coef_pos, h_Y_pos] = funcica(t, s, data)




data_train_coef = getcoef(myfd_data_train);
pca_results = pca_fd(myfd_data_train, p, myfdPar);
%pca_results = pca_fd(myfd_data_train, p);
% plot_pca(pca_results);
pc_coef = getcoef(pca_results.harmfd);
pc_curves = basis_curves * pc_coef;
pc_scores = pca_results.harmscr;
mean_coef = getcoef(pca_results.meanfd);

%figure(1); plot(t, pc_curves(:,1));
%figure(2); plot(t, pc_curves(:,2));




% p_small should be automatically selected according to some
% reconstruction error threshold
%{
total_sum_var = 0;
for i = 1:p
  total_sum_var = total_sum_var + sum(pc_scores(:,i).^2);
end

sum_var = 0;
for p_small = 1:p
  sum_var = sum_var + sum(pc_scores(:,p_small).^2);
  disp(sprintf('i = %d, sum_var = %f', p_small, sum_var / total_sum_var));
  if sum_var / total_sum_var > 0.9
    break
  end
end
%}
p_small = 2;

p_small


sub_pc_coef = pc_coef(:,1:p_small);
E = pc_scores(:,1:p_small)';

%{
inv_pc_coef = inv(pc_coef);
calc_pc_scores = ...
    inv_pc_coef * (data_train_coef - ...
		   repmat(getcoef(pca_results.meanfd), 1, ...
			  size(data_train_coef, 2)));

disp(sprintf('the difference is %f', maxall(calc_pc_scores' - pc_scores)));
%}

[Y_pos,Y_neg,W_pos,W_neg] = find_opt_unmixing_matrix(E);

%save('YWE.mat', 'E', 'Y_pos', 'W_pos');

for i = 1:p_small
  h_E(i) = get_vasicek_entropy_estimate(E(i,:));
  h_Y_pos(i) = get_vasicek_entropy_estimate(Y_pos(i,:));
  h_Y_neg(i) = get_vasicek_entropy_estimate(Y_neg(i,:));
end



ic_coef_pos = (W_pos * sub_pc_coef')';
ic_coef_neg = (W_neg * sub_pc_coef')';

sub_pc_curves = basis_curves * sub_pc_coef;
ic_curves_pos = basis_curves * ic_coef_pos;
ic_curves_neg = basis_curves * ic_coef_neg;


%{
figure(1);
clf;
hold on;
plot(s, 'b');
plot(sub_pc_curves, 'r');
plot(ic_curves_pos, 'g');
plot(ic_curves_neg, 'c');
%}


