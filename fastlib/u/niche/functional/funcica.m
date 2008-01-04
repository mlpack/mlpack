% funcica() - functional ICA
function [ic_curves_pos, ic_coef_pos, h_Y_pos] = funcica(t, s, data);


p = 30; % hardcoded for now

N = size(data,2);


mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);


myfd_data = data2fd(data, t, mybasis);
coef = getcoef(myfd_data);
%data1 = basis_curves * coef(:,1);
pca_results = pca_fd(myfd_data, p);
pc_coef = getcoef(pca_results.harmfd);
pc_curves = basis_curves * pc_coef;
pc_scores = pca_results.harmscr;




% p_small should be automatically selected according to some
% reconstruction error threshold

total_sum_var = 0;
for i = 1:p
  total_sum_var = total_sum_var + sum(pc_scores(:,i).^2);
end

sum_var = 0;
for p_small = 1:p
  sum_var = sum_var + sum(pc_scores(:,p_small).^2);
  if sum_var / total_sum_var > 0.999
    break
  end
end

p_small


sub_pc_coef = pc_coef(:,1:p_small);
E = pc_scores(:,1:p_small)';

[Y_pos,Y_neg,W_pos,W_neg] = find_opt_unmixing_matrix(E);

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



figure(1);
clf;
hold on;
plot(s, 'b');
plot(sub_pc_curves, 'r');
plot(ic_curves_pos, 'g');
plot(ic_curves_neg, 'c');



