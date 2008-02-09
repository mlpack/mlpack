data = dlmread('combined.txt', '\t', 1, 1)';
data = data(7:7+17,:);
t = 0:7:119;





N = size(data,2);
p = 17;

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);

lambda = 0;
myfdPar = fdPar(mybasis, 2, lambda);

fprintf('calling funcica\n');
[ic_curves, ic_coef, Y, pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, myfd_data, p, basis_curves, myfdPar, basis_inner_products);
fprintf('funcica returned\n');

p_small = size(ic_coef, 2);

pc_curves = basis_curves * pc_coef;

ic_curves = pc_curves * W';

pc_coef = pc_coef';



% rescale the utilized parts of pc_coef such that
% the pc_curves square integrate to 1

for j = 1:p_small
  pc_coef_j = pc_coef(j,:);
  alpha = sqrt(sum(sum((pc_coef_j' * pc_coef_j) .* basis_inner_products)));
  pc_coef(j,:) = pc_coef(j,:) / alpha;
end

pc_scores = pc_scores';

ic_scores = W * pc_scores;


magnitude = sum(sum(pc_scores .^ 2));


for i = 1:p_small
  h_pc(i) = get_vasicek_entropy_estimate_std(pc_scores(i,:));
  h_ic(i) = get_vasicek_entropy_estimate_std(ic_scores(i,:));
end

h_pc_sum = sum(h_pc);
h_ic_sum = sum(h_ic);
logiszero