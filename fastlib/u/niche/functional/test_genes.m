load_gene_data;

t = 0:7:119;
first = 7;
span = 18;

window = first:(first + span - 1);

% filter out those observations that are missing more than one value
good_indices = [];
for i = 1:size(data, 2)
  if length(find(data(window, i) == -inf)) <= 1
    good_indices(end+1) = i;
  end
end

data = data(window, good_indices);
phases = phases(good_indices);
clusters = clusters(good_indices);

% indicate missing values with NaN, as required by the fd tools data2fd()
data(find(data == -inf)) = NaN;

% data ready!


N = size(data,2);
p = 17;

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(0:1:119, mybasis);
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





% discriminant analysis using pc features %
% pc_scores is d x N

g1_indices = find(phases == G1_phase);
nong1_indices = find(phases ~= G1_phase);

svm_data = [pc_scores(:,g1_indices) pc_scores(:,nong1_indices)]';
svm_labels = [1 * ones(length(g1_indices),1);
	      -1 * ones(length(nong1_indices),1)];


latestSVM = svml('latestSVM','Kernel',1,'KernelParam',3,'C',1, ...
		 'ExecPath','/home/niche/matlab/toolboxes/svml');

for i=1:size(svm_data, 1)
  latestSVM = ...
      svmltrain(latestSVM, [svm_data([1:(i-1) (i+1):end], :)], ...
		svm_labels([1:(i-1) (i+1):end]));
  ypred(i) = svmlfwd(latestSVM, svm_data(i,:), svm_labels(i));
end
