% test script

c3 = 14;
c4 = 18;

argvals = .001:.001:.5;

mybasis = create_bspline_basis([0 .5], 30, 4);
basis_curves = eval_basis(argvals, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

load /home/niche/neurofunk/bci_comp_2003_dataSet_IV/sp1s_aa_1000Hz;

x_train = CAR(x_train);
ground_truth = y_train;

num_train = 216;
% num_train = 316;

y_train = squeeze(x_train(:,c3,1:num_train)) - squeeze(x_train(:,c4,1:num_train));
y_test = squeeze(x_train(:,c3,217:end)) - squeeze(x_train(:,c4,217:end));

y_train_mean = mean(y_train,2);

y_train = y_train - repmat(y_train_mean, [1 num_train]);
y_test = y_test - repmat(y_train_mean, [1 100]);

% y_shifted = x_train_c3_shifted;

% y is num samples (500) by num epochs (318)

myfd_train = data2fd(y_train, argvals, mybasis);
myfd_test = data2fd(y_test, argvals, mybasis);
%myfd_shifted = data2fd(y_shifted, argvals, mybasis);

myfdPar = fdPar(mybasis, 2, 1e-6);

[ic_curves, ic_coef, Y, pc_coef, pc_curves, pc_scores, W] = ...
    funcica(argvals, myfd_train, 30, basis_curves, myfdPar, ...
	    basis_inner_products);


p_small = size(W, 1);

%pc_curves = basis_curves * pc_coef;
%ic_curves = basis_curves * ic_coef_pos;

pc_coef = pc_coef';

scores_train = get_scores(getcoef(myfd_train), pc_coef(1:p_small,:), ...
					basis_inner_products)';
scores_test = get_scores(getcoef(myfd_test), pc_coef(1:p_small,:), ...
				       basis_inner_products)';

Y_scores_train = W * scores_train;
Y_scores_test = W * scores_test;


scores_train = scores_train';
scores_test = scores_test';
Y_scores_train = Y_scores_train';
Y_scores_test = Y_scores_test';



lefts_train = find(ground_truth(1:num_train) == 0);
rights_train = find(ground_truth(1:num_train) == 1);

lefts_test = find(ground_truth((num_train+1):end) == 0);
rights_test = find(ground_truth((num_train+1):end) == 1);


used_scores_train = Y_scores_train;
used_scores_test = Y_scores_test;

for pc = 1:p_small

  lmean = mean(used_scores_train(lefts_train, pc));
  rmean = mean(used_scores_train(rights_train, pc));
  
  
  left_correct = ...
      abs(used_scores_test(lefts_test, pc) - lmean) < ...
      abs(used_scores_test(lefts_test, pc) - rmean);
  
  right_correct = ...
      abs(used_scores_test(rights_test, pc) - rmean) < ...
      abs(used_scores_test(rights_test, pc) - lmean);

  fprintf('pc = %d\n', pc);
  [sum(left_correct) / length(left_correct)
   sum(right_correct) / length(right_correct)]
end

