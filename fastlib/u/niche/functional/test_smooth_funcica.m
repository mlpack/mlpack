%function [h_Y1_train_set, h_Y2_train_set, h_Y_train_set, h_Y1_test_set, ...
%	  h_Y2_test_set, h_Y_test_set, ...
%	  h_P1_train_set, h_P2_train_set, h_P_train_set, h_P1_test_set, ...
%	  h_P2_test_set, h_P_test_set] = ...
%    test_smooth_funcica(data, t, s, basis_inner_products);

N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);

cut_fraction = .5;

cut = round(cut_fraction * N);
data_coef = getcoef(myfd_data);

num_tests = 1;


% a simple method for smoothing
%indices = 1:200:1000;
%data = data(indices,:);


%lambda_set = 0;
lambda_set = [0 1e-6 1e-5 1e-4 1e-3 5e-3 1e-2];

myfdPar_set = cell(1,length(lambda_set));
for lambda_i = 1:length(lambda_set)
  myfdPar_set{lambda_i} = fdPar(mybasis, 2, lambda_set(lambda_i));
end
  

for test_num = 1:num_tests
    
  disp(sprintf('TEST %d', test_num));
  
  % generate random train and test indices
  indices = 1:N;
  shuffled_indices = shuffle(indices);
  rand_train_indices = sort(shuffled_indices(1:cut));
  rand_test_indices = sort(shuffled_indices((cut+1):end));
  
  % extract and center train data
  data_train = center(data(:, rand_train_indices));
  myfd_data_train = data2fd(data_train, t, mybasis);
  data_train_coef = getcoef(myfd_data_train);
  
  % extract and center test data
  data_test = center(data(:, rand_test_indices));
  myfd_data_test = data2fd(data_test, t, mybasis);
  data_test_coef = getcoef(myfd_data_test);
    
  % alternate way of generating train and test data
  % not used because no centering
  %{
  data_train_coef = data_coef(:,rand_train_indices);
  data_test_coef = data_coef(:,rand_test_indices);
  myfd_data_train = ...
      fd(data_train_coef, getbasis(myfd_data), getnames(myfd_data));
  myfd_data_test = ...
      fd(data_test_coef, getbasis(myfd_data), getnames(myfd_data));
  %}
  
  
  for lambda_i = 1:length(lambda_set)
    
    lambda = lambda_set(lambda_i);
    disp(sprintf('LAMBDA = %.4f', lambda));
    myfdPar = myfdPar_set{lambda_i};
    
    
    
    [ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, pc_scores, mean_coef, W, whitening_transform] = ...
	funcica(t, s, myfd_data_train, p, basis_curves, myfdPar, ...
		basis_inner_products);
    
    
    size(pc_scores)
    
    p_small = size(ic_coef_pos, 2);
    
    pc_curves = basis_curves * pc_coef;
    
    ic_curves = pc_curves(:,1:2) * W';
    
    pc_coef = pc_coef';
    
    
    
    
    
    % rescale the utilized parts of pc_coef such that
    % the pc_curves square integrate to 1
    
    for j = 1:p_small
      pc_coef_j = pc_coef(j,:);
      alpha = sqrt(sum(sum((pc_coef_j' * pc_coef_j) .* basis_inner_products)));
      pc_coef(j,:) = pc_coef(j,:) / alpha;
    end
    
    scores_train = pc_scores';
    scores_test = ...
	get_scores(data_test_coef, ...
		   pc_coef(1:p_small,:), ...
		   basis_inner_products)';
    
    Y_scores_train = W * scores_train(1:2,:);
    Y_scores_test = W * scores_test(1:2,:);
    

    magnitude_train_set(test_num, lambda_i) = ...
	sum(sum(scores_train .^ 2));
    
    magnitude_test_set(test_num, lambda_i) = ...
	sum(sum(scores_test .^ 2));
    
    
    
    h_P1_train = ...
	get_vasicek_entropy_estimate_std(scores_train(1,:));
    h_P2_train = ...
	get_vasicek_entropy_estimate_std(scores_train(2,:));
    h_P1_train_set(test_num, lambda_i) = h_P1_train;
    h_P2_train_set(test_num, lambda_i) = h_P2_train;
    h_P_train_set(test_num, lambda_i) = h_P1_train + h_P2_train;
    
    h_P1_test = ...
	get_vasicek_entropy_estimate_std(scores_test(1,:));
    h_P2_test = ...
	get_vasicek_entropy_estimate_std(scores_test(2,:));
    h_P1_test_set(test_num, lambda_i) = h_P1_test;
    h_P2_test_set(test_num, lambda_i) = h_P2_test;
    h_P_test_set(test_num, lambda_i) = h_P1_test + h_P2_test;

    
    h_Y1_train = ...
	get_vasicek_entropy_estimate_std(Y_scores_train(1,:));
    h_Y2_train = ...
	get_vasicek_entropy_estimate_std(Y_scores_train(2, :));
    h_Y1_train_set(test_num, lambda_i) = h_Y1_train;
    h_Y2_train_set(test_num, lambda_i) = h_Y2_train;
    h_Y_train_set(test_num, lambda_i) = h_Y1_train + h_Y2_train;

    h_Y1_test = ...
	get_vasicek_entropy_estimate_std(Y_scores_test(1,:));
    h_Y2_test = ...
	get_vasicek_entropy_estimate_std(Y_scores_test(2,:));
    h_Y1_test_set(test_num, lambda_i) = h_Y1_test;
    h_Y2_test_set(test_num, lambda_i) = h_Y2_test;
    h_Y_test_set(test_num, lambda_i) = h_Y1_test + h_Y2_test;
    
  end
end
