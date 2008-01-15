function [h_Y1_train_set, h_Y2_train_set, h_Y_train_set, h_Y1_test_set, ...
	  h_Y2_test_set, h_Y_test_set] = test_smooth_funcica(data, t, s);

N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);
pen = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);

cut_fraction = .5;

cut = round(cut_fraction * N);
myfd_data_coef = getcoef(myfd_data);

num_tests = 100;


%indices = 1:200:1000;
%data = data(indices,:);


lambda_set = [0 1e-4 1e-3 5e-3 1e-2];

for lambda_i = 1:5

  lambda = lambda_set(lambda_i);
  disp(sprintf('LAMBDA = %.4f', lambda));
  myfdPar = fdPar(mybasis, 2, lambda);
  
  
  entropies1_set = zeros(num_tests,1);
  entropies2_set = zeros(num_tests,1);
  joint_entropies = zeros(num_tests,1);
  
  for test_num = 1:num_tests
    
    disp(sprintf('TEST %d', test_num));
    
    indices = 1:N;
    shuffled_indices = shuffle(indices);
    rand_train_indices = sort(shuffled_indices(1:cut));
    rand_test_indices = sort(shuffled_indices((cut+1):end));
    
    myfd_data_train_coef = myfd_data_coef(:,rand_train_indices);
    myfd_data_test_coef = myfd_data_coef(:,rand_test_indices);
    myfd_data_train = ...
	fd(myfd_data_train_coef, getbasis(myfd_data), getnames(myfd_data));
    myfd_data_test = ...
	fd(myfd_data_test_coef, getbasis(myfd_data), getnames(myfd_data));
    
    [ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, pc_scores, mean_coef, W, whitening_transform] = ...
	funcica(t, s, myfd_data_train, p, basis_curves, myfdPar);
    
    p_small = size(ic_coef_pos, 2);
    
    my_data_train_coef = myfd_data_train_coef';
    my_data_test_coef = myfd_data_test_coef';
    pc_coef = pc_coef';
    scores_train = zeros(cut, p_small);
    scores_test = zeros(N - cut, p_small);
    
    for j = 1:p_small
      pc_coef_j = pc_coef(j,:);
      for i = 1:cut
	scores_train(i,j) = sum(sum((myfd_data_train_coef(:,i) * ...
				     pc_coef_j) .* pen));
	
      end
    
      for i = 1:(N - cut)
	scores_test(i,j) = sum(sum((myfd_data_test_coef(:,i) * ...
				    pc_coef_j) .* pen));
      end
    end
  
    scores_train = scores_train';
    scores_test = scores_test';
    Y_scores_train = W * scores_train(1:2,:);
    Y_scores_test = W * scores_test(1:2,:);
    
    h_Y1_train = ...
	get_vasicek_entropy_estimate_std(Y_scores_train(1,:));
    h_Y2_train = ...
	get_vasicek_entropy_estimate_std(Y_scores_train(2, :));
    
    h_Y1_test = ...
	get_vasicek_entropy_estimate_std(Y_scores_test(1,:));
    h_Y2_test = ...
	get_vasicek_entropy_estimate_std(Y_scores_test(2,:));
    
    h_Y1_train_set(test_num, lambda_i) = h_Y1_train;
    h_Y2_train_set(test_num, lambda_i) = h_Y2_train;
    h_Y_train_set(test_num, lambda_i) = h_Y1_train + h_Y2_train;
    
    h_Y1_test_set(test_num, lambda_i) = h_Y1_test;
    h_Y2_test_set(test_num, lambda_i) = h_Y2_test;
    h_Y_test_set(test_num, lambda_i) = h_Y1_test + h_Y2_test;
    
  end
end
