N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);


myfd_data = data2fd(data, t, mybasis);

cut_fraction = .5;


cut = round(cut_fraction * N);
myfd_data_coef = getcoef(myfd_data);

num_tests = 1;


%indices = 1:200:1000;
%data = data(indices,:);




for j = 1:5

  if j == 1
    lambda = 0;
  elseif j == 2
    lambda = 1e-4;
  elseif j == 3
    lambda = 1e-3;
  elseif j == 4
    lambda = 5e-3;
  elseif j == 5
    lambda = 1e-2;
  else
    disp(sprintf('invalid j, j = %d', j));
    break;
  end
  disp(sprintf('LAMBDA = %.4f', lambda));
  myfdPar = fdPar(mybasis, 2, lambda);
  
  
  entropies1_set = zeros(num_tests,1);
  entropies2_set = zeros(num_tests,1);
  joint_entropies = zeros(num_tests,1);
  
  for i = 1:num_tests
    
    disp(sprintf('TEST %d', i));
    
    indices = 1:N;
    shuffled_indices = indices;%shuffle(indices);
    rand_train_indices = sort(shuffled_indices(1:cut));
    rand_test_indices = sort(shuffled_indices((cut+1):end));
    %train_data = data(:, rand_train_indices);
    %test_data =  data(:, rand_test_indices);
    
    myfd_data_train_coef = myfd_data_coef(:,rand_train_indices);
    myfd_data_test_coef = myfd_data_coef(:,rand_test_indices);
    myfd_data_train = ...
	fd(myfd_data_train_coef, getbasis(myfd_data), getnames(myfd_data));
    myfd_data_test = ...
	fd(myfd_data_test_coef, getbasis(myfd_data), getnames(myfd_data));
    
    [ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, pc_scores, mean_coef, W, whitening_transform] = ...
	funcica(t, s, myfd_data_train, p, basis_curves, myfdPar);
    
    inv_pc_coef = inv(pc_coef);
    
    train_pc_score = inv_pc_coef * (myfd_data_train_coef - ...
				    repmat(mean_coef, 1, cut));
    
    % does this make sense? I think we should instead subtract the
    % mean of the test data - the point is just to center the data
    test_pc_score = inv_pc_coef * (myfd_data_test_coef - ...
				   repmat(mean_coef, 1, N - cut));
    
    train_sub_pc_score = train_pc_score(1:2,:);
    test_sub_pc_score = test_pc_score(1:2,:);

    train_ic_score = W * train_sub_pc_score;
    test_ic_score = W * test_sub_pc_score;
    

    train_entropies1(i) = ...
	get_vasicek_entropy_estimate_std(train_ic_score(1,:));
    train_entropies2(i) = ...
	get_vasicek_entropy_estimate_std(train_ic_score(2,:));
    
    train_joint_entropies(i) = ...
	train_entropies1(i) + train_entropies2(i);

    
    test_entropies1(i) = ...
	get_vasicek_entropy_estimate_std(test_ic_score(1,:));
    test_entropies2(i) = ...
	get_vasicek_entropy_estimate_std(test_ic_score(2,:));
    
    
    % check magnitudes of projection against pc's
    magnitude(1) = ...
	dot(train_sub_pc_score(1,:), train_sub_pc_score(1,:));
    magnitude(2) = ...
	dot(train_sub_pc_score(2,:), train_sub_pc_score(2,:));
    magnitude
    
    test_joint_entropies(i) = ...
	test_entropies1(i) + test_entropies2(i);
    
    
  end
  
  
  lambda_set(j) = lambda;
  train_entropies1_set(:,j) = train_entropies1;
  train_entropies2_set(:,j) = train_entropies2;
  train_joint_entropies_set(:,j) = train_joint_entropies;
  
  test_entropies1_set(:,j) = test_entropies1;
  test_entropies2_set(:,j) = test_entropies2;
  test_joint_entropies_set(:,j) = test_joint_entropies;

  
end
