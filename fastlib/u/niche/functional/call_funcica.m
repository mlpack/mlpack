N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);


myfd_data = data2fd(data, t, mybasis);
data_coef = getcoef(myfd_data);

cut_fraction = .5;


cut = round(cut_fraction * N);
myfd_data_coef = getcoef(myfd_data);

num_tests = 30;



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
    shuffled_indices = shuffle(indices);
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
    
    [ic_curves_pos, ic_coef_pos, h_Y_pos, pc_coef, pc_curves] = ...
	funcica(t, s, myfd_data_train, p, basis_curves, myfdPar);
    
    inv_pc_coef = inv(pc_coef);
    
    data_test_coef = getcoef(myfd_data_test);
    
    test_score = inv_pc_coef * data_test_coef;
    
    test_score_normalized = zeros(size(test_score));
    test_score_normalized(1,:) = test_score(1,:) / std(test_score(1,:));
    test_score_normalized(2,:) = test_score(2,:) / std(test_score(2,:));
   
    entropies1(i) = ...
	get_vasicek_entropy_estimate(test_score_normalized(1,:));
    entropies2(i) = ...
	get_vasicek_entropy_estimate(test_score_normalized(2,:));
    
    joint_entropies(i) = vasicek_sum(test_score_normalized(1:2,:));
    
  end
  
  lambda_set(j) = lambda;
  entropies1_set(:,j) = entropies1;
  entropies2_set(:,j) = entropies2;
  joint_entropies_set(:,j) = joint_entropies;
end