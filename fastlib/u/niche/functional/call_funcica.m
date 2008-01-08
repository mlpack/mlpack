N = size(data, 2);
p = 30; % hardcoded for now

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);

lambda = 0;
myfdPar = fdPar(mybasis, 2, lambda);

myfd_data = data2fd(data, t, mybasis);
data_coef = getcoef(myfd_data);

cut_fraction = .9;

indices = 1:N;
cut = round(.9 * N);
myfd_data_coef = getcoef(myfd_data);



num_tests = 30;
joint_entropies = zeros(1,num_tests);

for i = 1:num_tests
  
  disp(sprintf('TEST %d', i));

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
  
  joint_entropies(i) = vasicek_sum(test_score_normalized(1:2,:));
  
end

joint_entropies