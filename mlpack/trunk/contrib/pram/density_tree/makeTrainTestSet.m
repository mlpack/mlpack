function makeTrainTestSet(data, classes, split_frac, file_prefix)

[n d] = size(data);

for i = 1:length(classes)
  class_ind = find(data(:, d) == classes(i));

  class = data(class_ind, 1:d-1);
  length(class_ind)

  test_set_ind = 1:split_frac:length(class_ind);
  train_set_ind = setdiff(1:1:length(class_ind), test_set_ind);

  test = class(test_set_ind, :);
  train = class(train_set_ind, :);

  test_file = sprintf('%s%d_test.csv', file_prefix, classes(i));
  train_file = sprintf('%s%d_train.csv', file_prefix, classes(i));

  csvwrite(test_file, test);
  csvwrite(train_file, train);

  clear test_set_ind;
  clear train_set_ind;

end
