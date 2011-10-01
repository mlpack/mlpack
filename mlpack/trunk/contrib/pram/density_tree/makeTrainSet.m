function makeTrainSet(data, classes, file_prefix)

[n d] = size(data);

for i = 1:length(classes)
  class_ind = find(data(:, d) == classes(i));

  class = data(class_ind, 1:d-1);
  length(class_ind)

  train_file = sprintf('%s%d.csv', file_prefix, classes(i));

  csvwrite(train_file, class);
end
