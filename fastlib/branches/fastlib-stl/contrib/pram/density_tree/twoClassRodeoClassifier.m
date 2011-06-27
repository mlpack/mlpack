function [acc] = twoClassRodeoClassifier(data, class0, class1)

[n d] = size(data);

class0_ind = find(data(:, d) == class0);
class1_ind = find(data(:, d) == class1);

class0 = data(class0_ind, 1:d-1);
class1 = data(class1_ind, 1:d-1);

length(class0_ind)
length(class1_ind)

test_set_ind = 1:3:length(class0_ind);
train_set_ind = setdiff(1:1:length(class0_ind), test_set_ind);

test0 = class0(test_set_ind, :);
train0 = class0(train_set_ind, :);

clear test_set_ind;
clear train_set_ind;

test_set_ind = 1:3:length(class1_ind);
train_set_ind = setdiff(1:1:length(class1_ind), test_set_ind);

test1 = class1(test_set_ind, :);
train1 = class1(train_set_ind, :);

p0 = size(train0, 1) / (size(train0, 1) + size(train1, 1));
p1 = size(train1, 1) / (size(train0, 1) + size(train1, 1));

d00 = local_rodeo(train0, test0);
d01 = local_rodeo(train1, test0);

d10 = local_rodeo(train0, test1);
d11 = local_rodeo(train1, test1);


dp0 = [d00' * p0 ; d01' * p1]';
dp1 = [d11' * p1 ; d10' * p0]';

c0 = 0;
for i = 1:size(test0, 1)
  if dp0(i,1) > dp0(i,2)
    c0 = c0 + 1;
  end
end

c1 = 0;
for i = 1:size(test1, 1)
  if dp1(i,1) > dp1(i,2)
    c1 = c1 + 1;
  end
end

acc = (c0 + c1) / (size(test0,1) + size(test1,1));
